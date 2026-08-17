# JH DINOv2 Dynamic GPU Retrieval

## 목적

기존 DINOv2 retrieval의 고정 B=1 ONNX 호출을 Dynamic B=4로 변경하고,
이미지 전처리부터 feature 추출과 gallery score 계산까지 CUDA tensor를
사용하도록 개선한 버전입니다.

최종 적용 방식은 다음과 같습니다.

```text
Query crop / 4-tile 분할       CPU, NumPy
AutoImageProcessor             GPU, torch.Tensor
DINOv2 Dynamic ONNX B=4        GPU, ONNX Runtime I/O Binding
L2 normalization               GPU, PyTorch
Gallery feature                GPU 상주
Gallery score                  GPU, matrix multiplication
Top-k 결과 반환                CPU, NumPy
```

## 원본 대비 변경 파일

### `modules_6d/retrieval_dino.py`

- Dynamic TensorRT min/opt/max profile 지원
- `encode_batch_rgb()` 추가
- `encode_4rgb()`를 B=1 네 번에서 B=4 한 번으로 변경
- ONNX Runtime CUDA I/O Binding 추가
- `AutoImageProcessor(..., device="cuda:0")` 지원
- CUDA input/output tensor에서 L2 normalization 수행
- gallery feature를 최초 한 번 VRAM으로 이동하는 함수 추가
- PyTorch CUDA 전처리와 ONNX Runtime 사이에
  `io_binding.synchronize_inputs()` 적용

### `JH/estimate_object_pose.py`

- Dynamic TensorRT profile을 DINO session에 전달
- `use_io_binding`, `preprocess_on_gpu`, `device_id` 전달
- 실제 batch 크기 B=4로 warmup
- `retrieve_best()`와 `retrieve_topk()`에서 gallery feature의 device를
  query feature와 일치시킨 뒤 score 계산
- JH 전용 `JH/ope_config.json`을 읽도록 설정

### `JH/ope_config.json`

- Dynamic B=4 ONNX 사용
- TensorRT profile `min=1, opt=4, max=4`
- CUDA I/O Binding 활성화
- GPU AutoImageProcessor 활성화

## ONNX

사용 파일:

```text
weights/dino_vits14_224_dynamic_batch.onnx
```

입출력 형식:

```text
input
pixel_values:       float32 [B, 3, 224, 224]

outputs
last_hidden_state:  float32 [B, 257, 384]
pooler_output:       float32 [B, 384]
```

ONNX의 batch 축은 dynamic입니다. 현재 TensorRT profile에서는
`1 <= B <= 4`만 허용합니다.

생성에 사용한 model:

```text
facebook/dinov2-small
revision: ed25f3a31f01632728cabb09d1542f84ab7b0056
opset: 17
```

생성 스크립트:

```text
scripts/export_dinov2_dynamic_onnx.py
```

생성 명령:

```bash
python scripts/export_dinov2_dynamic_onnx.py \
  --trace_batch_size 4 \
  --local_files_only \
  --out weights/dino_vits14_224_dynamic_batch.onnx
```

생성된 ONNX SHA-256:

```text
e8cb92d0cc18f0de99152883251083a711a2af60a185f86f17c00e72d876ce39
```

## 최종 설정

`JH/ope_config.json`의 `feat_extractor` 설정은 다음과 같습니다.

```json
"feat_extractor": {
    "name": "DINOv2",
    "options": {
        "model": "dinov2_vits14",
        "input_size": 224,
        "onnx": "weights/dino_vits14_224_dynamic_batch.onnx",
        "trt_cache_dir": "weights/trt_cache_dino_dynamic_b4",
        "batch_size": 4,
        "use_io_binding": true,
        "preprocess_on_gpu": true,
        "device_id": 0,
        "trt_profile_min_shapes": "pixel_values:1x3x224x224",
        "trt_profile_opt_shapes": "pixel_values:4x3x224x224",
        "trt_profile_max_shapes": "pixel_values:4x3x224x224",
        "trt_max_workspace_size": 4294967296,
        "warmup": true,
        "warmup_runs": 1,
        "require_gpu": true
    }
}


## Gallery feature 처리

`g_features.npy`의 3,024개 gallery feature는 파일에서 CPU RAM으로 읽은 직후,
query 처리 전에 전체를 VRAM으로 이동합니다. 현재 `[3024, 1536]` float32
feature의 VRAM 사용량은 약 17.7 MiB이며, 실행 중 모든 query가 같은 CUDA
tensor를 재사용합니다.

```python
gallery_info = construct_galleryInfo(obj_dir)
move_dino_gallery_features_to_device(
    gallery_info,
    "cuda:0",
)
```

Retrieval에서는 이미 GPU에 있는 feature를 바로 사용합니다.

```python
scores = (g_feats @ qfeat).detach().cpu().numpy()
```

이 VRAM 상주는 현재 Python process가 실행되는 동안만 유지됩니다. 프로그램을
종료하면 해제되며, 다음 실행에서는 `g_features.npy`를 다시 읽어 VRAM으로
전송합니다. TensorRT의 disk engine cache와는 별개의 동작입니다.

## 주의사항

- TensorRT engine cache는 GPU, CUDA, TensorRT 및 ONNX Runtime 환경에
  종속됩니다. 환경이 바뀌면 cache를 다시 생성해야 합니다.
- `preprocess_on_gpu=true`에서는 CPU와 GPU의 resize 수치가 완전히 같지 않을 수
  있습니다. Feature가 유사해도 score가 가까운 top-k 경계의 gallery 순서는
  달라질 수 있으므로 최종 pose까지 확인해야 합니다.
- CUDA 전처리 tensor와 ONNX Runtime의 실행 stream을 맞추기 위해
  `io_binding.synchronize_inputs()`를 제거하면 안 됩니다.
- TensorRT profile의 max가 B=4이므로 한 번에 5장 이상 입력하려면 profile을
  변경하고 engine cache를 다시 생성해야 합니다.
- `weights/`는 원본 저장소의 `.gitignore` 대상입니다. ONNX를 GitHub에 포함할
  경우 Git LFS 또는 별도의 release asset을 사용합니다.
