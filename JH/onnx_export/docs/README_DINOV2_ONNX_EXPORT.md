# DINOv2 Dynamic Batch ONNX Export Guide

## 개요

Hugging Face DINOv2 모델을 retrieval용 Dynamic Batch ONNX로 생성하는 방법을
설명합니다.

사용 모델:

```text
model: facebook/dinov2-small
revision: ed25f3a31f01632728cabb09d1542f84ab7b0056
input size: 224×224
patch size: 14
hidden size: 384
ONNX opset: 17
```

Revision을 고정해 이후 Hugging Face 모델 변경으로 export 결과가 달라지는 것을
방지합니다.

## 필요한 파일

```text
export repository/
└── scripts/
    └── export_dinov2_dynamic_onnx.py
```

Checkpoint와 생성된 ONNX는 Git에 포함하지 않고 Hugging Face cache 또는 NAS에
보관합니다.

기존 B=1 ONNX와 결과 호환성까지 확인하려면 baseline 파일도 준비합니다.

```text
/mnt/nas/.../dino_vits14_224.onnx
```

## 환경 구성

```bash
conda create -n dino_onnx_export python=3.10 -y
conda activate dino_onnx_export

pip install torch torchvision
pip install transformers onnx==1.17.0 onnxruntime==1.18.0
```

ONNX simplification 옵션을 사용한다면 다음 패키지도 설치합니다.

```bash
pip install onnxsim==0.4.36
```

환경 확인:

```bash
python -c "import torch, transformers, onnx, onnxruntime; print(torch.__version__, transformers.__version__, onnx.__version__, onnxruntime.__version__)"
```

## ONNX 범위

ONNX에는 DINOv2 backbone만 포함됩니다.

포함:

- DINOv2 patch embedding 및 transformer 연산
- `last_hidden_state`
- CLS token 기반 `pooler_output`

포함하지 않음:

- RGB 이미지 decode
- `AutoImageProcessor` resize/center crop
- rescale 및 mean/std normalization
- feature L2 normalization
- gallery feature score 및 top-k

따라서 ONNX 입력은 전처리가 완료된 tensor여야 합니다.

## 입출력 형식

```text
input
pixel_values:       float32 [B, 3, 224, 224]

outputs
last_hidden_state:  float32 [B, 257, 384]
pooler_output:       float32 [B, 384]
```

Batch 축만 dynamic입니다. Height와 width는 224로 고정됩니다.

Retrieval에서는 일반적으로 `pooler_output`을 사용한 뒤 마지막 차원에 L2
normalization을 적용합니다.

```python
feature = torch.nn.functional.normalize(pooler_output, dim=-1)
```

## Hugging Face model 준비

### 온라인 환경

처음 실행할 때 model과 config가 Hugging Face cache에 다운로드됩니다.

```bash
huggingface-cli download \
  facebook/dinov2-small \
  --revision ed25f3a31f01632728cabb09d1542f84ab7b0056
```

### 오프라인 환경

인터넷이 가능한 장비에서 cache를 준비한 뒤 export 장비로 복사합니다.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

오프라인 export 명령에는 `--local_files_only`를 추가합니다. Cache에 지정 revision이
없으면 export가 실패합니다.

## 기본 export

기존 B=1 ONNX가 NAS에 있는 경우:

```bash
python scripts/export_dinov2_dynamic_onnx.py \
  --model facebook/dinov2-small \
  --revision ed25f3a31f01632728cabb09d1542f84ab7b0056 \
  --height 224 \
  --width 224 \
  --trace_batch_size 4 \
  --opset 17 \
  --baseline /mnt/nas/path/to/dino_vits14_224.onnx \
  --validate_batches 1,4 \
  --out outputs/dino_vits14_224_dynamic_batch.onnx
```

스크립트가 수행하는 작업:

1. 지정된 revision의 DINOv2 model 로드
2. B=4 dummy tensor로 graph trace
3. input/output batch 축을 symbolic `batch`로 지정
4. ONNX checker 실행
5. B=1/B=4에서 PyTorch와 ONNXRuntime CPU 결과 비교
6. B=1에서 기존 baseline ONNX와 결과 비교
7. model 정보와 전처리 범위를 ONNX metadata로 기록

## 오프라인 export

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/export_dinov2_dynamic_onnx.py \
  --trace_batch_size 4 \
  --local_files_only \
  --baseline /mnt/nas/path/to/dino_vits14_224.onnx \
  --validate_batches 1,4 \
  --out outputs/dino_vits14_224_dynamic_batch.onnx
```

## Baseline 없이 export

기존 고정 B=1 ONNX가 없다면 빈 문자열을 전달해 baseline 비교만 생략합니다.
PyTorch와 새 ONNX의 B=1/B=4 비교는 계속 수행됩니다.

```bash
python scripts/export_dinov2_dynamic_onnx.py \
  --trace_batch_size 4 \
  --baseline "" \
  --validate_batches 1,4 \
  --out outputs/dino_vits14_224_dynamic_batch.onnx
```

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model` | `facebook/dinov2-small` | Hugging Face model ID |
| `--revision` | 고정 commit | 사용할 model revision |
| `--height` | `224` | ONNX 입력 높이 |
| `--width` | `224` | ONNX 입력 너비 |
| `--trace_batch_size` | `4` | export trace에 사용할 예제 B |
| `--opset` | `17` | ONNX opset |
| `--baseline` | 기존 B=1 ONNX 경로 | 호환성 비교 대상 |
| `--validate_batches` | `1,4` | ONNXRuntime 검증 batch 목록 |
| `--atol` | `1e-4` | 절대 오차 허용치 |
| `--rtol` | `1e-4` | 상대 오차 허용치 |
| `--local_files_only` | off | Hugging Face local cache만 사용 |
| `--simplify` | off | onnxsim simplification 수행 |
| `--skip_validation` | off | 수치 검증 생략 |

`--trace_batch_size 4`는 ONNX의 최대 batch를 4로 제한하지 않습니다. 생성된
ONNX의 batch 축은 dynamic입니다. TensorRT의 실제 허용 범위는 runtime에서
별도로 설정하는 optimization profile에 의해 결정됩니다.

## 입력 크기 변경

DINOv2-small의 patch size는 14이므로 H/W는 14로 나누어져야 합니다. 현재
retrieval runtime과 gallery feature는 224×224 전처리를 기준으로 생성되어
있으므로 특별한 이유가 없다면 224를 유지합니다.

입력 크기를 바꾸면 token 수가 달라집니다.

```text
tokens = (H / 14) * (W / 14) + 1
```

입력 크기 변경 시 query와 gallery feature를 같은 전처리 조건으로 다시 생성해야
합니다.

## ONNX simplification

필요한 경우 `--simplify`를 추가합니다.

```bash
python scripts/export_dinov2_dynamic_onnx.py \
  --trace_batch_size 4 \
  --baseline "" \
  --simplify \
  --out outputs/dino_vits14_224_dynamic_batch.onnx
```

Simplification 전후에는 dynamic batch 축이 유지되는지 다시 확인해야 합니다.

## ONNX 구조 확인

```bash
python - <<'PY'
import onnx

path = "outputs/dino_vits14_224_dynamic_batch.onnx"
model = onnx.load(path)
onnx.checker.check_model(model)

def shape(value):
    return [
        d.dim_param if d.dim_param else d.dim_value
        for d in value.type.tensor_type.shape.dim
    ]

for value in model.graph.input:
    print("input ", value.name, shape(value))
for value in model.graph.output:
    print("output", value.name, shape(value))
print("metadata", {p.key: p.value for p in model.metadata_props})
PY
```

예상 결과:

```text
input  pixel_values [batch, 3, 224, 224]
output last_hidden_state [batch, 257, 384]
output pooler_output [batch, 384]
```

## ONNXRuntime smoke test

```bash
python - <<'PY'
import numpy as np
import onnxruntime as ort

path = "outputs/dino_vits14_224_dynamic_batch.onnx"
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

for batch in (1, 2, 4):
    sample = np.zeros((batch, 3, 224, 224), dtype=np.float32)
    hidden, pooled = session.run(None, {input_name: sample})
    print(f"B={batch}: hidden={hidden.shape}, pooled={pooled.shape}")
PY
```

## TensorRT 사용 시 profile

현재 retrieval runtime에서 B=1 단일 image와 B=4 tile batch를 모두 사용하려면
다음 profile을 사용합니다.

```text
min: pixel_values:1x3x224x224
opt: pixel_values:4x3x224x224
max: pixel_values:4x3x224x224
```

ONNX가 dynamic이어도 max보다 큰 B는 해당 TensorRT engine에 입력할 수
없습니다. Profile을 변경하면 기존 engine cache를 재사용하지 말고 새로
생성합니다.

## NAS 업로드

권장 파일명:

```text
dino_vits14_224_dynamic_batch.onnx
```

Checksum 생성:

```bash
sha256sum outputs/dino_vits14_224_dynamic_batch.onnx \
  > outputs/dino_vits14_224_dynamic_batch.onnx.sha256
```

현재 검증된 파일의 SHA-256:

```text
e8cb92d0cc18f0de99152883251083a711a2af60a185f86f17c00e72d876ce39
```

NAS에는 ONNX, 필요하면 Hugging Face snapshot, checksum을 보관하고 Git에는
export script와 README만 저장하는 구성을 권장합니다.

## 문제 해결

### Hugging Face model을 찾지 못하는 경우

- model ID와 revision을 확인합니다.
- 오프라인이면 지정 revision이 local cache에 있는지 확인합니다.
- `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `--local_files_only` 설정을 함께
  확인합니다.

### Baseline ONNX를 찾지 못하는 경우

- NAS mount와 `--baseline` 경로를 확인합니다.
- Baseline 비교가 필요 없으면 `--baseline ""`을 사용합니다.

### H/W validation 오류

- H/W가 patch size 14로 나누어지는지 확인합니다.
- Runtime과 gallery feature가 동일 입력 크기를 사용하는지 확인합니다.

### PyTorch와 ONNX 결과 비교 실패

- model revision이 동일한지 확인합니다.
- opset과 package 버전을 기록합니다.
- 우선 `--simplify` 없이 export합니다.
- 필요할 때만 `--atol`, `--rtol`을 조정하고 차이의 원인을 확인합니다.

### TensorRT에서 특정 batch가 실패하는 경우

- ONNX batch 축이 symbolic인지 확인합니다.
- TensorRT min/opt/max profile 범위를 확인합니다.
- 이전 profile로 생성된 engine cache를 제거하고 다시 생성합니다.
