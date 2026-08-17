# EDM ONNX Export Guide

## 개요

EDM checkpoint에서 다음 ONNX 모델을 생성하는 방법을 설명합니다.

- 단일 image pair 전용 고정 B=1 모델
- 정확히 두 image pair를 동시에 처리하는 고정 B=2 모델
- 실행 시 image pair 개수를 변경할 수 있는 dynamic batch 모델

기준 원본은 EDM 공식 저장소의 deploy 코드입니다.

- EDM: <https://github.com/chicleee/EDM>
- 공식 baseline: `EDM/deploy/export_onnx.py`

이 프로젝트에서는 원본의 단일 pair export를 유지하면서 batch 차원이 출력까지
보존되도록 별도 wrapper를 사용합니다.

## 필요한 파일

```text
export repository/
├── scripts/
│   ├── export_edm_pair_onnx.py
│   └── export_edm_batch_onnx.py
└── EDM_repo/                         # EDM 공식 저장소
    ├── configs/
    ├── src/
    └── ...
```

`export_edm_batch_onnx.py`는 같은 디렉터리의
`export_edm_pair_onnx.py`에서 `build_edm()`을 import하므로 두 파일을 함께
보관해야 합니다.

Checkpoint는 Git에 포함하지 않고 NAS 경로를 직접 전달합니다.

```text
/mnt/nas/.../edm_outdoor.ckpt
```

Checkpoint는 다음 구조여야 합니다.

```python
checkpoint["state_dict"]
```

## 환경 구성

EDM 공식 환경을 먼저 구성합니다.

```bash
git clone https://github.com/chicleee/EDM.git EDM_repo

conda create -n edm_export python=3.10 -y
conda activate edm_export

pip install torch torchvision
pip install onnx==1.17.0 onnxsim==0.4.36 onnxruntime==1.18.0
pip install -r EDM_repo/requirements.txt
```

GPU에서 ONNXRuntime/TensorRT 검증까지 수행하려면 CPU용 `onnxruntime` 대신
환경에 맞는 `onnxruntime-gpu`와 TensorRT를 설치합니다. ONNX export 자체와
ONNX checker는 CPU에서도 가능합니다.

환경 확인:

```bash
python -c "import torch, onnx, onnxsim; print(torch.__version__, onnx.__version__)"
```

## 입출력 형식

### 단일 pair 모델

```text
input:  float32 [1, 2, H, W]
output: float32 [K, 11]
```

### Batch 모델

```text
input:  float32 [B, 2, H, W]
output: float32 [B, K, 11]
```

입력의 두 채널은 하나의 matching pair입니다.

```text
input[b, 0] = query grayscale image
input[b, 1] = gallery grayscale image
```

입력 조건:

- dtype: `float32`
- value range: `[0, 1]`
- image format: grayscale
- shape: `[B, 2, H, W]`
- H/W는 32의 배수를 권장

출력의 마지막 11개 값은 EDM deploy 출력 구조를 유지합니다.

```text
[mkpts0_c, mkpts1_c, offset01, offset10, score01, score10, mconf]
```

`K`는 다음 식으로 정해집니다.

```text
K = int((H / 8) * (W / 8) * 0.35)
```

대표 입력 크기:

| 입력 H×W | K |
|---|---:|
| 672×672 | 2469 |
| 160×160 | 140 |

## B=1 baseline export

```bash
python scripts/export_edm_pair_onnx.py \
  --edm_repo EDM_repo \
  --ckpt /mnt/nas/path/to/edm_outdoor.ckpt \
  --height 672 \
  --width 672 \
  --out outputs/edm_outdoor_w672_h672_topk2469.onnx
```

생성 결과:

```text
input:  [1, 2, 672, 672]
output: [2469, 11]
```

ONNX simplification을 생략하려면 `--no_simplify`를 추가합니다.

## 고정 B=2 export

Stereo Left/Right의 두 pair를 항상 동시에 처리하는 경우 사용합니다.

```bash
python scripts/export_edm_batch_onnx.py \
  --edm_repo EDM_repo \
  --ckpt /mnt/nas/path/to/edm_outdoor.ckpt \
  --height 672 \
  --width 672 \
  --mode fixed \
  --batch_size 2 \
  --out outputs/edm_outdoor_w672_h672_topk2469_b2.onnx
```

생성 결과:

```text
input:  [2, 2, 672, 672]
output: [2, 2469, 11]
```

고정 B=2 모델에는 항상 pair 두 개를 입력해야 합니다. Pair가 하나뿐이면 동일
pair를 두 번 넣고 첫 번째 결과만 사용하는 방법은 가능하지만, B가 가변적인
서비스에는 dynamic 모델이 더 적합합니다.

## Dynamic batch export

`--batch_size`는 export trace에 사용하는 예제 B이며 ONNX의 고정 batch 크기가
아닙니다. 실제 ONNX input/output의 첫 축은 `batch`라는 symbolic dimension으로
생성됩니다.

672×672 dynamic 모델:

```bash
python scripts/export_edm_batch_onnx.py \
  --edm_repo EDM_repo \
  --ckpt /mnt/nas/path/to/edm_outdoor.ckpt \
  --height 672 \
  --width 672 \
  --mode dynamic \
  --batch_size 2 \
  --out outputs/edm_outdoor_w672_h672_topk2469_dynamic_batch.onnx
```

생성 결과:

```text
input:  [batch, 2, 672, 672]
output: [batch, 2469, 11]
```

160×160 dynamic 모델:

```bash
python scripts/export_edm_batch_onnx.py \
  --edm_repo EDM_repo \
  --ckpt /mnt/nas/path/to/edm_outdoor.ckpt \
  --height 160 \
  --width 160 \
  --mode dynamic \
  --batch_size 2 \
  --out outputs/edm_outdoor_w160_h160_topk140_dynamic_batch.onnx
```

Dynamic ONNX라도 TensorRT에서 허용되는 batch 범위는 engine 생성 시 설정하는
optimization profile에 의해 결정됩니다. 예를 들어 `min=1, opt=2, max=4`이면
ONNX는 dynamic이지만 해당 TensorRT engine에는 B=1~4만 입력할 수 있습니다.

## ONNX 구조 확인

```bash
python - <<'PY'
import onnx

path = "outputs/edm_outdoor_w672_h672_topk2469_dynamic_batch.onnx"
model = onnx.load(path)
onnx.checker.check_model(model)

def shape(value):
    return [
        d.dim_param if d.dim_param else d.dim_value
        for d in value.type.tensor_type.shape.dim
    ]

print("input :", model.graph.input[0].name, shape(model.graph.input[0]))
print("output:", model.graph.output[0].name, shape(model.graph.output[0]))
print("metadata:", {p.key: p.value for p in model.metadata_props})
PY
```

Batch export script는 다음 metadata를 ONNX에 기록합니다.

```text
edm.batch_mode
edm.trace_batch_size
edm.height
edm.width
edm.topk
edm.output_layout
```

## ONNXRuntime smoke test

```bash
python - <<'PY'
import numpy as np
import onnxruntime as ort

path = "outputs/edm_outdoor_w672_h672_topk2469_dynamic_batch.onnx"
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

for batch in (1, 2, 4):
    sample = np.zeros((batch, 2, 672, 672), dtype=np.float32)
    output = session.run(None, {input_name: sample})[0]
    print(f"B={batch}: {output.shape}")
PY
```

예상 출력:

```text
B=1: (1, 2469, 11)
B=2: (2, 2469, 11)
B=4: (4, 2469, 11)
```

## NAS 업로드

파일명에 입력 크기, top-k, batch 형식을 포함합니다.

```text
edm_outdoor_w672_h672_topk2469_b2.onnx
edm_outdoor_w672_h672_topk2469_dynamic_batch.onnx
edm_outdoor_w160_h160_topk140_dynamic_batch.onnx
```

업로드 전에 checksum을 생성합니다.

```bash
sha256sum outputs/*.onnx > outputs/SHA256SUMS
```

NAS에는 ONNX, checkpoint, `SHA256SUMS`를 함께 보관하고 Git에는 export script와
README만 저장하는 구성을 권장합니다.

## 문제 해결

### Checkpoint를 읽지 못하는 경우

- `--ckpt`가 실제 NAS mount 경로인지 확인합니다.
- checkpoint에 `state_dict` key가 있는지 확인합니다.
- outdoor checkpoint에는 outdoor EDM config를 사용해야 합니다.

### Output shape가 예상과 다른 경우

- H/W와 계산된 K를 확인합니다.
- Batch 모델은 wrapper가 출력을 `[B,K,11]`로 reshape하므로
  `export_edm_batch_onnx.py`를 사용했는지 확인합니다.

### Dynamic 모델이 특정 B에서 실패하는 경우

- ONNX input 첫 축이 symbolic `batch`인지 확인합니다.
- TensorRT를 사용한다면 min/opt/max profile 범위를 확인합니다.
- 새로운 profile을 사용할 때는 기존 TensorRT engine cache를 삭제하고 다시
  생성합니다.

### ONNX simplification이 실패하는 경우

`--no_simplify`로 먼저 원본 ONNX를 생성한 뒤 ONNX checker와 runtime 실행을
확인합니다. 이후 호환되는 `onnxsim` 버전에서 별도로 simplify합니다.
