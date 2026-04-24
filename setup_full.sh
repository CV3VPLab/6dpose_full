set -e

ENV_NAME="6dpose_test"
REPO_DIR="$HOME/6dpose_pipeline"
SAM2_DIR="$REPO_DIR/sam2_repo"

CONDA_BASE="$(conda info --base)"
PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"
PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"

echo "============================================================"
echo "  ENV    : $ENV_NAME  (Python 3.10)"
echo "  torch  : 2.5.1+cu118"
echo "  REPO   : $REPO_DIR"
echo "============================================================"

# ── 1. conda env ──────────────────────────────────────────────────────────────
echo ""
echo ">>> [1/7] Creating conda env '$ENV_NAME' ..."
conda create -n "$ENV_NAME" python=3.10 -y

# ── 2. PyTorch ────────────────────────────────────────────────────────────────
echo ""
echo ">>> [2/7] PyTorch 2.5.1+cu118 ..."
"$PIP" install \
    torch==2.5.1+cu118 \
    torchvision==0.20.1+cu118 \
    torchaudio==2.5.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ── 3. Core packages ──────────────────────────────────────────────────────────
echo ""
echo ">>> [3/7] Core packages ..."
"$PIP" install \
    numpy==1.26.4 \
    scipy \
    h5py \
    opencv-python \
    pillow \
    matplotlib \
    tqdm \
    plyfile \
    joblib \
    lpips \
    einops \
    imageio \
    ninja

# ── 4. 6dpose packages ────────────────────────────────────────────────────────
echo ""
echo ">>> [4/7] 6dpose packages ..."
"$PIP" install \
    ultralytics==8.4.37 \
    kornia==0.8.2 \
    timm==1.0.26 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    iopath \
    portalocker \
    transformers \
    huggingface_hub \
    safetensors \
    tokenizers \
    psutil \
    polars \
    typer \
    rich

# ── 5. SAM2 (git clone → editable install) ────────────────────────────────────
echo ""
echo ">>> [5/7] SAM2 ..."
if [ ! -d "$SAM2_DIR" ]; then
    git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
else
    echo "    sam2_repo already exists, skipping clone"
fi
"$PIP" install -e "$SAM2_DIR"

# ── 6. gsplat + GS CUDA submodules ───────────────────────────────────────────
echo ""
echo ">>> [6/7] gsplat + GS CUDA submodules ..."
export TORCH_CUDA_ARCH_LIST="8.6"
"$PIP" install --upgrade setuptools wheel

# gsplat: PyPI 패키지 — 경로 불필요
echo "    gsplat 1.5.3 ..."
"$PIP" install gsplat==1.5.3

# CUDA submodules: 항상 새로 clone 후 build
mkdir -p "$REPO_DIR/submodules"

echo "    Removing old GS CUDA submodules ..."
rm -rf "$REPO_DIR/submodules/diff-gaussian-rasterization"
rm -rf "$REPO_DIR/submodules/simple-knn"
rm -rf "$REPO_DIR/submodules/fused-ssim"

echo "    diff-gaussian-rasterization ..."
git clone --recursive --branch dr_aa \
    https://github.com/graphdeco-inria/diff-gaussian-rasterization.git \
    "$REPO_DIR/submodules/diff-gaussian-rasterization"

cd "$REPO_DIR/submodules/diff-gaussian-rasterization"
git submodule update --init --recursive

if [ ! -f "third_party/glm/glm/glm.hpp" ]; then
    echo "[ERROR] GLM not found: third_party/glm/glm/glm.hpp"
    echo "        diff-gaussian-rasterization submodule was not initialized correctly."
    exit 1
fi

rm -rf build *.egg-info
"$PIP" install --no-build-isolation -e .


echo "    simple-knn ..."
git clone \
    https://gitlab.inria.fr/bkerbl/simple-knn.git \
    "$REPO_DIR/submodules/simple-knn"

cd "$REPO_DIR/submodules/simple-knn"
rm -rf build *.egg-info
"$PIP" install --no-build-isolation -e .


echo "    fused-ssim ..."
git clone \
    https://github.com/rahul-goel/fused-ssim.git \
    "$REPO_DIR/submodules/fused-ssim"

cd "$REPO_DIR/submodules/fused-ssim"
rm -rf build *.egg-info
"$PIP" install --no-build-isolation -e .

cd "$REPO_DIR"

# ── 7. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> [7/7] Verifying ..."
"$PYTHON" - <<'PYEOF'
import sys
print(f"Python  : {sys.version.split()[0]}")

import torch
print(f"PyTorch : {torch.__version__}  CUDA={torch.version.cuda}  GPU={torch.cuda.is_available()}")

import cv2;          print(f"OpenCV  : {cv2.__version__}")
import ultralytics;  print(f"YOLO    : {ultralytics.__version__}")
import kornia;       print(f"Kornia  : {kornia.__version__}")
import timm;         print(f"timm    : {timm.__version__}")
import sam2;         print(f"SAM2    : OK")
import gsplat;       print(f"gsplat  : {gsplat.__version__}")
import scipy;        print(f"scipy   : {scipy.__version__}")
import h5py;         print(f"h5py    : {h5py.__version__}")
import lpips;        print(f"lpips   : {lpips.__version__}")

from diff_gaussian_rasterization import GaussianRasterizer
print("diff-gaussian-rasterization : OK")
import simple_knn;   print("simple-knn : OK")
import fused_ssim;   print("fused-ssim : OK")

print("\n============================================================")
print(f"  All checks passed — conda activate {sys.prefix.split('/')[-1]}")
print("============================================================")
PYEOF

echo ""
echo "NOTE: pyorbbecsdk2 (Orbbec depth camera) is hardware-specific."
echo "      pip install ~/Downloads/pyorbbecsdk2-*-cp310-cp310-linux_x86_64.whl"
