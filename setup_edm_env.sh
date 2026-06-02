#!/usr/bin/env bash
# Source this before running the EDM/TRT pipeline:
#   source setup_edm_env.sh
#
# It only sets runtime library paths. It does not install packages.

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "[setup_edm_env] CONDA_PREFIX is empty. Activate the conda env first."
  return 1 2>/dev/null || exit 1
fi

_prepend_ld_path() {
  local p="$1"
  if [ -d "$p" ] && [[ ":${LD_LIBRARY_PATH:-}:" != *":$p:"* ]]; then
    export LD_LIBRARY_PATH="$p:${LD_LIBRARY_PATH:-}"
  fi
}

# TensorRT pip wheels usually place libnvinfer/libnvonnxparser here.
for p in "$CONDA_PREFIX"/lib/python*/site-packages/tensorrt_libs; do
  _prepend_ld_path "$p"
  [ -d "$p" ] && export TENSORRT_LIB_DIR="$p"
done

# onnxruntime-gpu 1.18 + CUDA 11.8 commonly needs cuDNN 8.
# Prefer an explicit env var if the user already set it.
if [ -n "${CUDNN_LIB_DIR:-}" ]; then
  _prepend_ld_path "$CUDNN_LIB_DIR"
else
  for p in \
    "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/cudnn/lib \
    "$(conda info --base 2>/dev/null)"/pkgs/pytorch-*_cuda11.8_cudnn8*/lib/python*/site-packages/torch/lib \
    "$(conda info --base 2>/dev/null)"/pkgs/cudnn-8*/lib
  do
    if [ -e "$p/libcudnn.so.8" ]; then
      _prepend_ld_path "$p"
      export CUDNN_LIB_DIR="$p"
      break
    fi
  done
fi

echo "[setup_edm_env] TENSORRT_LIB_DIR=${TENSORRT_LIB_DIR:-not found}"
echo "[setup_edm_env] CUDNN_LIB_DIR=${CUDNN_LIB_DIR:-not found}"
echo "[setup_edm_env] LD_LIBRARY_PATH updated."
