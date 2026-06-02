"""Minimal EDM matcher integration.

This module intentionally contains only the pieces needed by
``estimate_object_pose.py``: model loading and pairwise matching.
"""

import sys
from pathlib import Path
import ctypes
import glob
import os

import cv2
import numpy as np
import torch


LOCAL_RESOLUTION = 8


def _prepend_library_path(path):
    path = str(path)
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in cur.split(":") if p]
    if path not in parts:
        os.environ["LD_LIBRARY_PATH"] = path + (":" + cur if cur else "")


def _try_preload_ort_gpu_libs():
    lib_dirs = []

    for env_name in ("TENSORRT_LIB_DIR", "CUDNN_LIB_DIR", "CUDA_LIB_DIR"):
        env_path = os.environ.get(env_name)
        if env_path:
            lib_dirs.append(Path(env_path))

    conda_prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix))
    site_patterns = [
        str(conda_prefix / "lib/python*/site-packages/tensorrt_libs"),
        str(conda_prefix / "lib/python*/site-packages/nvidia/cudnn/lib"),
        str(conda_prefix / "lib/python*/site-packages/torch/lib"),
    ]
    for pattern in site_patterns:
        lib_dirs.extend(Path(p) for p in glob.glob(pattern))

    conda_base = conda_prefix.parent.parent if conda_prefix.name != "envs" else conda_prefix
    pkgs_dir = conda_base / "pkgs"
    if pkgs_dir.exists():
        lib_dirs.extend(
            Path(p)
            for p in glob.glob(
                str(pkgs_dir / "pytorch-*_cuda11.8_cudnn8*" / "lib/python*/site-packages/torch/lib")
            )
        )
        lib_dirs.extend(
            Path(p)
            for p in glob.glob(str(pkgs_dir / "cudnn-8*" / "lib"))
        )

    lib_dirs = [p for p in dict.fromkeys(lib_dirs) if p.exists()]

    for lib_dir in lib_dirs:
        _prepend_library_path(lib_dir)

    # Best-effort preload so dlopen can resolve ORT provider dependencies even
    # when the user did not export LD_LIBRARY_PATH before launching Python.
    preload_names = [
        "libcudnn.so.8",
        "libcudnn_ops_infer.so.8",
        "libcudnn_cnn_infer.so.8",
        "libcudnn_adv_infer.so.8",
        "libnvinfer.so.10",
        "libnvinfer_plugin.so.10",
        "libnvonnxparser.so.10",
    ]
    for lib_dir in lib_dirs:
        for name in preload_names:
            path = lib_dir / name
            if path.exists():
                try:
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def load_edm_model(edm_repo, ckpt_path, device="cuda"):
    edm_repo = Path(edm_repo)
    if str(edm_repo) not in sys.path:
        sys.path.insert(0, str(edm_repo))

    from src.config.default import get_cfg_defaults
    from src.edm.edm import EDM
    from src.utils.misc import lower_config

    config = get_cfg_defaults()
    config.merge_from_file(str(edm_repo / "configs/edm/outdoor/edm_base.py"))
    config.merge_from_file(str(edm_repo / "configs/data/megadepth_test_1500.py"))
    edm_config = lower_config(config)["edm"]

    matcher = EDM(config=edm_config).to(device).eval()
    state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=False)["state_dict"]
    state_dict = {k.replace("matcher.", ""): v for k, v in state_dict.items()}
    matcher.load_state_dict(state_dict)
    return matcher


def image_to_edm_tensor(img_rgb, device="cuda"):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    x = torch.from_numpy((gray.astype(np.float32) / 255.0)[None, None])
    return x.to(device)


def compute_edm_matches(matcher, img0_rgb, img1_rgb, device="cuda"):
    batch = {
        "image0": image_to_edm_tensor(img0_rgb, device=device),
        "image1": image_to_edm_tensor(img1_rgb, device=device),
    }
    with torch.no_grad():
        matcher(batch)
    mkpts0 = batch["mkpts0_f"].cpu().numpy()
    mkpts1 = batch["mkpts1_f"].cpu().numpy()
    conf = batch["mconf"].cpu().numpy()
    return mkpts0, mkpts1, conf


def load_edm_trt_session(onnx_path, trt_cache_dir=None, require_gpu=True):
    _try_preload_ort_gpu_libs()
    import onnxruntime as ort

    providers = []
    trt_options = {}
    if trt_cache_dir:
        trt_cache_dir = Path(trt_cache_dir)
        trt_cache_dir.mkdir(parents=True, exist_ok=True)
        trt_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(trt_cache_dir),
        }
    providers.append(("TensorrtExecutionProvider", trt_options))
    providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f"  [EDM TRT] Loaded: {onnx_path}")
    print(f"  [EDM TRT] Providers: {session.get_providers()}")
    if require_gpu and not any(
        p in session.get_providers()
        for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider")
    ):
        raise RuntimeError(
            "EDM_TRT requested, but ONNXRuntime fell back to CPUExecutionProvider. "
            "Install/load the CUDA/cuDNN/TensorRT libraries required by onnxruntime-gpu. "
            "If the libraries are installed outside the active conda env, set "
            "TENSORRT_LIB_DIR and CUDNN_LIB_DIR or run: source setup_edm_env.sh"
        )
    return {
        "session": session,
        "input_name": session.get_inputs()[0].name,
        "output_name": session.get_outputs()[0].name,
    }


def warmup_edm_trt_session(matcher, height, width, n_runs=1):
    dummy = np.zeros((1, 2, int(height), int(width)), dtype=np.float32)
    session = matcher["session"]
    input_name = matcher["input_name"]
    output_name = matcher["output_name"]
    for _ in range(int(n_runs)):
        session.run([output_name], {input_name: dummy})


def _image_pair_to_onnx_input(img0_rgb, img1_rgb):
    img0_gray = cv2.cvtColor(img0_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    img1_gray = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return np.stack([img0_gray, img1_gray], axis=0)[None].astype(np.float32)


def _filter_edm_deploy_output(output, conf_thr, width, height):
    output = np.asarray(output)
    if output.size == 0:
        empty_pts = np.empty((0, 2), dtype=np.float32)
        empty_conf = np.empty((0,), dtype=np.float32)
        return empty_pts, empty_pts.copy(), empty_conf

    kpts0_c = output[:, :2]
    kpts1_c = output[:, 2:4]
    fine_offset01 = output[:, 4:6]
    fine_offset10 = output[:, 6:8]
    pred_score01 = output[:, 8]
    pred_score10 = output[:, 9]
    mconf = output[:, 10]

    mkpts0_f = kpts0_c
    mkpts1_f = kpts1_c + fine_offset01 * LOCAL_RESOLUTION
    mkpts0_f_rev = kpts0_c + fine_offset10 * LOCAL_RESOLUTION
    mkpts1_f_rev = kpts1_c

    mkpts0 = np.concatenate([mkpts0_f, mkpts0_f_rev], axis=0)
    mkpts1 = np.concatenate([mkpts1_f, mkpts1_f_rev], axis=0)
    pred_score = np.concatenate([pred_score01, pred_score10], axis=0)
    conf = np.concatenate([mconf, mconf], axis=0)

    border = LOCAL_RESOLUTION * 2
    sigma_thr = 1e-6
    keep = (
        (conf > conf_thr)
        & (mkpts0[:, 0] >= border)
        & (mkpts0[:, 0] <= width - border)
        & (mkpts0[:, 1] >= border)
        & (mkpts0[:, 1] <= height - border)
        & (mkpts1[:, 0] >= border)
        & (mkpts1[:, 0] <= width - border)
        & (mkpts1[:, 1] >= border)
        & (mkpts1[:, 1] <= height - border)
    )

    forward_better = pred_score01 > pred_score10
    score_keep = np.concatenate([forward_better, ~forward_better], axis=0)
    keep &= score_keep & (pred_score > sigma_thr)

    return (
        mkpts0[keep].astype(np.float32),
        mkpts1[keep].astype(np.float32),
        conf[keep].astype(np.float32),
    )


def compute_edm_trt_matches(matcher, img0_rgb, img1_rgb, conf_thr=0.5):
    session = matcher["session"]
    input_name = matcher["input_name"]
    output_name = matcher["output_name"]
    onnx_input = _image_pair_to_onnx_input(img0_rgb, img1_rgb)
    output = session.run([output_name], {input_name: onnx_input})[0]
    height, width = img0_rgb.shape[:2]
    return _filter_edm_deploy_output(output, conf_thr, width, height)
