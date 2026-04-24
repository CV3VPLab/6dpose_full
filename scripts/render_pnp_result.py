"""
render_pnp_result.py
====================
PnP RANSAC으로 구한 R|t와 canonical GS PLY를 사용해 이미지를 렌더링하는 독립 스크립트.

Usage:
    python scripts/render_pnp_result.py \
        --pose_json  data/outputs_0422/q2_00044_3dgs_scale_ds_100iter/step6_pose_after_pnp.json \
        --model_dir  data/can_data/3dgs_pepsi_pinset/pepsi_painted_canonical \
        --gs_iter    30000 \
        --output     /tmp/pnp_render.png \
        [--gs_mode   3dgs] \
        [--bg_color  0,0,0] \
        [--save_xyz  /tmp/pnp_render_xyz.npy]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Repo root 등록 (gsplat, scene 등 import 가능하게)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gsplat import rasterization, rasterization_2dgs
from scene.gaussian_model import GaussianModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pose_json(path: str):
    with open(path, "r") as f:
        d = json.load(f)
    width  = int(d["width"])
    height = int(d["height"])
    K = np.array(d["K"], dtype=np.float64)
    R = np.array(d["R_obj_to_cam"], dtype=np.float64)
    t = np.array(d["t_obj_to_cam"], dtype=np.float64).reshape(3)
    return width, height, K, R, t


def load_gs_model(model_dir: str, gs_iter: int) -> GaussianModel:
    ply_path = Path(model_dir) / "point_cloud" / f"iteration_{gs_iter}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")
    gaussians = GaussianModel(3)
    gaussians.load_ply(str(ply_path))
    print(f"[OK] Loaded GaussianModel: {ply_path.name}  ({gaussians.get_xyz.shape[0]:,} gaussians)")
    return gaussians


def render(gaussians, R, t, width, height, fx, fy, cx, cy,
           bg_color_str="0,0,0", gs_mode="3dgs", render_depth=False,
           device="cuda"):
    means     = gaussians.get_xyz
    quats     = gaussians.get_rotation
    opacities = gaussians.get_opacity.squeeze(-1)
    colors    = gaussians.get_features
    sh_degree = int(gaussians.active_sh_degree)

    if gs_mode == "2dgs":
        scales_2d = gaussians.get_scaling
        scales = torch.cat([
            scales_2d,
            torch.full((scales_2d.shape[0], 1), 1e-10,
                       dtype=scales_2d.dtype, device=scales_2d.device)
        ], dim=-1)
    else:
        scales = gaussians.get_scaling

    R_t = torch.as_tensor(R, dtype=torch.float32, device=device)
    t_t = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(3)
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    viewmat[:3, :3] = R_t
    viewmat[:3, 3]  = t_t
    viewmat = viewmat.unsqueeze(0)                              # (1,4,4)

    K_t = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device
    ).unsqueeze(0)                                              # (1,3,3)

    bg = torch.tensor(
        [float(v) / 255.0 for v in str(bg_color_str).split(",")],
        dtype=torch.float32, device=device
    ).unsqueeze(0)                                              # (1,3)

    render_mode = "RGB+D" if render_depth else "RGB"

    with torch.no_grad():
        if gs_mode == "2dgs":
            renders, _alphas, _nn, _sn, _dist, _med, _meta = rasterization_2dgs(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmats=viewmat, Ks=K_t,
                width=int(width), height=int(height),
                sh_degree=sh_degree,
                near_plane=0.01, far_plane=100.0,
                backgrounds=bg, render_mode=render_mode, packed=False,
            )
        else:
            renders, _alphas, _meta = rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmats=viewmat, Ks=K_t,
                width=int(width), height=int(height),
                sh_degree=sh_degree,
                near_plane=0.01, far_plane=100.0,
                backgrounds=bg, render_mode=render_mode, packed=False,
            )

    renders = renders[0]                                        # (H,W,C)
    rgb_chw = renders[..., :3].permute(2, 0, 1).clamp(0, 1)    # (3,H,W)
    depth_hw = renders[..., 3] if render_depth else None
    return rgb_chw, depth_hw


def depth_to_xyz_obj(depth_np, fx, fy, cx, cy, R, t):
    """depth map → canonical object 좌표계 XYZ map"""
    H, W = depth_np.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_np.astype(np.float32)
    Xc = (uu - cx) * Z / fx
    Yc = (vv - cy) * Z / fy
    xyz_cam = np.stack([Xc, Yc, Z], axis=-1)           # (H,W,3) 카메라 좌표
    pts = xyz_cam.reshape(-1, 3)
    valid = np.isfinite(Z.ravel()) & (Z.ravel() > 1e-8)
    pts_obj = np.zeros_like(pts)
    pts_obj[valid] = (pts[valid] - t[None, :]) @ R      # Xo = R^T (Xc - t)
    return pts_obj.reshape(H, W, 3)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PnP R|t + canonical GS PLY → rendered image")
    p.add_argument("--pose_json",  required=True,  help="step6_pose_after_pnp.json 또는 initial_pose.json")
    p.add_argument("--model_dir",  required=True,  help="canonical GS model directory")
    p.add_argument("--gs_iter",    type=int, required=True, help="iteration number (e.g. 30000)")
    p.add_argument("--output",     required=True,  help="출력 PNG 경로")
    p.add_argument("--gs_mode",    default="3dgs", choices=["3dgs", "2dgs"])
    p.add_argument("--bg_color",   default="0,0,0")
    p.add_argument("--save_xyz",   default=None,   help="XYZ map 저장 경로 (.npy). 없으면 저장 안 함")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = "cuda"

    # ── 1. Pose load ─────────────────────────────────────────────────────────
    width, height, K, R, t = load_pose_json(args.pose_json)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    print(f"[Pose]")
    print(f"  size : {width}x{height}")
    print(f"  R    :\n{np.array2string(R, precision=4, suppress_small=True)}")
    print(f"  t    : [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")

    # ── 2. GS model load ─────────────────────────────────────────────────────
    gaussians = load_gs_model(args.model_dir, args.gs_iter)

    # ── 3. Render ─────────────────────────────────────────────────────────────
    need_depth = args.save_xyz is not None
    rgb_chw, depth_hw = render(
        gaussians, R, t, width, height, fx, fy, cx, cy,
        bg_color_str=args.bg_color,
        gs_mode=args.gs_mode,
        render_depth=need_depth,
        device=device,
    )

    # ── 4. Save RGB ───────────────────────────────────────────────────────────
    img_np = (rgb_chw.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_bgr)
    print(f"[OK] Saved render: {out_path}")

    # ── 5. Save XYZ map (optional) ────────────────────────────────────────────
    if args.save_xyz and depth_hw is not None:
        depth_np = depth_hw.float().cpu().numpy()

        # gsplat depth is already in metric units — no scale correction.
        xyz_obj = depth_to_xyz_obj(depth_np, fx, fy, cx, cy, R, t)

        xyz_path = Path(args.save_xyz)
        xyz_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(xyz_path), xyz_obj.astype(np.float16))
        print(f"[OK] Saved XYZ map: {xyz_path}")


if __name__ == "__main__":
    main()
