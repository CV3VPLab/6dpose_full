import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ----------------------------------------------------------------------
# Repo root 등록
# ----------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gsplat import rasterization, rasterization_2dgs
from scene.gaussian_model import GaussianModel


def load_intrinsics(path):
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals.extend([float(x) for x in line.split()])

    if len(vals) == 9:
        K = np.array(vals, dtype=np.float64).reshape(3, 3)
    elif len(vals) == 4:
        fx, fy, cx, cy = vals
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported intrinsics format: {path}")
    return K


def load_pose_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    width = int(data["width"])
    height = int(data["height"])
    K = np.array(data["K"], dtype=np.float64)
    R = np.array(data["R_obj_to_cam"], dtype=np.float64)
    t = np.array(data["t_obj_to_cam"], dtype=np.float64).reshape(3)
    return width, height, K, R, t


def render_with_gsplat_2dgs(gaussians, R_obj_to_cam, t_obj_to_cam,
                              width, height, fx, fy, cx, cy,
                              bg_color_str="0,0,0", device="cuda",
                              render_depth=False, gs_mode="2dgs"):
    """
    Render a GaussianModel using gsplat.
    gs_mode="2dgs" uses rasterization_2dgs; gs_mode="3dgs" uses rasterization.

    Returns:
        render_chw : (3, H, W) float32 tensor [0, 1]
        depth_hw   : (H, W) float32 tensor or None
    """
    means = gaussians.get_xyz
    quats = gaussians.get_rotation          # (N, 4) wxyz
    opacities = gaussians.get_opacity.squeeze(-1)   # (N,)
    colors = gaussians.get_features                 # (N, D, 3) SH coefficients
    sh_degree = int(gaussians.active_sh_degree)

    if gs_mode == "2dgs":
        scales_2d = gaussians.get_scaling   # (N, 2) — 2DGS in-plane scales
        # Pad to (N, 3): the third dimension is the normal-direction scale
        scales = torch.cat([
            scales_2d,
            torch.full((scales_2d.shape[0], 1), 1e-10,
                       dtype=scales_2d.dtype, device=scales_2d.device)
        ], dim=-1)
    else:
        scales = gaussians.get_scaling      # (N, 3) — 3DGS full scales

    R = torch.as_tensor(R_obj_to_cam, dtype=torch.float32, device=device)
    t = torch.as_tensor(t_obj_to_cam, dtype=torch.float32, device=device).reshape(3)
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t
    viewmat = viewmat.unsqueeze(0)  # (1, 4, 4)

    K_mat = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)  # (1, 3, 3)

    bg = torch.tensor(
        [float(v) / 255.0 for v in str(bg_color_str).split(",")],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)  # (1, 3)

    render_mode = "RGB+D" if render_depth else "RGB"

    if gs_mode == "2dgs":
        renders, _alphas, _normals, _surf_normals, _distort, _median, _meta = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K_mat,
            width=int(width),
            height=int(height),
            sh_degree=sh_degree,
            near_plane=0.01,
            far_plane=100.0,
            backgrounds=bg,
            render_mode=render_mode,
            packed=False,
        )
    else:
        renders, _alphas, _meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K_mat,
            width=int(width),
            height=int(height),
            sh_degree=sh_degree,
            near_plane=0.01,
            far_plane=100.0,
            backgrounds=bg,
            render_mode=render_mode,
            packed=False,
        )
    # renders: (1, H, W, 3) or (1, H, W, 4)
    renders = renders[0]  # (H, W, ...)
    render_chw = renders[..., :3].permute(2, 0, 1).clamp(0.0, 1.0)  # (3, H, W)
    depth_hw = renders[..., 3] if render_depth else None
    return render_chw, depth_hw


def depth_to_xyz_map(depth_np, fx, fy, cx, cy, R_obj_to_cam, t_obj_to_cam):
    """render_gallery.py와 동일한 로직으로 depth → canonical XYZ map 변환"""
    H, W = depth_np.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_np.astype(np.float32)
    valid = np.isfinite(Z) & (Z > 1e-8)
    Xc = (uu - cx) * Z / fx
    Yc = (vv - cy) * Z / fy
    xyz_cam = np.stack([Xc, Yc, Z], axis=-1).astype(np.float32)
    R = np.asarray(R_obj_to_cam, dtype=np.float32)
    t = np.asarray(t_obj_to_cam, dtype=np.float32)
    pts = xyz_cam.reshape(-1, 3)
    valid_flat = valid.reshape(-1).astype(bool)
    pts_valid = pts[valid_flat]
    pts_obj_valid = (pts_valid - t[None, :]) @ R   # Xo = R^T (Xc - t)
    xyz_obj = np.zeros_like(xyz_cam.reshape(-1, 3))
    xyz_obj[valid_flat] = pts_obj_valid
    return xyz_obj.reshape(H, W, 3)


def parse_bg_color(bg_color_str):
    vals = [float(x) for x in bg_color_str.split(",")]
    if len(vals) != 3:
        raise ValueError("--bg_color must be like 0,0,0")
    return vals  # returned as list, used in render_with_gsplat_2dgs


def find_iteration_path(model_dir, iteration):
    pc_dir = Path(model_dir) / "point_cloud" / f"iteration_{iteration}"
    if not pc_dir.exists():
        raise FileNotFoundError(f"Iteration folder not found: {pc_dir}")
    return pc_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--intrinsics_path", type=str, required=True)
    parser.add_argument("--pose_json", type=str, required=True)
    parser.add_argument("--output_png", type=str, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--bg_color", type=str, default="0,0,0")
    parser.add_argument("--gs_mode", type=str, default="2dgs", choices=["2dgs", "3dgs"])
    parser.add_argument("--save_xyz", action="store_true", help="Save XYZ map as .npy")
    parser.add_argument("--xyz_output", type=str, default=None, help="Path to save XYZ map .npy")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This script currently expects CUDA.")

    K_txt = load_intrinsics(args.intrinsics_path)
    width, height, K_pose, R, t = load_pose_json(args.pose_json)

    # pose_json 기준으로 width/height/K를 우선 사용
    K = K_pose if K_pose is not None else K_txt
    width = int(width)
    height = int(height)

    _ = find_iteration_path(args.model_dir, args.iteration)

    gaussians = GaussianModel(3)
    gaussians.load_ply(
        str(Path(args.model_dir) / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply")
    )

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    need_depth = args.save_xyz
    with torch.no_grad():
        render_chw, depth_hw = render_with_gsplat_2dgs(
            gaussians=gaussians,
            R_obj_to_cam=R, t_obj_to_cam=t,
            width=width, height=height,
            fx=fx, fy=fy, cx=cx, cy=cy,
            bg_color_str=args.bg_color,
            device=device,
            render_depth=need_depth,
            gs_mode=args.gs_mode,
        )

    image = (render_chw.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    out_path = Path(args.output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image_bgr)
    print(f"[OK] saved render: {out_path}")

    if args.save_xyz:
        if depth_hw is None:
            print("[WARN] depth not available, skipping XYZ map")
        else:
            depth_np = depth_hw.float().detach().cpu().numpy()

            # gsplat depth is already in metric units — no scale correction.
            xyz_obj = depth_to_xyz_map(
                depth_np=depth_np,
                fx=fx, fy=fy, cx=cx, cy=cy,
                R_obj_to_cam=R,
                t_obj_to_cam=t,
            )

            xyz_path = Path(args.xyz_output) if args.xyz_output else out_path.with_suffix(".npy")
            xyz_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(xyz_path), xyz_obj.astype(np.float16))
            print(f"[OK] saved XYZ map: {xyz_path}")


if __name__ == "__main__":
    main()