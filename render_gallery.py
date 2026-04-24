import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from tqdm import tqdm
import cv2

from gaussian_renderer import GaussianModel
from gsplat import rasterization as _gsplat_rasterize


def parse_args():
    p = argparse.ArgumentParser(description="Render gallery images from custom poses for GS model")
    p.add_argument("--model_dir", required=True, type=str,
                   help="Canonicalized GS model directory")
    p.add_argument("--gallery_pose_json", required=True, type=str,
                   help="Path to gallery_poses.json")
    p.add_argument("--intrinsics_path", required=True, type=str,
                   help="Path to intrinsics txt file")
    p.add_argument("--output_dir", required=True, type=str,
                   help="Directory to save rendered gallery images")
    p.add_argument("--width", required=True, type=int)
    p.add_argument("--height", required=True, type=int)
    p.add_argument("--background", default="0,0,0", type=str,
                   help="R,G,B in 0-255")
    p.add_argument("--iteration", default=-1, type=int,
                   help="If -1, use latest iteration under point_cloud/")
    p.add_argument("--sh_degree", default=3, type=int)
    p.add_argument("--convert_SHs_python", action="store_true")
    p.add_argument("--compute_cov3D_python", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--antialiasing", action="store_true")
    p.add_argument("--gs_mode", default="3dgs", choices=["3dgs", "2dgs"])
    p.add_argument("--save_depth", action="store_true")
    p.add_argument("--save_xyz", action="store_true")
    p.add_argument("--depth_dir", type=str, default=None)
    p.add_argument("--depth_vis_dir", type=str, default=None)
    p.add_argument("--xyz_dir", type=str, default=None)
    return p.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def search_for_max_iteration(point_cloud_dir: Path):
    iters = []
    for p in point_cloud_dir.iterdir():
        if p.is_dir() and p.name.startswith("iteration_"):
            try:
                iters.append(int(p.name.split("_")[-1]))
            except Exception:
                pass
    if not iters:
        raise FileNotFoundError(f"No iteration_* directories found in {point_cloud_dir}")
    return max(iters)


def resolve_ply_path(model_dir: Path, iteration: int):
    point_cloud_root = model_dir / "point_cloud"
    if not point_cloud_root.exists():
        raise FileNotFoundError(f"point_cloud directory not found: {point_cloud_root}")

    if iteration == -1:
        iteration = search_for_max_iteration(point_cloud_root)

    ply_path = point_cloud_root / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"point_cloud.ply not found: {ply_path}")

    return ply_path, iteration


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
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
    elif len(vals) == 4:
        fx, fy, cx, cy = vals
    else:
        raise ValueError(f"Unsupported intrinsics format in: {path}")

    return fx, fy, cx, cy


def load_gallery_poses(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_projection_matrix_with_cx_cy(znear, zfar, fx, fy, cx, cy, width, height):
    """
    cx, cy를 반영한 OpenGL-style projection matrix.
    GS renderer는 column-major (row vector @ matrix) 방식이므로 transpose해서 사용.

    OpenCV projection:
      u = fx * Xc/Zc + cx
      v = fy * Yc/Zc + cy

    OpenGL NDC 변환:
      P[0,0] = 2*fx/W,  P[0,2] = 2*cx/W - 1  (x 주점 오프셋)
      P[1,1] = 2*fy/H,  P[1,2] = 2*cy/H - 1  (y 주점 오프셋)
    """
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 2.0 * fx / width
    P[1, 1] = 2.0 * fy / height
    P[0, 2] = 2.0 * cx / width - 1.0
    P[1, 2] = 2.0 * cy / height - 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P


def render_with_gsplat(gaussians, R_obj_to_cam, t_obj_to_cam,
                       width, height, fx, fy, cx, cy,
                       bg_color_str="0,0,0", device="cuda",
                       render_depth=False):
    """
    Render a GaussianModel (or RigidPoseGaussianProxy) with gsplat.

    When R_obj_to_cam is None, uses identity viewmat (the proxy has already
    applied the pose transform to the Gaussian positions).

    Returns:
        render_chw : (3, H, W) float32 tensor [0, 1]  — on device, grad-capable
        depth_hw   : (H, W) float32 tensor or None
    """
    means     = gaussians.get_xyz
    quats     = gaussians.get_rotation
    scales    = gaussians.get_scaling
    opacities = gaussians.get_opacity.squeeze(-1)
    colors    = gaussians.get_features
    sh_degree = int(gaussians.active_sh_degree)

    if R_obj_to_cam is None:
        viewmat = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        R = torch.as_tensor(R_obj_to_cam, dtype=torch.float32, device=device)
        t = torch.as_tensor(t_obj_to_cam, dtype=torch.float32, device=device).reshape(3)
        viewmat = torch.eye(4, dtype=torch.float32, device=device)
        viewmat[:3, :3] = R
        viewmat[:3, 3]  = t
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

    renders, _alphas, _ = _gsplat_rasterize(
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
    depth_hw   = renders[..., 3] if render_depth else None
    return render_chw, depth_hw


def save_depth_npy_and_vis(depth_tensor, npy_path, vis_path):
    if depth_tensor.ndim == 3:
        depth_np = depth_tensor.squeeze(0).detach().cpu().numpy()
    else:
        depth_np = depth_tensor.detach().cpu().numpy()

    np.save(str(npy_path), depth_np.astype(np.float32))

    valid = depth_np > 1e-8
    if np.any(valid):
        dmin = depth_np[valid].min()
        dmax = depth_np[valid].max()
        vis = np.zeros_like(depth_np, dtype=np.uint8)
        vis[valid] = ((depth_np[valid] - dmin) / (dmax - dmin + 1e-8) * 255).astype(np.uint8)
    else:
        vis = np.zeros_like(depth_np, dtype=np.uint8)

    cv2.imwrite(str(vis_path), vis)
    return depth_np


def depth_to_xyz_map(depth_np, fx, fy, cx, cy, R_obj_to_cam, t_obj_to_cam):
    H, W = depth_np.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))

    Z = depth_np.astype(np.float32)
    valid = Z > 1e-8

    Xc = (uu - cx) * Z / fx
    Yc = (vv - cy) * Z / fy

    xyz_cam = np.stack([Xc, Yc, Z], axis=-1).astype(np.float32)   # [H,W,3]

    R = np.asarray(R_obj_to_cam, dtype=np.float32)
    t = np.asarray(t_obj_to_cam, dtype=np.float32)

    xyz_obj = np.zeros_like(xyz_cam, dtype=np.float32)

    pts = xyz_cam.reshape(-1, 3)
    valid_flat = valid.reshape(-1)

    pts_valid = pts[valid_flat]
    # Xc = R Xo + t  ->  Xo = R^T (Xc - t)
    pts_obj_valid = (pts_valid - t[None, :]) @ R

    xyz_obj = xyz_obj.reshape(-1, 3)
    xyz_obj[valid_flat] = pts_obj_valid
    xyz_obj = xyz_obj.reshape(H, W, 3)

    return xyz_obj

def project_obj_to_image(X_obj, fx, fy, cx, cy, R_obj_to_cam, t_obj_to_cam):
    X_obj = np.asarray(X_obj, dtype=np.float32).reshape(3, 1)
    R = np.asarray(R_obj_to_cam, dtype=np.float32)
    t = np.asarray(t_obj_to_cam, dtype=np.float32).reshape(3, 1)

    X_cam = R @ X_obj + t
    xc, yc, zc = X_cam.reshape(3)

    if zc <= 1e-8:
        return None

    u = fx * xc / zc + cx
    v = fy * yc / zc + cy
    return np.array([u, v], dtype=np.float32)


def save_xyz_reprojection_check(render_bgr, xyz_obj, fx, fy, cx, cy, R, t, out_path, stride=200):
    vis = render_bgr.copy()
    H, W = xyz_obj.shape[:2]

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            X_obj = xyz_obj[y, x]
            if not np.isfinite(X_obj).all():
                continue
            if np.linalg.norm(X_obj) < 1e-8:
                continue

            uv = project_obj_to_image(X_obj, fx, fy, cx, cy, R, t)
            if uv is None:
                continue

            u2, v2 = int(round(uv[0])), int(round(uv[1]))

            cv2.circle(vis, (x, y), 2, (0, 255, 255), -1, cv2.LINE_AA)

            if 0 <= u2 < W and 0 <= v2 < H:
                cv2.circle(vis, (u2, v2), 2, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.line(vis, (x, y), (u2, v2), (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), vis)


def load_gaussians(model_dir, iteration=-1, sh_degree=3):
    """Load GaussianModel from PLY. Call once and keep in GPU memory."""
    ply_path, resolved_iter = resolve_ply_path(Path(model_dir), iteration)
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(ply_path), use_train_test_exp=False)
    print(f"[GS] Loaded gaussians: {ply_path}  (iter={resolved_iter})")
    return gaussians, ply_path, resolved_iter


def run_render_gallery(args, gaussians=None):
    """
    Run gallery rendering.
    Pass pre-loaded gaussians to skip model loading (for in-process preloading).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This renderer is expected to run on CUDA.")

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    depth_dir = Path(args.depth_dir) if args.depth_dir else (Path(args.output_dir).parent / "gallery_depth_gs")
    depth_vis_dir = Path(args.depth_vis_dir) if args.depth_vis_dir else (Path(args.output_dir).parent / "gallery_depth_vis_gs")
    xyz_dir = Path(args.xyz_dir) if args.xyz_dir else (Path(args.output_dir).parent / "gallery_xyz_gs")

    if args.save_depth:
        depth_dir.mkdir(parents=True, exist_ok=True)
        depth_vis_dir.mkdir(parents=True, exist_ok=True)

    if args.save_xyz:
        xyz_dir.mkdir(parents=True, exist_ok=True)

    gallery = load_gallery_poses(args.gallery_pose_json)
    fx, fy, cx, cy = load_intrinsics(args.intrinsics_path)

    ply_path, resolved_iter = resolve_ply_path(model_dir, args.iteration)

    print("=" * 60)
    print("[render_gallery.py] GS gallery render")
    print(f"  model_dir   : {model_dir}")
    print(f"  ply_path    : {ply_path}")
    print(f"  iteration   : {resolved_iter}")
    print(f"  width/height: {args.width} x {args.height}")
    print(f"  fx, fy      : {fx:.4f}, {fy:.4f}")
    print(f"  cx, cy      : {cx:.4f}, {cy:.4f}")
    print("=" * 60)

    if gaussians is None:
        gaussians = GaussianModel(args.sh_degree)
        gaussians.load_ply(str(ply_path), use_train_test_exp=False)
    else:
        print("[render_gallery.py] Using pre-loaded GaussianModel (skipping PLY load)")

    render_meta = {
        "model_dir": str(model_dir),
        "ply_path": str(ply_path),
        "iteration": resolved_iter,
        "gallery_pose_json": str(args.gallery_pose_json),
        "intrinsics_path": str(args.intrinsics_path),
        "width": args.width,
        "height": args.height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "principal_point_handling": "cx_cy_in_projection_matrix",
        "background": args.background,
        "gs_mode": args.gs_mode,
        "num_poses": len(gallery["poses"]),
        "renders": []
    }

    for pose in tqdm(gallery["poses"], desc="GS gallery rendering"):
        idx = int(pose["index"])
        out_name = f"{idx:04d}.png"
        out_path = output_dir / out_name

        R = np.array(pose["R_obj_to_cam"], dtype=np.float32)
        t = np.array(pose["t_obj_to_cam"], dtype=np.float32)

        with torch.no_grad():
            rgb_chw, depth_hw = render_with_gsplat(
                gaussians=gaussians,
                R_obj_to_cam=R, t_obj_to_cam=t,
                width=args.width, height=args.height,
                fx=fx, fy=fy, cx=cx, cy=cy,
                bg_color_str=args.background, device=device,
                render_depth=(args.save_depth or args.save_xyz),
            )

            rgb_np = (rgb_chw.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"{idx:04d}.png"), rgb_bgr)

            depth_np = None
            if args.save_depth and depth_hw is not None:
                depth_np = save_depth_npy_and_vis(
                    depth_tensor=depth_hw,
                    npy_path=depth_dir / f"{idx:04d}.npy",
                    vis_path=depth_vis_dir / f"{idx:04d}.png",
                )

            if args.save_xyz:
                if depth_np is None and depth_hw is not None:
                    depth_np = depth_hw.detach().cpu().numpy()

                # gsplat depth output is already in metric units (same unit as
                # canonical Gaussian means after apply_scale=1). No scale
                # correction needed: the rendered depth at the object center
                # pixel equals the front-surface depth, not the canonical
                # origin depth, so origin-based heuristics overcorrect by ~radius/distance.

                xyz_obj = depth_to_xyz_map(
                    depth_np=depth_np,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    R_obj_to_cam=pose["R_obj_to_cam"],
                    t_obj_to_cam=pose["t_obj_to_cam"],
                )
                np.save(str(xyz_dir / f"{idx:04d}.npy"), xyz_obj.astype(np.float16))

                save_xyz_reprojection_check(
                    render_bgr=rgb_bgr,
                    xyz_obj=xyz_obj,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    R=pose["R_obj_to_cam"],
                    t=pose["t_obj_to_cam"],
                    out_path=xyz_dir / f"{idx:04d}_reproj_check.png",
                    stride=250,
                )

        render_meta["renders"].append({
            "index": idx,
            "file": out_name,
            "azimuth_deg": pose.get("azimuth_deg", None),
            "elevation_deg": pose.get("elevation_deg", None),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "principal_point_handling": "cx_cy_in_projection_matrix",
        })

    save_json(output_dir / "render_meta.json", render_meta)

    print("=" * 60)
    print("[render_gallery.py] Done")
    print(f"  output_dir : {output_dir}")
    print("=" * 60)


def main():
    run_render_gallery(parse_args())


if __name__ == "__main__":
    main()