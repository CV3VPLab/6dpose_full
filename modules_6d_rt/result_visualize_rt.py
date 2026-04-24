"""
result_visualize_rt.py
======================
Real-time result visualization: draws a 3D bounding box on the query image
using the refined pose from step7 (refined_pose.json).
Saves a single output image: {gs_output_dir}/bbox_3d_result_rt.png
"""

import json
from pathlib import Path

import cv2
import numpy as np


def load_intrinsics(path):
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals.extend([float(x) for x in line.strip().split()])
    if len(vals) == 9:
        return np.array(vals, dtype=np.float64).reshape(3, 3)
    elif len(vals) == 4:
        fx, fy, cx, cy = vals
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"Unsupported intrinsics format: {path}")


def draw_3d_bounding_box(img, R, t, K, obj_width, obj_height, obj_depth):
    """
    Draw a 3D bounding box and XYZ axes on img.
    obj_width, obj_height, obj_depth in metres (same units as t).
    """
    x = obj_width / 2
    y = obj_depth / 2
    z = obj_height / 2

    corners_3d = np.array([
        [-x, -y, -z], [ x, -y, -z], [ x,  y, -z], [-x,  y, -z],
        [-x, -y,  z], [ x, -y,  z], [ x,  y,  z], [-x,  y,  z],
    ], dtype=np.float64)

    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    tvec = np.asarray(t, dtype=np.float64).reshape(3, 1)
    dist = np.zeros((4, 1))

    corners_2d, _ = cv2.projectPoints(corners_3d, rvec, tvec, K, dist)
    corners_2d = corners_2d.reshape(-1, 2).astype(int)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),   # top face
        (0, 4), (1, 5), (2, 6), (3, 7),   # vertical edges
    ]
    for start, end in edges:
        cv2.line(img, tuple(corners_2d[start]), tuple(corners_2d[end]), (0, 255, 0), 2)

    # XYZ axes
    axis_length = max(obj_width, obj_height, obj_depth) * 0.4
    axes_3d = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length],
    ], dtype=np.float64)
    axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist)
    axes_2d = axes_2d.reshape(-1, 2).astype(int)
    origin = tuple(axes_2d[0])
    cv2.line(img, origin, tuple(axes_2d[1]), (0, 0, 255), 4)   # X: red
    cv2.line(img, origin, tuple(axes_2d[2]), (0, 255, 0), 4)   # Y: green
    cv2.line(img, origin, tuple(axes_2d[3]), (255, 0, 0), 4)   # Z: blue
    cv2.circle(img, origin, 6, (255, 255, 255), -1)

    return img


def run_result_visualize_rt(args, model_cache=None):
    """
    Reads refined_pose.json from args.gs_output_dir and draws the 3D bbox
    on the original query image.
    Also saves a GS render overlay image if gaussians are available.
    """
    gs_output_dir = Path(args.gs_output_dir)
    refined_pose_path = gs_output_dir / "refined_pose.json"

    if not refined_pose_path.exists():
        raise FileNotFoundError(
            f"refined_pose.json not found: {refined_pose_path}\n"
            "Run step7 (GS pose refinement) first."
        )

    query_img = cv2.imread(str(args.query_img))
    if query_img is None:
        raise FileNotFoundError(f"Query image not found: {args.query_img}")

    with open(refined_pose_path, "r") as f:
        pose_data = json.load(f)

    R = np.array(pose_data["R_obj_to_cam_refined"], dtype=np.float64)
    t = np.array(pose_data["t_obj_to_cam_refined"], dtype=np.float64)
    K = load_intrinsics(args.intrinsics_path)

    result_img = draw_3d_bounding_box(
        query_img.copy(), R, t, K,
        obj_width=args.obj_width,
        obj_height=args.obj_height,
        obj_depth=args.obj_depth,
    )

    out_path = gs_output_dir / "bbox_3d_result_rt.png"
    cv2.imwrite(str(out_path), result_img)

    print("=" * 60)
    print("[Result Visualize RT] 3D bounding box drawn")
    print(f"  R_refined:\n{R}")
    print(f"  t_refined: {t.tolist()}")
    print(f"  output   : {out_path}")

    # ── GS render overlay ────────────────────────────────────────────────
    gaussians = model_cache.gaussians if model_cache is not None else None
    if gaussians is not None:
        try:
            import torch
            import sys
            gs_repo = getattr(args, "gs_repo", None)
            if gs_repo and str(gs_repo) not in sys.path:
                sys.path.insert(0, str(gs_repo))
            from gsplat import rasterization, rasterization_2dgs

            render_w = int(getattr(args, "render_width", query_img.shape[1]))
            render_h = int(getattr(args, "render_height", query_img.shape[0]))
            bg_color = getattr(args, "bg_color", "0,0,0")
            gs_mode  = getattr(args, "gs_mode", "2dgs")
            fx = float(K[0, 0]); fy = float(K[1, 1])
            cx = float(K[0, 2]); cy = float(K[1, 2])
            device = "cuda"

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
            viewmat = viewmat.unsqueeze(0)

            K_mat = torch.tensor(
                [[fx, 0., cx], [0., fy, cy], [0., 0., 1.]],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)

            bg = torch.tensor(
                [float(v) / 255.0 for v in str(bg_color).split(",")],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                if gs_mode == "2dgs":
                    renders, _, _, _, _, _, _ = rasterization_2dgs(
                        means=means, quats=quats, scales=scales,
                        opacities=opacities, colors=colors,
                        viewmats=viewmat, Ks=K_mat,
                        width=render_w, height=render_h,
                        sh_degree=sh_degree,
                        near_plane=0.01, far_plane=100.0,
                        backgrounds=bg, render_mode="RGB", packed=False,
                    )
                else:
                    renders, _, _ = rasterization(
                        means=means, quats=quats, scales=scales,
                        opacities=opacities, colors=colors,
                        viewmats=viewmat, Ks=K_mat,
                        width=render_w, height=render_h,
                        sh_degree=sh_degree,
                        near_plane=0.01, far_plane=100.0,
                        backgrounds=bg, render_mode="RGB", packed=False,
                    )
            rgb_chw = renders[0].permute(2, 0, 1).clamp(0, 1)

            render_np = (rgb_chw.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            render_bgr = cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR)

            # resize render to match query if needed
            qh, qw = query_img.shape[:2]
            if render_bgr.shape[:2] != (qh, qw):
                render_bgr = cv2.resize(render_bgr, (qw, qh), interpolation=cv2.INTER_LINEAR)

            # alpha blend: render over query
            alpha = 0.5
            overlay = cv2.addWeighted(query_img, 1.0 - alpha, render_bgr, alpha, 0)

            overlay_path = gs_output_dir / "gs_render_overlay_rt.png"
            cv2.imwrite(str(overlay_path), overlay)
            print(f"  render overlay: {overlay_path}")

            # also save the render itself
            render_path = gs_output_dir / "gs_render_full_rt.png"
            cv2.imwrite(str(render_path), render_bgr)
            print(f"  render full   : {render_path}")

        except Exception as e:
            print(f"  [WARN] GS render overlay failed: {e}")

    print("=" * 60)
