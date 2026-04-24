"""
step45_translation_rt.py
========================
Real-time version of step6: LoFTR correspondences → PnP → translation refinement.
All debug visualization calls removed.
Outputs only:
  - initial_pose.json  (read by step7)
"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate unmapping
# ─────────────────────────────────────────────────────────────────────────────

def unmap_from_square_resize(pts_resized, orig_hw, resize_target=840):
    h, w = orig_hw
    side = max(h, w)
    x0 = (side - w) // 2
    y0 = (side - h) // 2
    pts_square = np.asarray(pts_resized, dtype=np.float64) * (side / resize_target)
    return pts_square - np.array([[x0, y0]], dtype=np.float64)


def to_full_image_coords(pts_crop, nonblack_bbox_xyxy):
    x1, y1 = float(nonblack_bbox_xyxy[0]), float(nonblack_bbox_xyxy[1])
    return np.asarray(pts_crop, dtype=np.float64) + np.array([[x1, y1]])


# ─────────────────────────────────────────────────────────────────────────────
# XYZ map helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_xyz_map_path(xyz_dir: Path, render_filename: str):
    stem = Path(render_filename).stem
    candidates = [
        xyz_dir / f"{stem}.npy",
        xyz_dir / f"{stem}_xyz.npy",
        xyz_dir / f"{stem}_xyz_map.npy",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"XYZ map not found for '{render_filename}' in {xyz_dir}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def lookup_xyz_at_pixels(xyz_map, pts_uv, bilinear=False):
    H, W = xyz_map.shape[:2]
    N = len(pts_uv)
    pts3d = np.full((N, 3), np.nan, dtype=np.float64)

    if not bilinear:
        u = np.round(pts_uv[:, 0]).astype(int)
        v = np.round(pts_uv[:, 1]).astype(int)
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        pts3d[in_bounds] = xyz_map[v[in_bounds], u[in_bounds]].astype(np.float64)
    else:
        # Vectorized bilinear interpolation (replaces Python loop)
        uf = pts_uv[:, 0]
        vf = pts_uv[:, 1]
        u0 = np.floor(uf).astype(int)
        v0 = np.floor(vf).astype(int)
        u1 = u0 + 1
        v1 = v0 + 1
        in_bounds = (u0 >= 0) & (v0 >= 0) & (u1 < W) & (v1 < H)
        idx = np.where(in_bounds)[0]
        if idx.size > 0:
            du = (uf[idx] - u0[idx])[:, None]
            dv = (vf[idx] - v0[idx])[:, None]
            val = (xyz_map[v0[idx], u0[idx]] * (1 - du) * (1 - dv) +
                   xyz_map[v0[idx], u1[idx]] * du       * (1 - dv) +
                   xyz_map[v1[idx], u0[idx]] * (1 - du) * dv +
                   xyz_map[v1[idx], u1[idx]] * du       * dv)
            pts3d[idx] = val.astype(np.float64)

    finite = np.all(np.isfinite(pts3d), axis=1)
    nonzero = np.abs(pts3d).sum(axis=1) > 1e-6
    valid = finite & nonzero
    return pts3d, valid


# ─────────────────────────────────────────────────────────────────────────────
# PLY helper
# ─────────────────────────────────────────────────────────────────────────────

def load_ply_xyz(ply_path):
    from plyfile import PlyData
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Pose estimation helpers
# ─────────────────────────────────────────────────────────────────────────────

def rotation_matrix_to_quaternion(R):
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q


def estimate_t_linear(pts2d, pts3d, K, R):
    K_inv = np.linalg.inv(K)
    N = len(pts2d)
    X3d_cam = (R @ pts3d.T).T
    uvh = (K_inv @ np.column_stack([pts2d, np.ones(N)]).T).T

    A = np.zeros((2 * N, 3), dtype=np.float64)
    b = np.zeros(2 * N, dtype=np.float64)
    for i in range(N):
        un, vn = uvh[i, 0], uvh[i, 1]
        rx, ry, rz = X3d_cam[i]
        A[2*i,   0] = 1;  A[2*i,   2] = -un
        b[2*i]   = -(rx - un * rz)
        A[2*i+1, 1] = 1;  A[2*i+1, 2] = -vn
        b[2*i+1] = -(ry - vn * rz)

    t_ls, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return t_ls.astype(np.float64)


def solve_pose_pnp(pts2d, pts3d, K, R_init,
                   reproj_thresh=5.0, min_inliers=6,
                   use_ransac=True):
    N = len(pts2d)
    if N < 4:
        print(f"  [PnP] {N} points < 4, cannot estimate pose")
        t_linear = estimate_t_linear(pts2d, pts3d, K, R_init) if N > 0 else np.zeros(3)
        return R_init, t_linear, "linear_T_fixed_R_insufficient_points", 0, np.inf, np.array([], dtype=np.int32)

    dist = np.zeros((4, 1), dtype=np.float64)

    if not use_ransac:
        rvec_init, _ = cv2.Rodrigues(R_init.astype(np.float64))
        t_init_linear = estimate_t_linear(pts2d, pts3d, K, R_init)
        retval, rvec, tvec = cv2.solvePnP(
            pts3d.astype(np.float32), pts2d.astype(np.float32),
            K.astype(np.float64), dist,
            rvec=rvec_init, tvec=t_init_linear.reshape(3, 1).astype(np.float64),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not retval:
            return R_init, t_init_linear, "linear_T_fixed_R_no_ransac_failed", 0, np.inf, np.array([], dtype=np.int32)
        R_out, _ = cv2.Rodrigues(rvec)
        t_out = tvec.ravel()
        inlier_idx = np.arange(N, dtype=np.int32)
        proj, _ = cv2.projectPoints(pts3d.astype(np.float32), rvec, tvec, K.astype(np.float64), dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - pts2d, axis=1).mean()
        print(f"  [PnP] No-RANSAC: all {N} pts used, reproj_err={err:.2f}px")
        return R_out.astype(np.float64), t_out.astype(np.float64), "pnp_no_ransac_iterative", N, float(err), inlier_idx

    # RANSAC: Stage 1 EPnP
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d.astype(np.float32), pts2d.astype(np.float32),
        K.astype(np.float64), dist,
        useExtrinsicGuess=False, iterationsCount=1000,
        reprojectionError=reproj_thresh, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if not success or inliers is None or len(inliers) < min_inliers:
        print(f"  [PnP] Stage 1 (EPnP) failed. Using linear T.")
        t_linear = estimate_t_linear(pts2d, pts3d, K, R_init)
        return R_init, t_linear, "linear_T_fixed_R", 0, np.inf, np.array([], dtype=np.int32)

    inlier_idx = inliers.ravel().astype(np.int32)
    print(f"  [PnP] Stage 1 (EPnP): {len(inlier_idx)} inliers / {N} pts")

    # Stage 2: ITERATIVE refine on inliers
    if len(inlier_idx) >= 6:
        pts3d_in = pts3d[inlier_idx].astype(np.float32)
        pts2d_in = pts2d[inlier_idx].astype(np.float32)
        retval, rvec_ref, tvec_ref = cv2.solvePnP(
            pts3d_in, pts2d_in, K.astype(np.float64), dist,
            rvec=rvec.copy(), tvec=tvec.copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if retval:
            rvec, tvec = rvec_ref, tvec_ref
            print(f"  [PnP] Stage 2 (ITERATIVE refine on {len(inlier_idx)} inliers): OK")

    R_out, _ = cv2.Rodrigues(rvec)
    t_out = tvec.ravel()

    pts3d_in = pts3d[inlier_idx]
    pts2d_in = pts2d[inlier_idx]
    proj, _ = cv2.projectPoints(pts3d_in.astype(np.float32), rvec, tvec, K.astype(np.float64), dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - pts2d_in, axis=1).mean()
    print(f"  [PnP] Final: {len(inlier_idx)} inliers, reproj_err={err:.2f}px")

    return (R_out.astype(np.float64), t_out.astype(np.float64),
            "pnp_2stage_epnp", len(inlier_idx), float(err), inlier_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Point filtering / sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def filter_points_by_binary_mask(pts2d, mask_img):
    pts2d = np.asarray(pts2d, dtype=np.float64)
    h, w = mask_img.shape[:2]
    u = np.round(pts2d[:, 0]).astype(int)
    v = np.round(pts2d[:, 1]).astype(int)
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    inside = np.zeros(len(pts2d), dtype=bool)
    inside[in_bounds] = mask_img[v[in_bounds], u[in_bounds]] > 0
    return inside


def compute_points_bbox(pts2d, img_w=None, img_h=None, pad=0):
    pts2d = np.asarray(pts2d, dtype=np.float64)
    x1, y1 = float(np.min(pts2d[:, 0])) - pad, float(np.min(pts2d[:, 1])) - pad
    x2, y2 = float(np.max(pts2d[:, 0])) + pad, float(np.max(pts2d[:, 1])) + pad
    if img_w is not None:
        x1, x2 = max(0.0, x1), min(float(img_w - 1), x2)
    if img_h is not None:
        y1, y2 = max(0.0, y1), min(float(img_h - 1), y2)
    return x1, y1, x2, y2


def uniform_sample_points_2d(pts2d, scores=None, grid_rows=2, grid_cols=2, max_per_cell=60):
    pts2d = np.asarray(pts2d, dtype=np.float64)
    N = len(pts2d)
    if N == 0:
        return np.array([], dtype=np.int32), {}
    if scores is None:
        scores = np.ones(N, dtype=np.float64)
    else:
        scores = np.asarray(scores, dtype=np.float64)

    x1, y1, x2, y2 = compute_points_bbox(pts2d)
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)

    col_ids = np.clip(np.floor((pts2d[:, 0] - x1) / width * grid_cols).astype(int), 0, grid_cols - 1)
    row_ids = np.clip(np.floor((pts2d[:, 1] - y1) / height * grid_rows).astype(int), 0, grid_rows - 1)

    cell_to_indices = {}
    for i in range(N):
        key = (int(row_ids[i]), int(col_ids[i]))
        cell_to_indices.setdefault(key, []).append(i)

    keep = []
    cell_stats = {}
    for key, idxs in cell_to_indices.items():
        idxs = np.array(idxs, dtype=np.int32)
        order = np.argsort(-scores[idxs])
        chosen = idxs[order[:max_per_cell]]
        keep.append(chosen)
        cell_stats[str(key)] = int(len(chosen))

    if not keep:
        return np.array([], dtype=np.int32), cell_stats
    return np.unique(np.concatenate(keep, axis=0)).astype(np.int32), cell_stats


# ─────────────────────────────────────────────────────────────────────────────
# Translation refinement helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_to_binary_mask(render_img_bgr, nonblack_thresh=8):
    gray = cv2.cvtColor(render_img_bgr, cv2.COLOR_BGR2GRAY)
    return (gray > nonblack_thresh).astype(np.uint8) * 255


def binary_mask_bbox_stats(mask_img):
    mask = (mask_img > 0)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "w": float(x2 - x1 + 1), "h": float(y2 - y1 + 1),
        "area": float(mask.sum()),
        "cx": 0.5 * (x1 + x2), "cy": 0.5 * (y1 + y2),
    }


def binary_mask_iou(mask_a, mask_b):
    a, b = (mask_a > 0), (mask_b > 0)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def estimate_tz_from_mask_bbox(query_mask_img, fy, object_height_m):
    stats = binary_mask_bbox_stats(query_mask_img)
    if stats is None:
        return None
    return float(fy) * float(object_height_m) / max(1.0, stats["h"])


def clamp_tz_by_prior(t, tz_bbox, prior_range=(0.80, 1.20)):
    t = np.asarray(t, dtype=np.float64).copy()
    if tz_bbox is None:
        return t
    tz_min = max(0.05, float(prior_range[0]) * float(tz_bbox))
    tz_max = max(tz_min + 1e-6, float(prior_range[1]) * float(tz_bbox))
    t[2] = np.clip(float(t[2]), tz_min, tz_max)
    return t


def make_query_reference_mask(query_masked_img=None, query_mask_path=None, nonblack_thresh=8):
    if query_mask_path is not None:
        query_mask_path = Path(query_mask_path)
        if query_mask_path.exists():
            qmask = cv2.imread(str(query_mask_path), cv2.IMREAD_GRAYSCALE)
            if qmask is not None:
                return (qmask > 0).astype(np.uint8) * 255
    if query_masked_img is not None:
        return render_to_binary_mask(query_masked_img, nonblack_thresh=nonblack_thresh)
    return None


def score_render_mask_against_query(query_mask, render_mask):
    q_stats = binary_mask_bbox_stats(query_mask)
    r_stats = binary_mask_bbox_stats(render_mask)
    if q_stats is None or r_stats is None:
        return {"score": -1e9, "iou": 0.0, "center_px": None, "height_ratio": None, "area_ratio": None}

    iou = binary_mask_iou(query_mask, render_mask)
    dx = r_stats["cx"] - q_stats["cx"]
    dy = r_stats["cy"] - q_stats["cy"]
    center_px = float(np.hypot(dx, dy))
    H, W = query_mask.shape[:2]
    diag = max(1.0, float(np.hypot(W, H)))
    center_term = float(np.exp(-3.0 * center_px / diag))
    h_ratio = r_stats["h"] / max(q_stats["h"], 1e-8)
    area_ratio = r_stats["area"] / max(q_stats["area"], 1e-8)
    height_term = float(np.exp(-abs(np.log(max(h_ratio, 1e-8)))))
    area_term = float(np.exp(-abs(np.log(max(area_ratio, 1e-8)))))
    score = 0.55 * iou + 0.25 * center_term + 0.15 * height_term + 0.05 * area_term
    return {"score": float(score), "iou": float(iou), "center_px": float(center_px),
            "height_ratio": float(h_ratio), "area_ratio": float(area_ratio)}


def save_camera_pose_json(path, K, R, t, width, height):
    data = {
        "width": int(width), "height": int(height),
        "K": np.asarray(K, dtype=np.float64).tolist(),
        "R_obj_to_cam": np.asarray(R, dtype=np.float64).tolist(),
        "t_obj_to_cam": np.asarray(t, dtype=np.float64).reshape(3).tolist(),
    }
    save_json(path, data)


def _parse_bg_color_tensor(bg_color_str, device="cuda"):
    import torch
    vals = [float(x) for x in str(bg_color_str).split(",")]
    if len(vals) != 3:
        raise ValueError("bg_color must be 'R,G,B'")
    return torch.tensor(vals, dtype=torch.float32, device=device)


def render_single_pose_direct(gaussians, K, R, t, width, height,
                               bg_color="0,0,0", device="cuda"):
    """
    In-process GS render for a single pose using gsplat.
    Returns rendered image as BGR uint8 numpy array (H, W, 3).
    """
    import torch
    from render_gallery import render_with_gsplat

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    with torch.no_grad():
        rgb_chw, _ = render_with_gsplat(
            gaussians=gaussians,
            R_obj_to_cam=R, t_obj_to_cam=t,
            width=int(width), height=int(height),
            fx=fx, fy=fy, cx=cx, cy=cy,
            bg_color_str=bg_color, device=device,
            render_depth=False,
        )

    image_np = (rgb_chw.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


def render_single_pose_gs(gs_python, gs_repo, gs_model_dir, gs_iter,
                          intrinsics_path, pose_json_path, output_png_path,
                          width, height, bg_color="0,0,0", gs_mode="2dgs",
                          gaussians=None, K=None, R=None, t=None):
    """Render one GS pose.  Uses in-process rendering when `gaussians` is provided."""
    if gaussians is not None and K is not None and R is not None and t is not None:
        img_bgr = render_single_pose_direct(
            gaussians=gaussians, K=K, R=R, t=t,
            width=width, height=height, bg_color=bg_color,
        )
        out = Path(output_png_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img_bgr)
        return

    gs_repo = Path(gs_repo)
    script_path = gs_repo / "scripts" / "render_single_pose.py"
    if not script_path.exists():
        raise FileNotFoundError(f"render_single_pose.py not found: {script_path}")
    cmd = [
        str(gs_python), str(script_path),
        "--model_dir", str(gs_model_dir),
        "--iteration", str(gs_iter),
        "--intrinsics_path", str(intrinsics_path),
        "--pose_json", str(pose_json_path),
        "--output_png", str(output_png_path),
        "--width", str(width),
        "--height", str(height),
        "--bg_color", str(bg_color),
        "--gs_mode", str(gs_mode),
    ]
    subprocess.run(cmd, check=True)


def refine_translation_xyz_with_mask(
    *, R, t_init, K, width, height, query_mask_img, object_height_m,
    gs_python, gs_repo, gs_model_dir, gs_iter, intrinsics_path,
    bg_color="0,0,0", gs_mode="2dgs", nonblack_thresh=8, prior_range=(0.80, 1.20),
    num_iters=4, alpha_xy=0.65, alpha_z=0.55,
    gaussians=None,
):
    t_cur = np.asarray(t_init, dtype=np.float64).reshape(3).copy()
    info = {"status": "not_run", "tz_pnp": float(t_init[2]), "num_iters": int(num_iters),
            "history": [], "t_refined": t_cur.tolist(), "best_score": None}

    if query_mask_img is None:
        info["status"] = "skipped_no_query_mask"
        return t_cur, info

    q_stats = binary_mask_bbox_stats(query_mask_img)
    if q_stats is None:
        info["status"] = "skipped_invalid_query_mask"
        return t_cur, info

    fy = float(K[1, 1])
    fx = float(K[0, 0])
    tz_bbox = estimate_tz_from_mask_bbox(query_mask_img, fy=fy, object_height_m=object_height_m)
    info["tz_bbox_prior"] = float(tz_bbox) if tz_bbox is not None else None
    t_cur = clamp_tz_by_prior(t_cur, tz_bbox, prior_range=prior_range)

    best_t = t_cur.copy()
    best_score = -1e9
    best_metrics = None

    with tempfile.TemporaryDirectory(prefix="step6_rt_xyz_refine_") as td:
        td = Path(td)
        for it in range(int(num_iters)):
            pose_json_path = td / f"iter_{it:02d}.json"
            render_png_path = td / f"iter_{it:02d}.png"

            save_camera_pose_json(pose_json_path, K=K, R=R, t=t_cur, width=width, height=height)

            try:
                render_single_pose_gs(
                    gs_python=gs_python, gs_repo=gs_repo, gs_model_dir=gs_model_dir,
                    gs_iter=gs_iter, intrinsics_path=intrinsics_path,
                    pose_json_path=pose_json_path, output_png_path=render_png_path,
                    width=width, height=height, bg_color=bg_color, gs_mode=gs_mode,
                    gaussians=gaussians, K=K, R=R, t=t_cur,
                )
                render_img = cv2.imread(str(render_png_path), cv2.IMREAD_COLOR)
                if render_img is None:
                    raise RuntimeError("render output not found")
                render_mask = render_to_binary_mask(render_img, nonblack_thresh=nonblack_thresh)
            except Exception as e:
                print(f"  [xyz refine] iter={it} render failed: {e}")
                info["status"] = "render_failed"
                return best_t, info

            r_stats = binary_mask_bbox_stats(render_mask)
            metrics = score_render_mask_against_query(query_mask_img, render_mask)

            if metrics["score"] > best_score:
                best_score = metrics["score"]
                best_t = t_cur.copy()
                best_metrics = metrics

            if r_stats is None:
                break

            du = q_stats["cx"] - r_stats["cx"]
            dv = q_stats["cy"] - r_stats["cy"]
            dtx = alpha_xy * (du / fx) * float(t_cur[2])
            dty = alpha_xy * (dv / fy) * float(t_cur[2])

            t_new = t_cur.copy()
            t_new[0] += dtx
            t_new[1] += dty

            h_ratio = r_stats["h"] / max(q_stats["h"], 1e-8)
            z_scale = float(np.clip((1.0 - alpha_z) + alpha_z * h_ratio, 0.85, 1.15))
            t_new[2] *= z_scale
            t_new = clamp_tz_by_prior(t_new, tz_bbox, prior_range=prior_range)

            info["history"].append({"iter": int(it), "metrics": metrics,
                                    "du_px": float(du), "dv_px": float(dv)})
            t_cur = t_new

    info["status"] = "ok"
    info["t_refined"] = best_t.tolist()
    info["best_score"] = float(best_score) if best_metrics is not None else None
    info["best_metrics"] = best_metrics
    return best_t, info


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def run_step6_translation_rt(args, model_cache=None):
    import time as _time
    _TIMING = True   # ← set False to disable sub-step timing prints
    _ts = []          # list of (label, timestamp) — preserves insertion order
    def _t(label): _ts.append((label, _time.perf_counter()))

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    _t("start")
    # ── 1. Load step5 loftr data ──────────────────────────────────────────────
    npz_path = out_dir / "loftr_best_match_data.npz"
    meta_path = out_dir / "loftr_best_match_meta.json"
    loftr_json_path = out_dir / "loftr_scores.json"

    if not npz_path.exists():
        raise FileNotFoundError(
            f"loftr_best_match_data.npz not found: {npz_path}\n"
            "Run step5 (dino_loftr) first."
        )

    match_data = np.load(str(npz_path))
    mkpts0_840 = match_data["mkpts0_inlier_840"].astype(np.float64)
    mkpts1_840 = match_data["mkpts1_inlier_840"].astype(np.float64)
    conf_inlier = match_data["conf_inlier"].astype(np.float64)
    _t("1_load_loftr_data")

    meta = load_json(meta_path)
    loftr_json = load_json(loftr_json_path)
    best_render = loftr_json["best_render"]
    print(f"  Loaded {len(mkpts0_840)} inlier matches for '{best_render}'")

    # ── 2. Coordinate unmapping ───────────────────────────────────────────────
    resize_target = int(meta["loftr_resize_target"])

    q_crop_hw = tuple(meta["query_crop_hw"])
    q_bbox = meta["query_nonblack_bbox_xyxy"]
    pts_q_crop = unmap_from_square_resize(mkpts0_840, q_crop_hw, resize_target)
    pts_q_full = to_full_image_coords(pts_q_crop, q_bbox)

    # Backwards compat: if query was cropped (old meta), apply step1 offset
    if q_bbox[0] != 0 or q_bbox[1] != 0:
        step1_json_path = out_dir / "step1_result.json"
        if step1_json_path.exists():
            step1_data = load_json(step1_json_path)
            s1_bbox = step1_data.get("mask_bbox_xyxy") or step1_data.get("bbox_xyxy")
            if s1_bbox is not None:
                pts_q_full = pts_q_full + np.array([[float(s1_bbox[0]), float(s1_bbox[1])]])

    g_crop_hw = tuple(meta["gallery_crop_hw"])
    g_bbox = meta["gallery_nonblack_bbox_xyxy"]
    pts_g_crop = unmap_from_square_resize(mkpts1_840, g_crop_hw, resize_target)
    pts_g_full = to_full_image_coords(pts_g_crop, g_bbox)
    _t("2_coord_unmap")

    # ── 3. XYZ map load + optional scale correction ───────────────────────────
    xyz_dir = Path(args.gallery_xyz_dir)
    xyz_map_path = find_xyz_map_path(xyz_dir, best_render)
    xyz_map = np.load(str(xyz_map_path)).astype(np.float16)
    print(f"  XYZ map: {xyz_map_path.name}  shape={xyz_map.shape}")

    # Scale correction: use pre-computed factor if available (constant per gallery),
    # otherwise fall back to per-frame percentile/median computation.
    _cached_scale = getattr(model_cache, "xyz_scale_factor", None) if model_cache is not None else None
    if _cached_scale is not None:
        xyz_map = xyz_map * _cached_scale
        print(f"  [XYZ scale fix] correction={_cached_scale:.6f} (cached)")
    else:
        canonical_ply_path = getattr(args, "canonical_ply_path", None)
        _cached_med_norm = getattr(model_cache, "ply_med_norm", None) if model_cache is not None else None
        if canonical_ply_path is not None and Path(canonical_ply_path).exists():
            if _cached_med_norm is not None:
                _ply_med_norm = _cached_med_norm
            else:
                _ply_xyz = load_ply_xyz(Path(canonical_ply_path))
                _ply_med_norm = float(np.median(np.linalg.norm(_ply_xyz, axis=1)))
            _xyz_valid_mask = np.abs(xyz_map).sum(axis=-1) > 1e-6
            if _xyz_valid_mask.any():
                _xyz_valid = xyz_map[_xyz_valid_mask]
                _N_SAMPLE = 50_000
                if len(_xyz_valid) > _N_SAMPLE:
                    _rng = np.random.default_rng(0)
                    _s_idx = _rng.choice(len(_xyz_valid), _N_SAMPLE, replace=False)
                    _xyz_sample = _xyz_valid[_s_idx]
                else:
                    _xyz_sample = _xyz_valid
                _xyz_norms = np.linalg.norm(_xyz_sample, axis=1)
                _p5, _p95 = np.percentile(_xyz_norms, [5, 95])
                _core_mask = (_xyz_norms >= _p5) & (_xyz_norms <= _p95)
                if _core_mask.sum() > 100:
                    _xyz_med_norm = float(np.median(_xyz_norms[_core_mask]))
                    _scale = _ply_med_norm / _xyz_med_norm
                    if 0.8 < _scale < 1.25:
                        xyz_map = xyz_map * _scale
                        print(f"  [XYZ scale fix] correction={_scale:.6f}")
    _t("3_xyz_load_scale")

    # ── 4. XYZ lookup ─────────────────────────────────────────────────────────
    pts3d, valid_mask = lookup_xyz_at_pixels(xyz_map, pts_g_full, bilinear=True)
    n_valid = int(valid_mask.sum())
    print(f"  Valid 2D-3D correspondences: {n_valid} / {len(pts_g_full)}")
    _t("4_xyz_lookup")

    if n_valid < 4:
        raise RuntimeError(f"Not enough valid 2D-3D correspondences ({n_valid}).")

    pts2d_corr = pts_q_full[valid_mask]
    pts3d_corr = pts3d[valid_mask]
    conf_corr = conf_inlier[valid_mask]

    # # ── 5. Query-mask filtering ───────────────────────────────────────────────
    # query_mask_path = getattr(args, "query_mask_path", None)
    # n_inside_query_mask = None

    # if query_mask_path:
    #     query_mask_img = cv2.imread(str(query_mask_path), cv2.IMREAD_GRAYSCALE)
    #     if query_mask_img is None:
    #         raise FileNotFoundError(f"Failed to load query mask: {query_mask_path}")
    #     inside_mask = filter_points_by_binary_mask(pts2d_corr, query_mask_img)
    #     n_inside_query_mask = int(inside_mask.sum())
    #     print(f"  Query-mask inside: {n_inside_query_mask} / {len(pts2d_corr)}")
    #     pts2d_corr = pts2d_corr[inside_mask]
    #     pts3d_corr = pts3d_corr[inside_mask]
    #     conf_corr = conf_corr[inside_mask]
    #     if len(pts2d_corr) < 4:
    #         raise RuntimeError(f"Not enough correspondences after query-mask filtering: {len(pts2d_corr)}")
    # else:
    #     print("  [WARN] --query_mask_path not provided, skipping query-mask filtering")
    # _t("5_query_mask_filter")

    # # ── 6. Uniform spatial sampling ───────────────────────────────────────────
    # uniform_keep_idx, _ = uniform_sample_points_2d(
    #     pts2d_corr, scores=conf_corr, grid_rows=10, grid_cols=10, max_per_cell=60,
    # )
    # print(f"  Uniform sampling: {len(uniform_keep_idx)} / {len(pts2d_corr)} kept")
    # pts2d_corr = pts2d_corr[uniform_keep_idx]
    # pts3d_corr = pts3d_corr[uniform_keep_idx]
    # conf_corr = conf_corr[uniform_keep_idx]
    # if len(pts2d_corr) < 4:
    #     raise RuntimeError(f"Not enough correspondences after uniform sampling: {len(pts2d_corr)}")

    # # ── 7. 3D outlier filter ──────────────────────────────────────────────────
    # _norms = np.linalg.norm(pts3d_corr, axis=1)
    # _outlier_thresh = np.median(_norms) * 5.0
    # _outlier_mask = _norms < _outlier_thresh
    # n_before_outlier = len(pts3d_corr)
    # pts2d_corr = pts2d_corr[_outlier_mask]
    # pts3d_corr = pts3d_corr[_outlier_mask]
    # conf_corr = conf_corr[_outlier_mask]
    # print(f"  3D outlier filter: {len(pts3d_corr)} / {n_before_outlier} kept")
    # if len(pts2d_corr) < 4:
    #     raise RuntimeError(f"Not enough correspondences after 3D outlier filtering: {len(pts2d_corr)}")
    # _t("6_uniform_sample_outlier")

    # ── 8. Gallery pose ───────────────────────────────────────────────────────
    render_idx = int(Path(best_render).stem)
    _pose_dict = getattr(model_cache, "gallery_pose_dict", None) if model_cache is not None else None
    if _pose_dict is not None:
        pose_record = _pose_dict.get(render_idx)
    else:
        # Fallback: load JSON and linear-scan (single-stage mode without ModelCache)
        gallery = load_json(args.gallery_pose_json)
        pose_record = next((p for p in gallery["poses"] if int(p["index"]) == render_idx), None)
    if pose_record is None:
        raise KeyError(f"Pose for render '{best_render}' not found in gallery_poses.json")

    R_gallery = np.array(pose_record["R_obj_to_cam"], dtype=np.float64)
    t_gallery = np.array(pose_record["t_obj_to_cam"], dtype=np.float64)
    _t("8_gallery_pose_lookup")

    # ── 9. Intrinsics ─────────────────────────────────────────────────────────
    K = load_intrinsics(args.intrinsics_path)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f"  K: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    # ── 10. Render / query masked image for no_pnp / t_refine ─────────────────
    render_width = int(getattr(args, "render_width", 3840))
    render_height = int(getattr(args, "render_height", 2160))
    gs_model_dir = getattr(args, "gs_model_dir", None)
    gs_iter = getattr(args, "gs_iter", None)
    gs_repo = getattr(args, "gs_repo", None)
    gs_python = getattr(args, "gs_python", None) or sys.executable
    bg_color = getattr(args, "bg_color", "0,0,0")
    gs_mode = getattr(args, "gs_mode", "2dgs")

    query_masked_img = None
    qmp = getattr(args, "query_masked_path", None)
    if qmp and Path(qmp).exists():
        query_masked_img = cv2.imread(str(qmp), cv2.IMREAD_COLOR)
    _t("10_img_load")

    # ── 11. PnP ───────────────────────────────────────────────────────────────
    reproj_thresh = float(getattr(args, "pnp_reproj_error", 5.0))
    no_pnp = bool(getattr(args, "no_pnp", False))

    if no_pnp:
        R_out = R_gallery.copy()
        inlier_idx = np.arange(len(pts2d_corr), dtype=np.int32)
        inlier_count = len(pts2d_corr)
        reproj_err = np.inf
        pose_method = "no_pnp_gallery_R"

        object_height_m_init = float(getattr(args, "object_height_m", 0.125))
        _qmask_init = make_query_reference_mask(
            query_masked_img=query_masked_img,
            query_mask_path=(out_dir / "query_mask.png"),
        )
        if _qmask_init is not None:
            tz_init = estimate_tz_from_mask_bbox(_qmask_init, fy=fy, object_height_m=object_height_m_init)
            q_stats_init = binary_mask_bbox_stats(_qmask_init)
            if tz_init is not None and q_stats_init is not None:
                tx_init = (q_stats_init["cx"] - cx) / fx * tz_init
                ty_init = (q_stats_init["cy"] - cy) / fy * tz_init
                t_out = np.array([tx_init, ty_init, tz_init], dtype=np.float64)
                print(f"  [no_pnp] bbox t_init: [{tx_init:.4f}, {ty_init:.4f}, {tz_init:.4f}]")
            else:
                t_out = t_gallery.copy()
        else:
            t_out = t_gallery.copy()
    else:
        R_out, t_out, pose_method, inlier_count, reproj_err, inlier_idx = solve_pose_pnp(
            pts2d_corr, pts3d_corr, K, R_init=R_gallery,
            reproj_thresh=reproj_thresh,
            use_ransac=not bool(getattr(args, "no_pnp_ransac", False)),
        )
    _t("11_pnp")

    # ── 12. Translation refinement ────────────────────────────────────────────
    xyz_refine_info = {"status": "not_run"}
    skip_t_refine = bool(getattr(args, "skip_t_refine", False))
    iou_accept_thresh = float(getattr(args, "t_refine_iou_thresh", 0.30))
    t_pnp = t_out.copy()

    if skip_t_refine:
        print("  [xyz refine] skipped (--skip_t_refine)")
        xyz_refine_info["status"] = "skipped_by_flag"
    elif gs_model_dir and gs_iter is not None and (gs_repo or (model_cache is not None and model_cache.gaussians is not None)):
        query_ref_mask = make_query_reference_mask(
            query_masked_img=query_masked_img,
            query_mask_path=(out_dir / "query_mask.png"),
        )
        if query_ref_mask is not None:
            object_height_m = float(getattr(args, "object_height_m", 0.125))
            _cached_gaussians = model_cache.gaussians if model_cache is not None else None
            t_refined, xyz_refine_info = refine_translation_xyz_with_mask(
                R=R_out, t_init=t_out, K=K,
                width=render_width, height=render_height,
                query_mask_img=query_ref_mask, object_height_m=object_height_m,
                gs_python=gs_python, gs_repo=gs_repo, gs_model_dir=gs_model_dir,
                gs_iter=gs_iter, intrinsics_path=args.intrinsics_path,
                bg_color=bg_color, gs_mode=gs_mode,
                gaussians=_cached_gaussians,
            )
            best_iou = (xyz_refine_info.get("best_metrics") or {}).get("iou", 0.0)
            t_ref = np.asarray(xyz_refine_info["t_refined"], dtype=np.float64)
            print(f"  [xyz refine] status={xyz_refine_info['status']}, best_iou={best_iou:.4f}")
            print(f"  [xyz refine] t_pnp=[{t_pnp[0]:.4f}, {t_pnp[1]:.4f}, {t_pnp[2]:.4f}]")
            print(f"  [xyz refine] t_ref=[{t_ref[0]:.4f}, {t_ref[1]:.4f}, {t_ref[2]:.4f}]")
            if best_iou >= iou_accept_thresh:
                t_out = t_refined
                xyz_refine_info["t_applied"] = True
                print(f"  [xyz refine] IoU={best_iou:.4f} >= {iou_accept_thresh:.2f} → t_refined applied")
            else:
                t_out = t_pnp
                xyz_refine_info["t_applied"] = False
                print(f"  [xyz refine] IoU={best_iou:.4f} < {iou_accept_thresh:.2f} → keep PnP t")
        else:
            print("  [xyz refine] no query mask, skipping")
            xyz_refine_info["status"] = "skipped_no_query_mask"
    else:
        print("  [xyz refine] GS render args missing, skipping")
        xyz_refine_info["status"] = "skipped_no_gs_args"

    _t("12_t_refine")
    # ── 13. Save ──────────────────────────────────────────────────────────────
    q_out = rotation_matrix_to_quaternion(R_out)
    pose_out = {
        "stage": "step6",
        "best_render": best_render,
        "n_loftr_inliers_input": int(len(mkpts0_840)),
        "n_valid_2d3d_corr": n_valid,
        "pnp_reproj_error_px": float(reproj_err) if np.isfinite(reproj_err) else None,
        "pose_method": pose_method,
        "pnp_inlier_count": int(inlier_count),
        "xyz_refine_status": xyz_refine_info.get("status"),
        "xyz_refine_best_iou": (xyz_refine_info.get("best_metrics") or {}).get("iou"),
        "R_gallery_init": R_gallery.tolist(),
        "R_obj_to_cam": R_out.tolist(),
        "t_obj_to_cam": t_out.tolist(),
        "quat_wxyz": q_out.tolist(),
        "query_img": str(args.query_img),
        "query_masked_path": str(getattr(args, "query_masked_path", "")),
        "intrinsics_path": str(args.intrinsics_path),
        # "query_mask_path": str(query_mask_path) if query_mask_path else None,
        "n_after_query_mask_filter": int(len(pts2d_corr)),
        # "n_inside_query_mask": n_inside_query_mask,
    }

    save_json(out_dir / "initial_pose.json", pose_out)

    _t("13_save")

    if _TIMING:
        print("  [Step6 sub-timing]")
        for i in range(1, len(_ts)):
            label, t1 = _ts[i]
            _, t0 = _ts[i - 1]
            print(f"    {label:<30} {(t1 - t0)*1000:6.1f}ms")
        total = _ts[-1][1] - _ts[0][1]
        print(f"    {'TOTAL':<30} {total*1000:6.1f}ms")

    print("=" * 60)
    print("[Step 6 RT] PnP translation estimation complete")
    print(f"  best_render  : {best_render}")
    print(f"  pose_method  : {pose_method}")
    print(f"  n_corr       : {n_valid}")
    if np.isfinite(reproj_err):
        print(f"  reproj_err   : {reproj_err:.2f}px")
    print(f"  R_obj_to_cam :\n{R_out}")
    print(f"  t_obj_to_cam : [{t_out[0]:.8f}, {t_out[1]:.8f}, {t_out[2]:.8f}]")
    print(f"  initial_pose.json: {out_dir / 'initial_pose.json'}")
    print("=" * 60)
