import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import csv

import cv2
import numpy as np


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


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


def unmap_from_square_resize(pts_resized, orig_hw, resize_target=840):
    h, w = orig_hw
    side = max(h, w)
    x0 = (side - w) // 2
    y0 = (side - h) // 2

    pts_square = np.asarray(pts_resized, dtype=np.float64) * (side / resize_target)
    pts_crop = pts_square - np.array([[x0, y0]], dtype=np.float64)
    return pts_crop


def to_full_image_coords(pts_crop, nonblack_bbox_xyxy):
    x1, y1 = float(nonblack_bbox_xyxy[0]), float(nonblack_bbox_xyxy[1])
    pts = np.asarray(pts_crop, dtype=np.float64) + np.array([[x1, y1]])
    return pts


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
        for i in range(N):
            uf, vf = pts_uv[i, 0], pts_uv[i, 1]
            u0, v0 = int(math.floor(uf)), int(math.floor(vf))
            u1, v1 = u0 + 1, v0 + 1
            if u0 < 0 or v0 < 0 or u1 >= W or v1 >= H:
                continue
            du, dv = uf - u0, vf - v0
            val = (xyz_map[v0, u0] * (1-du) * (1-dv) +
                   xyz_map[v0, u1] * du * (1-dv) +
                   xyz_map[v1, u0] * (1-du) * dv +
                   xyz_map[v1, u1] * du * dv)
            pts3d[i] = val.astype(np.float64)

    finite = np.all(np.isfinite(pts3d), axis=1)
    nonzero = np.abs(pts3d).sum(axis=1) > 1e-6
    valid = finite & nonzero
    return pts3d, valid


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


def _save_correspondence_debug(out_dir, query_img, gallery_render_path,
                                pts2d_query, pts3d_obj, pts2d_gallery,
                                R, t, K, inlier_idx=None):
    try:
        import cv2 as _cv2
        import numpy as _np

        N = len(pts2d_query)
        if N == 0:
            return

        if inlier_idx is None or len(inlier_idx) == 0:
            inlier_mask = _np.ones(N, dtype=bool)
        else:
            inlier_mask = _np.zeros(N, dtype=bool)
            inlier_mask[inlier_idx] = True

        n_inliers  = int(inlier_mask.sum())
        n_outliers = N - n_inliers

        inlier_indices = _np.where(inlier_mask)[0]
        inlier_colors = [
            tuple(int(c) for c in _cv2.applyColorMap(
                _np.array([[int(k * 180 / max(n_inliers - 1, 1))]], dtype=_np.uint8),
                _cv2.COLORMAP_HSV
            )[0, 0])
            for k in range(n_inliers)
        ]
        OUTLIER_COLOR = (100, 100, 100)

        dist = _np.zeros((4, 1))
        rvec, _ = _cv2.Rodrigues(_np.asarray(R, dtype=_np.float64))
        tvec = _np.asarray(t, dtype=_np.float64).reshape(3, 1)
        pts3d_inliers = pts3d_obj[inlier_mask]
        pts2d_inliers = pts2d_query[inlier_mask]

        if len(pts3d_inliers) > 0:
            proj_in, _ = _cv2.projectPoints(
                pts3d_inliers.astype(_np.float32), rvec, tvec,
                K.astype(_np.float64), dist
            )
            pts2d_reproj_in = proj_in.reshape(-1, 2)
            per_point_err = _np.linalg.norm(pts2d_reproj_in - pts2d_inliers, axis=1)
            mean_inlier_err = float(per_point_err.mean())
        else:
            pts2d_reproj_in = _np.empty((0, 2))
            per_point_err   = _np.empty(0)
            mean_inlier_err = float("nan")

        q_vis = query_img.copy()

        for i in _np.where(~inlier_mask)[0]:
            pu, pv = int(round(pts2d_query[i, 0])), int(round(pts2d_query[i, 1]))
            _cv2.circle(q_vis, (pu, pv), 4, OUTLIER_COLOR, -1)

        for k, (i, c) in enumerate(zip(inlier_indices, inlier_colors)):
            pu, pv = int(round(pts2d_query[i, 0])), int(round(pts2d_query[i, 1]))
            ru, rv = int(round(pts2d_reproj_in[k, 0])), int(round(pts2d_reproj_in[k, 1]))
            _cv2.circle(q_vis, (pu, pv), 7, c, -1)
            _cv2.drawMarker(q_vis, (ru, rv), (0, 0, 255), _cv2.MARKER_CROSS, 14, 2)
            _cv2.line(q_vis, (pu, pv), (ru, rv), (0, 0, 255), 1)
            _cv2.putText(q_vis, f"{per_point_err[k]:.1f}",
                         (pu + 8, pv - 4), _cv2.FONT_HERSHEY_SIMPLEX,
                         0.4, (255, 255, 255), 1, _cv2.LINE_AA)

        _cv2.putText(q_vis,
                     f"inliers={n_inliers}/{N}  reproj_inlier={mean_inlier_err:.2f}px",
                     (10, 30), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, _cv2.LINE_AA)
        _cv2.imwrite(str(out_dir / "step6_corr_query.png"), q_vis)

        gallery_img = _cv2.imread(str(gallery_render_path))
        if gallery_img is not None:
            g_vis = gallery_img.copy()

            for i in _np.where(~inlier_mask)[0]:
                gu, gv = int(round(pts2d_gallery[i, 0])), int(round(pts2d_gallery[i, 1]))
                _cv2.circle(g_vis, (gu, gv), 4, OUTLIER_COLOR, -1)

            for k, (i, c) in enumerate(zip(inlier_indices, inlier_colors)):
                gu, gv = int(round(pts2d_gallery[i, 0])), int(round(pts2d_gallery[i, 1]))
                _cv2.circle(g_vis, (gu, gv), 7, c, -1)

            _cv2.putText(g_vis,
                         f"inliers={n_inliers}/{N}",
                         (10, 30), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, _cv2.LINE_AA)
            _cv2.imwrite(str(out_dir / "step6_corr_gallery.png"), g_vis)

            qh, qw = q_vis.shape[:2]
            gh, gw = g_vis.shape[:2]
            scale = qh / gh
            g_resized = _cv2.resize(g_vis, (int(gw * scale), qh))
            offset = g_resized.shape[1] + 4
            canvas = _np.zeros((qh, qw + offset, 3), dtype=_np.uint8)
            canvas[:, :g_resized.shape[1]] = g_resized
            canvas[:, offset:] = q_vis

            for k, (i, c) in enumerate(zip(inlier_indices, inlier_colors)):
                gu_r = int(round(pts2d_gallery[i, 0] * scale))
                gv_r = int(round(pts2d_gallery[i, 1] * scale))
                qu_r = int(round(pts2d_query[i, 0])) + offset
                qv_r = int(round(pts2d_query[i, 1]))
                _cv2.line(canvas, (gu_r, gv_r), (qu_r, qv_r), c, 1)

            _cv2.putText(canvas,
                         f"RANSAC inliers={n_inliers}/{N}  reproj={mean_inlier_err:.2f}px  outliers(gray)={n_outliers}",
                         (10, canvas.shape[0] - 10), _cv2.FONT_HERSHEY_SIMPLEX,
                         0.6, (0, 255, 255), 2, _cv2.LINE_AA)
            _cv2.imwrite(str(out_dir / "step6_corr_sidebyside.png"), canvas)

        print(f"  [Corr debug] saved: step6_corr_query/gallery/sidebyside.png  "
              f"(N={N}, inliers={n_inliers}, reproj_inlier={mean_inlier_err:.2f}px)")
    except Exception as _e:
        import traceback as _tb
        print(f"  [Corr debug] failed: {_e}")
        _tb.print_exc()


def solve_pose_pnp(pts2d, pts3d, K, R_init,
                   reproj_thresh=200.0, min_inliers=6,
                   use_ransac=True):
    N = len(pts2d)
    if N < 4:
        print(f"  [PnP] 2D-3D 대응점 {N}개 < 4, pose 추정 불가")
        t_linear = estimate_t_linear(pts2d, pts3d, K, R_init) if N > 0 else np.zeros(3, dtype=np.float64)
        return R_init, t_linear, "linear_T_fixed_R_insufficient_points", 0, np.inf, np.array([], dtype=np.int32)

    dist = np.zeros((4, 1), dtype=np.float64)

    if not use_ransac:
        rvec_init, _ = cv2.Rodrigues(R_init.astype(np.float64))
        t_init_linear = estimate_t_linear(pts2d, pts3d, K, R_init)

        retval, rvec, tvec = cv2.solvePnP(
            pts3d.astype(np.float32),
            pts2d.astype(np.float32),
            K.astype(np.float64),
            dist,
            rvec=rvec_init,
            tvec=t_init_linear.reshape(3, 1).astype(np.float64),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not retval:
            print(f"  [PnP] No-RANSAC ITERATIVE failed. Using linear T.")
            return R_init, t_init_linear, "linear_T_fixed_R_no_ransac_failed", 0, np.inf, np.array([], dtype=np.int32)

        R_out, _ = cv2.Rodrigues(rvec)
        t_out = tvec.ravel()
        inlier_idx = np.arange(N, dtype=np.int32)

        proj, _ = cv2.projectPoints(
            pts3d.astype(np.float32), rvec, tvec,
            K.astype(np.float64), dist
        )
        err = np.linalg.norm(proj.reshape(-1, 2) - pts2d, axis=1).mean()

        print(f"  [PnP] No-RANSAC: all {N} pts used, reproj_err={err:.2f}px")
        return (
            R_out.astype(np.float64),
            t_out.astype(np.float64),
            "pnp_no_ransac_iterative",
            N,
            float(err),
            inlier_idx,
        )

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d.astype(np.float32),
        pts2d.astype(np.float32),
        K.astype(np.float64),
        dist,
        useExtrinsicGuess=False,
        iterationsCount=500,
        reprojectionError=reproj_thresh,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )

    print(f" [PnP] rvec (rad): {rvec.ravel()}  tvec: {tvec.ravel()}")

    stage1_method = "epnp"

    if not success or inliers is None or len(inliers) < min_inliers:
        print(f"  [PnP] Stage 1 (EPnP) failed or insufficient inliers "
              f"({0 if inliers is None else len(inliers)}). Using linear T.")
        t_linear = estimate_t_linear(pts2d, pts3d, K, R_init)
        return R_init, t_linear, "linear_T_fixed_R", 0, np.inf, np.array([], dtype=np.int32)

    inlier_idx = inliers.ravel().astype(np.int32)
    print(f"  [PnP] Stage 1 ({stage1_method}): {len(inlier_idx)} inliers / {N} pts")

    if len(inlier_idx) >= 6:
        pts3d_in = pts3d[inlier_idx].astype(np.float32)
        pts2d_in = pts2d[inlier_idx].astype(np.float32)

        retval, rvec_ref, tvec_ref = cv2.solvePnP(
            pts3d_in,
            pts2d_in,
            K.astype(np.float64),
            dist,
            rvec=rvec.copy(),
            tvec=tvec.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if retval:
            rvec, tvec = rvec_ref, tvec_ref
            print(f"  [PnP] Stage 2 (ITERATIVE refine on {len(inlier_idx)} inliers): OK")
        else:
            print(f"  [PnP] Stage 2 refine failed, using Stage 1 result")

    print(f" [PnP] rvec (rad): {rvec.ravel()}  tvec: {tvec.ravel()}")

    R_out, _ = cv2.Rodrigues(rvec)
    t_out = tvec.ravel()

    pts3d_in = pts3d[inlier_idx]
    pts2d_in = pts2d[inlier_idx]
    proj, _ = cv2.projectPoints(
        pts3d_in.astype(np.float32), rvec, tvec,
        K.astype(np.float64), dist
    )
    err = np.linalg.norm(proj.reshape(-1, 2) - pts2d_in, axis=1).mean()

    print(f"  [PnP] Final: {len(inlier_idx)} inliers, reproj_err={err:.2f}px")

    return (
        R_out.astype(np.float64),
        t_out.astype(np.float64),
        f"pnp_2stage_{stage1_method}",
        len(inlier_idx),
        float(err),
        inlier_idx,
    )


def project_axes_overlay(img_bgr, K, R, t, axis_len_m, out_path=None):
    img = img_bgr.copy()
    obj_pts = np.array([
        [0, 0, 0],
        [axis_len_m, 0, 0],
        [0, axis_len_m, 0],
        [0, 0, axis_len_m],
    ], dtype=np.float32)

    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec = t.reshape(3, 1).astype(np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    imgpts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K.astype(np.float64), dist)
    imgpts = np.round(imgpts.reshape(-1, 2)).astype(int)

    o = tuple(imgpts[0])
    cv2.line(img, o, tuple(imgpts[1]), (0, 0, 255), 4, cv2.LINE_AA)
    cv2.line(img, o, tuple(imgpts[2]), (0, 255, 0), 4, cv2.LINE_AA)
    cv2.line(img, o, tuple(imgpts[3]), (255, 0, 0), 4, cv2.LINE_AA)
    cv2.circle(img, o, 6, (255, 255, 255), -1, cv2.LINE_AA)

    if out_path is not None:
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), img)
    return img


def load_ply_xyz(ply_path):
    from plyfile import PlyData
    ply = PlyData.read(str(ply_path))
    v   = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    return xyz


def rotation_geodesic_deg(R_a, R_b):
    R_a = np.asarray(R_a, dtype=np.float64)
    R_b = np.asarray(R_b, dtype=np.float64)
    R_rel = R_a @ R_b.T
    cos_theta = (np.trace(R_rel) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def build_same_corr_motion_trace(
    pts2d_query_obs,
    pts2d_gallery_obs,
    pts3d_obj,
    K,
    R_before,
    t_before,
    R_after,
    t_after,
):
    pts2d_query_obs = np.asarray(pts2d_query_obs, dtype=np.float64)
    pts2d_gallery_obs = np.asarray(pts2d_gallery_obs, dtype=np.float64)
    pts3d_obj = np.asarray(pts3d_obj, dtype=np.float64)

    proj_before = project_points_obj_to_img(pts3d_obj, K, R_before, t_before)
    proj_after  = project_points_obj_to_img(pts3d_obj, K, R_after,  t_after)

    cam_before = compute_camera_coords_of_points(pts3d_obj, R_before, t_before)
    cam_after  = compute_camera_coords_of_points(pts3d_obj, R_after,  t_after)

    delta_uv = proj_after - proj_before
    delta_uv_px = np.linalg.norm(delta_uv, axis=1)

    err_before_px = np.linalg.norm(proj_before - pts2d_query_obs, axis=1)
    err_after_px  = np.linalg.norm(proj_after  - pts2d_query_obs, axis=1)
    improvement_px = err_before_px - err_after_px
    gallery_reproj_err_px = np.linalg.norm(proj_before - pts2d_gallery_obs, axis=1)
    gallery_to_pnp_motion_px = np.linalg.norm(proj_after - proj_before, axis=1)

    delta_cam = cam_after - cam_before
    delta_cam_m = np.linalg.norm(delta_cam, axis=1)

    rows = []
    for i in range(len(pts3d_obj)):
        rows.append({
            "point_id": int(i),
            "query_uv": [float(pts2d_query_obs[i, 0]), float(pts2d_query_obs[i, 1])],
            "gallery_uv": [float(pts2d_gallery_obs[i, 0]), float(pts2d_gallery_obs[i, 1])],
            "xyz_obj": [float(pts3d_obj[i, 0]), float(pts3d_obj[i, 1]), float(pts3d_obj[i, 2])],
            "before_uv": [float(proj_before[i, 0]), float(proj_before[i, 1])],
            "after_uv": [float(proj_after[i, 0]), float(proj_after[i, 1])],
            "delta_uv": [float(delta_uv[i, 0]), float(delta_uv[i, 1])],
            "delta_uv_px": float(delta_uv_px[i]),
            "err_before_px": float(err_before_px[i]),
            "err_after_px": float(err_after_px[i]),
            "improvement_px": float(improvement_px[i]),
            "gallery_reproj_err_px": float(gallery_reproj_err_px[i]),
            "gallery_to_pnp_motion_px": float(gallery_to_pnp_motion_px[i]),
            "cam_before": [float(cam_before[i, 0]), float(cam_before[i, 1]), float(cam_before[i, 2])],
            "cam_after": [float(cam_after[i, 0]), float(cam_after[i, 1]), float(cam_after[i, 2])],
            "delta_cam": [float(delta_cam[i, 0]), float(delta_cam[i, 1]), float(delta_cam[i, 2])],
            "delta_cam_m": float(delta_cam_m[i]),
        })

    return {
        "num_points": int(len(rows)),
        "summary": {
            "mean_delta_uv_px": float(np.mean(delta_uv_px)) if len(delta_uv_px) > 0 else None,
            "median_delta_uv_px": float(np.median(delta_uv_px)) if len(delta_uv_px) > 0 else None,
            "max_delta_uv_px": float(np.max(delta_uv_px)) if len(delta_uv_px) > 0 else None,
            "mean_err_before_px": float(np.mean(err_before_px)) if len(err_before_px) > 0 else None,
            "mean_err_after_px": float(np.mean(err_after_px)) if len(err_after_px) > 0 else None,
            "mean_improvement_px": float(np.mean(improvement_px)) if len(improvement_px) > 0 else None,
            "median_improvement_px": float(np.median(improvement_px)) if len(improvement_px) > 0 else None,
            "max_improvement_px": float(np.max(improvement_px)) if len(improvement_px) > 0 else None,
            "min_improvement_px": float(np.min(improvement_px)) if len(improvement_px) > 0 else None,
            "mean_delta_cam_m": float(np.mean(delta_cam_m)) if len(delta_cam_m) > 0 else None,
            "max_delta_cam_m": float(np.max(delta_cam_m)) if len(delta_cam_m) > 0 else None,
            "mean_gallery_reproj_err_px": float(np.mean(gallery_reproj_err_px)) if len(gallery_reproj_err_px) > 0 else None,
            "median_gallery_reproj_err_px": float(np.median(gallery_reproj_err_px)) if len(gallery_reproj_err_px) > 0 else None,
            "max_gallery_reproj_err_px": float(np.max(gallery_reproj_err_px)) if len(gallery_reproj_err_px) > 0 else None,
            "mean_gallery_to_pnp_motion_px": float(np.mean(gallery_to_pnp_motion_px)) if len(gallery_to_pnp_motion_px) > 0 else None,
            "median_gallery_to_pnp_motion_px": float(np.median(gallery_to_pnp_motion_px)) if len(gallery_to_pnp_motion_px) > 0 else None,
            "max_gallery_to_pnp_motion_px": float(np.max(gallery_to_pnp_motion_px)) if len(gallery_to_pnp_motion_px) > 0 else None,
        },
        "rows": rows,
    }


def save_same_corr_motion_csv(csv_path, trace):
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)

    fieldnames = [
        "point_id",
        "query_u", "query_v",
        "gallery_u", "gallery_v",
        "x_obj", "y_obj", "z_obj",
        "before_u", "before_v",
        "after_u", "after_v",
        "delta_u", "delta_v", "delta_uv_px",
        "err_before_px", "err_after_px", "improvement_px",
        "gallery_reproj_err_px", "gallery_to_pnp_motion_px",
        "cam_before_x", "cam_before_y", "cam_before_z",
        "cam_after_x", "cam_after_y", "cam_after_z",
        "delta_cam_x", "delta_cam_y", "delta_cam_z",
        "delta_cam_m",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in trace["rows"]:
            writer.writerow({
                "point_id": row["point_id"],
                "query_u": row["query_uv"][0],
                "query_v": row["query_uv"][1],
                "gallery_u": row["gallery_uv"][0],
                "gallery_v": row["gallery_uv"][1],
                "x_obj": row["xyz_obj"][0],
                "y_obj": row["xyz_obj"][1],
                "z_obj": row["xyz_obj"][2],
                "before_u": row["before_uv"][0],
                "before_v": row["before_uv"][1],
                "after_u": row["after_uv"][0],
                "after_v": row["after_uv"][1],
                "delta_u": row["delta_uv"][0],
                "delta_v": row["delta_uv"][1],
                "delta_uv_px": row["delta_uv_px"],
                "err_before_px": row["err_before_px"],
                "err_after_px": row["err_after_px"],
                "improvement_px": row["improvement_px"],
                "gallery_reproj_err_px": row["gallery_reproj_err_px"],
                "gallery_to_pnp_motion_px": row["gallery_to_pnp_motion_px"],
                "cam_before_x": row["cam_before"][0],
                "cam_before_y": row["cam_before"][1],
                "cam_before_z": row["cam_before"][2],
                "cam_after_x": row["cam_after"][0],
                "cam_after_y": row["cam_after"][1],
                "cam_after_z": row["cam_after"][2],
                "delta_cam_x": row["delta_cam"][0],
                "delta_cam_y": row["delta_cam"][1],
                "delta_cam_z": row["delta_cam"][2],
                "delta_cam_m": row["delta_cam_m"],
            })


def augment_same_corr_trace_with_postrender_surface(
    trace,
    post_xyz_map,
    xyz_err_bad_thresh_m=0.005,
):
    rows = trace["rows"]
    if len(rows) == 0:
        return {
            "num_points": 0,
            "summary": {},
            "rows": [],
        }

    after_uv = np.asarray([r["after_uv"] for r in rows], dtype=np.float64)
    xyz_obj  = np.asarray([r["xyz_obj"]  for r in rows], dtype=np.float64)

    post_xyz_lookup, valid_lookup = lookup_xyz_at_pixels(
        post_xyz_map,
        after_uv,
        bilinear=True,
    )

    surface_err_vec = np.zeros_like(xyz_obj, dtype=np.float64)
    surface_err_m = np.full((len(rows),), np.nan, dtype=np.float64)

    if np.any(valid_lookup):
        surface_err_vec[valid_lookup] = post_xyz_lookup[valid_lookup] - xyz_obj[valid_lookup]
        surface_err_m[valid_lookup] = np.linalg.norm(surface_err_vec[valid_lookup], axis=1)

    aug_rows = []
    for i, r in enumerate(rows):
        rr = dict(r)
        rr["post_xyz_lookup"] = [
            float(post_xyz_lookup[i, 0]),
            float(post_xyz_lookup[i, 1]),
            float(post_xyz_lookup[i, 2]),
        ]
        rr["post_xyz_valid"] = bool(valid_lookup[i])
        rr["surface_err_vec"] = [
            float(surface_err_vec[i, 0]),
            float(surface_err_vec[i, 1]),
            float(surface_err_vec[i, 2]),
        ]
        rr["surface_err_m"] = None if not np.isfinite(surface_err_m[i]) else float(surface_err_m[i])
        rr["surface_err_mm"] = None if not np.isfinite(surface_err_m[i]) else float(surface_err_m[i] * 1000.0)
        rr["surface_bad"] = bool(np.isfinite(surface_err_m[i]) and surface_err_m[i] > float(xyz_err_bad_thresh_m))
        aug_rows.append(rr)

    valid_err = surface_err_m[np.isfinite(surface_err_m)]

    out = {
        "num_points": int(len(rows)),
        "summary": dict(trace.get("summary", {})),
        "rows": aug_rows,
    }

    out["summary"].update({
        "num_post_xyz_valid": int(np.sum(valid_lookup)),
        "num_surface_bad": int(np.sum([
            1 for r in aug_rows if r["surface_bad"]
        ])),
        "xyz_err_bad_thresh_m": float(xyz_err_bad_thresh_m),
        "mean_surface_err_m": float(np.mean(valid_err)) if len(valid_err) > 0 else None,
        "median_surface_err_m": float(np.median(valid_err)) if len(valid_err) > 0 else None,
        "max_surface_err_m": float(np.max(valid_err)) if len(valid_err) > 0 else None,
        "min_surface_err_m": float(np.min(valid_err)) if len(valid_err) > 0 else None,
    })

    return out


def save_same_corr_surface_csv(csv_path, trace_aug):
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)

    fieldnames = [
        "point_id",
        "query_u", "query_v",
        "gallery_u", "gallery_v",
        "after_u", "after_v",
        "x_obj", "y_obj", "z_obj",
        "post_x", "post_y", "post_z",
        "post_xyz_valid",
        "surface_err_mm",
        "surface_bad",
        "err_after_px",
        "improvement_px",
        "delta_uv_px",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in trace_aug["rows"]:
            px, py, pz = row["post_xyz_lookup"]
            writer.writerow({
                "point_id": row["point_id"],
                "query_u": row["query_uv"][0],
                "query_v": row["query_uv"][1],
                "gallery_u": row["gallery_uv"][0],
                "gallery_v": row["gallery_uv"][1],
                "after_u": row["after_uv"][0],
                "after_v": row["after_uv"][1],
                "x_obj": row["xyz_obj"][0],
                "y_obj": row["xyz_obj"][1],
                "z_obj": row["xyz_obj"][2],
                "post_x": px,
                "post_y": py,
                "post_z": pz,
                "post_xyz_valid": row["post_xyz_valid"],
                "surface_err_mm": row["surface_err_mm"],
                "surface_bad": row["surface_bad"],
                "err_after_px": row["err_after_px"],
                "improvement_px": row["improvement_px"],
                "delta_uv_px": row["delta_uv_px"],
            })


def draw_ply_overlay(query_img, ply_xyz, K, R, t,
                     max_points=8000, alpha=0.5,
                     point_color=(0, 255, 255), point_radius=2,
                     out_path=None):
    base    = query_img.copy()
    overlay = query_img.copy()
    h, w    = overlay.shape[:2]

    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec    = t.reshape(3, 1).astype(np.float64)
    dist    = np.zeros((4, 1), dtype=np.float64)

    xyz = np.asarray(ply_xyz, dtype=np.float32)
    if len(xyz) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        xyz = xyz[idx]

    imgpts, _ = cv2.projectPoints(xyz, rvec, tvec, K.astype(np.float64), dist)
    imgpts = imgpts.reshape(-1, 2)

    kept = 0
    for p in imgpts:
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(overlay, (x, y), point_radius, point_color, -1, cv2.LINE_AA)
            kept += 1

    result = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)

    if out_path is not None:
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), result)
        print(f"  [PLY overlay] projected {kept}/{len(imgpts)} pts -> {Path(out_path).name}")

    return result, kept


def project_points_obj_to_img(pts3d_obj, K, R, t):
    pts3d_obj = np.asarray(pts3d_obj, dtype=np.float64)
    if len(pts3d_obj) == 0:
        return np.empty((0, 2), dtype=np.float64)

    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec = t.reshape(3, 1).astype(np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    proj, _ = cv2.projectPoints(
        pts3d_obj.astype(np.float32),
        rvec,
        tvec,
        K.astype(np.float64),
        dist,
    )
    return proj.reshape(-1, 2).astype(np.float64)


def compute_camera_coords_of_points(pts3d_obj, R, t):
    pts3d_obj = np.asarray(pts3d_obj, dtype=np.float64)
    if len(pts3d_obj) == 0:
        return np.empty((0, 3), dtype=np.float64)
    return (R @ pts3d_obj.T).T + np.asarray(t, dtype=np.float64).reshape(1, 3)


def compute_reprojection_stats(pts2d_obs, pts2d_proj):
    pts2d_obs = np.asarray(pts2d_obs, dtype=np.float64)
    pts2d_proj = np.asarray(pts2d_proj, dtype=np.float64)

    if len(pts2d_obs) == 0 or len(pts2d_proj) == 0:
        return {
            "num_points": 0,
            "mean_px": None,
            "median_px": None,
            "max_px": None,
            "min_px": None,
        }

    d = np.linalg.norm(pts2d_proj - pts2d_obs, axis=1)
    return {
        "num_points": int(len(d)),
        "mean_px": float(np.mean(d)),
        "median_px": float(np.median(d)),
        "max_px": float(np.max(d)),
        "min_px": float(np.min(d)),
    }


def draw_correspondence_debug_single_pose(
    query_img,
    pts2d_query,
    pts3d_obj,
    K,
    R,
    t,
    out_path,
    draw_idx=None,
    max_draw=100,
    observed_color=(255, 100, 0),
    reproj_color=(0, 255, 255),
    line_color=(180, 180, 180),
    title_prefix="",
    stats_json_path=None,
):
    img = query_img.copy()
    h, w = img.shape[:2]

    pts2d_query = np.asarray(pts2d_query, dtype=np.float64)
    pts3d_obj = np.asarray(pts3d_obj, dtype=np.float64)

    if len(pts2d_query) != len(pts3d_obj):
        raise ValueError(
            f"Length mismatch: len(pts2d_query)={len(pts2d_query)} "
            f"vs len(pts3d_obj)={len(pts3d_obj)}"
        )

    if len(pts2d_query) == 0:
        cv2.putText(img, "No correspondences", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), img)
        if stats_json_path is not None:
            save_json(stats_json_path, {
                "title_prefix": title_prefix,
                "num_points_total": 0,
                "num_points_drawn": 0,
                "reprojection_stats_all": {
                    "num_points": 0,
                    "mean_px": None,
                    "median_px": None,
                    "max_px": None,
                    "min_px": None,
                },
            })
        return

    proj_all = project_points_obj_to_img(pts3d_obj, K, R, t)
    stats_all = compute_reprojection_stats(pts2d_query, proj_all)

    if draw_idx is None:
        N = min(len(pts2d_query), max_draw)
        rng = np.random.default_rng(42)
        draw_idx = rng.choice(len(pts2d_query), size=N, replace=False)
    else:
        draw_idx = np.asarray(draw_idx, dtype=np.int32)

    pts2d_draw = pts2d_query[draw_idx]
    proj_draw = proj_all[draw_idx]
    stats_draw = compute_reprojection_stats(pts2d_draw, proj_draw)

    drawn = 0
    for i in range(len(draw_idx)):
        px_q = tuple(np.round(pts2d_draw[i]).astype(int))
        px_p = tuple(np.round(proj_draw[i]).astype(int))

        in_bounds_q = 0 <= px_q[0] < w and 0 <= px_q[1] < h
        in_bounds_p = 0 <= px_p[0] < w and 0 <= px_p[1] < h

        if in_bounds_q and in_bounds_p:
            cv2.line(img, px_q, px_p, line_color, 1, cv2.LINE_AA)
        if in_bounds_q:
            cv2.circle(img, px_q, 4, observed_color, -1, cv2.LINE_AA)
        if in_bounds_p:
            cv2.circle(img, px_p, 4, reproj_color, -1, cv2.LINE_AA)

        if in_bounds_q or in_bounds_p:
            drawn += 1

    y = 25
    if title_prefix:
        cv2.putText(img, title_prefix, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28

    cv2.circle(img, (20, y - 5), 6, observed_color, -1)
    cv2.putText(img, "observed query 2D", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    y += 24

    cv2.circle(img, (20, y - 5), 6, reproj_color, -1)
    cv2.putText(img, "reprojected 3D->2D", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    y += 24

    cv2.line(img, (20, y - 5), (28, y - 5), line_color, 1, cv2.LINE_AA)
    cv2.putText(img, "observed <-> reprojection", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    y += 28

    cv2.putText(img, f"drawn={drawn}/{len(draw_idx)}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(img, f"mean={stats_draw['mean_px']:.2f}px  med={stats_draw['median_px']:.2f}px  max={stats_draw['max_px']:.2f}px",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), img)

    if stats_json_path is not None:
        save_json(stats_json_path, {
            "title_prefix": title_prefix,
            "num_points_total": int(len(pts2d_query)),
            "num_points_drawn": int(len(draw_idx)),
            "draw_idx": draw_idx.tolist(),
            "reprojection_stats_all": stats_all,
            "reprojection_stats_drawn_subset": stats_draw,
        })


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
    x1 = float(np.min(pts2d[:, 0]))
    y1 = float(np.min(pts2d[:, 1]))
    x2 = float(np.max(pts2d[:, 0]))
    y2 = float(np.max(pts2d[:, 1]))

    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad

    if img_w is not None:
        x1 = max(0.0, x1)
        x2 = min(float(img_w - 1), x2)
    if img_h is not None:
        y1 = max(0.0, y1)
        y2 = min(float(img_h - 1), y2)

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

    col_ids = np.floor((pts2d[:, 0] - x1) / width * grid_cols).astype(int)
    row_ids = np.floor((pts2d[:, 1] - y1) / height * grid_rows).astype(int)

    col_ids = np.clip(col_ids, 0, grid_cols - 1)
    row_ids = np.clip(row_ids, 0, grid_rows - 1)

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

    if len(keep) == 0:
        return np.array([], dtype=np.int32), cell_stats

    keep_idx = np.concatenate(keep, axis=0)
    keep_idx = np.unique(keep_idx)
    return keep_idx.astype(np.int32), cell_stats


def compute_mean_reprojection_error(pts2d, pts3d, K, R, t):
    pts2d = np.asarray(pts2d, dtype=np.float64)
    pts3d = np.asarray(pts3d, dtype=np.float64)

    if len(pts2d) == 0 or len(pts3d) == 0:
        return {
            "mean_px": None,
            "median_px": None,
            "max_px": None,
            "num_points": 0,
        }

    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec = t.reshape(3, 1).astype(np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    proj, _ = cv2.projectPoints(
        pts3d.astype(np.float32),
        rvec,
        tvec,
        K.astype(np.float64),
        dist,
    )
    proj = proj.reshape(-1, 2)

    d = np.linalg.norm(proj - pts2d, axis=1)

    return {
        "mean_px": float(np.mean(d)),
        "median_px": float(np.median(d)),
        "max_px": float(np.max(d)),
        "num_points": int(len(d)),
    }


def save_camera_pose_json(path, K, R, t, width, height):
    data = {
        "width": int(width),
        "height": int(height),
        "K": np.asarray(K, dtype=np.float64).tolist(),
        "R_obj_to_cam": np.asarray(R, dtype=np.float64).tolist(),
        "t_obj_to_cam": np.asarray(t, dtype=np.float64).reshape(3).tolist(),
    }
    save_json(path, data)


def overlay_render_on_query(query_img, render_img, out_path, alpha=0.45, nonblack_thresh=8):
    q = query_img.copy()
    r = render_img.copy()

    if q.shape[:2] != r.shape[:2]:
        r = cv2.resize(r, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    mask = gray > nonblack_thresh

    out = q.copy()
    out_float = out.astype(np.float32)
    r_float = r.astype(np.float32)

    out_float[mask] = (1.0 - alpha) * out_float[mask] + alpha * r_float[mask]
    out = np.clip(out_float, 0, 255).astype(np.uint8)

    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), out)
    return out


def render_to_binary_mask(render_img_bgr, nonblack_thresh=8):
    gray = cv2.cvtColor(render_img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > nonblack_thresh).astype(np.uint8) * 255
    return mask


def points_inside_binary_mask(pts2d, mask_img):
    pts2d = np.asarray(pts2d, dtype=np.float64)
    h, w = mask_img.shape[:2]

    u = np.round(pts2d[:, 0]).astype(int)
    v = np.round(pts2d[:, 1]).astype(int)

    inside = np.zeros(len(pts2d), dtype=bool)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    inside[valid] = mask_img[v[valid], u[valid]] > 0
    return inside


def get_outside_after_inlier_indices(
    pts3d_inliers,
    K,
    R_after,
    t_after,
    render_img_after,
    nonblack_thresh=8,
):
    if len(pts3d_inliers) == 0:
        return np.array([], dtype=np.int32), np.empty((0, 2), dtype=np.float64)

    after_mask = render_to_binary_mask(render_img_after, nonblack_thresh=nonblack_thresh)
    proj_after = project_points_obj_to_img(pts3d_inliers, K, R_after, t_after)
    inside_after = points_inside_binary_mask(proj_after, after_mask)
    bad_local_idx = np.where(~inside_after)[0].astype(np.int32)
    return bad_local_idx, proj_after


def binary_mask_iou(mask_a, mask_b):
    a = (mask_a > 0)
    b = (mask_b > 0)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def binary_mask_bbox_stats(mask_img):
    mask = (mask_img > 0)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    w = float(x2 - x1 + 1)
    h = float(y2 - y1 + 1)
    area = float(mask.sum())
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "w": w, "h": h, "area": area,
        "cx": float(cx), "cy": float(cy),
    }


def estimate_tz_from_mask_bbox(query_mask_img, fy, object_height_m):
    stats = binary_mask_bbox_stats(query_mask_img)
    if stats is None:
        return None
    bbox_h = max(1.0, stats["h"])
    tz = float(fy) * float(object_height_m) / bbox_h
    return float(tz)


def make_query_reference_mask(query_masked_img=None, query_mask_path=None, nonblack_thresh=8):
    if query_mask_path is not None:
        query_mask_path = Path(query_mask_path)
        if query_mask_path.exists():
            qmask = cv2.imread(str(query_mask_path), cv2.IMREAD_GRAYSCALE)
            if qmask is not None:
                return ((qmask > 0).astype(np.uint8) * 255)

    if query_masked_img is not None:
        return render_to_binary_mask(query_masked_img, nonblack_thresh=nonblack_thresh)

    return None


def clamp_tz_by_prior(t, tz_bbox, prior_range=(0.80, 1.20)):
    t = np.asarray(t, dtype=np.float64).copy()
    if tz_bbox is None:
        return t
    tz_min = max(0.05, float(prior_range[0]) * float(tz_bbox))
    tz_max = max(tz_min + 1e-6, float(prior_range[1]) * float(tz_bbox))
    t[2] = np.clip(float(t[2]), tz_min, tz_max)
    return t


def scale_translation_xy_with_tz(t_ref, tz_new):
    t_ref = np.asarray(t_ref, dtype=np.float64).reshape(3)
    tz_ref = max(1e-8, float(t_ref[2]))
    s = float(tz_new) / tz_ref

    t_new = t_ref.copy()
    t_new[0] *= s
    t_new[1] *= s
    t_new[2] = float(tz_new)
    return t_new


def score_render_mask_against_query(query_mask, render_mask):
    q_stats = binary_mask_bbox_stats(query_mask)
    r_stats = binary_mask_bbox_stats(render_mask)

    if q_stats is None or r_stats is None:
        return {
            "score": -1e9,
            "iou": 0.0,
            "center_px": None,
            "height_ratio": None,
            "area_ratio": None,
        }

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

    return {
        "score": float(score),
        "iou": float(iou),
        "center_px": float(center_px),
        "height_ratio": float(h_ratio),
        "area_ratio": float(area_ratio),
    }


def refine_translation_xyz_with_mask(
    *,
    R,
    t_init,
    K,
    width,
    height,
    query_mask_img,
    object_height_m,
    gs_python,
    gs_repo,
    gs_model_dir,
    gs_iter,
    intrinsics_path,
    bg_color="0,0,0",
    nonblack_thresh=8,
    prior_range=(0.80, 1.20),
    num_iters=4,
    alpha_xy=0.65,
    alpha_z=0.55,
):
    t_cur = np.asarray(t_init, dtype=np.float64).reshape(3).copy()

    info = {
        "status": "not_run",
        "tz_pnp": float(t_init[2]),
        "tz_bbox_prior": None,
        "num_iters": int(num_iters),
        "history": [],
        "t_refined": t_cur.tolist(),
        "best_score": None,
    }

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

    with tempfile.TemporaryDirectory(prefix="step45_xyz_refine_") as td:
        td = Path(td)

        for it in range(int(num_iters)):
            pose_json_path = td / f"iter_{it:02d}.json"
            render_png_path = td / f"iter_{it:02d}.png"

            save_camera_pose_json(
                pose_json_path,
                K=K,
                R=R,
                t=t_cur,
                width=width,
                height=height,
            )

            try:
                render_single_pose_gs(
                    gs_python=gs_python,
                    gs_repo=gs_repo,
                    gs_model_dir=gs_model_dir,
                    gs_iter=gs_iter,
                    intrinsics_path=intrinsics_path,
                    pose_json_path=pose_json_path,
                    output_png_path=render_png_path,
                    width=width,
                    height=height,
                    bg_color=bg_color,
                    gs_mode=gs_mode,
                    save_xyz=False,
                    xyz_output_path=None,
                )
                render_img = load_image(render_png_path)
                render_mask = render_to_binary_mask(render_img, nonblack_thresh=nonblack_thresh)
            except Exception as e:
                print(f"  [xyz refine] iter={it} render failed: {e}")
                info["status"] = "render_failed"
                return best_t, info

            r_stats = binary_mask_bbox_stats(render_mask)
            metrics = score_render_mask_against_query(query_mask_img, render_mask)

            hist = {
                "iter": int(it),
                "t_before_update": t_cur.tolist(),
                "metrics": metrics,
            }

            if metrics["score"] > best_score:
                best_score = metrics["score"]
                best_t = t_cur.copy()
                best_metrics = metrics

            if r_stats is None:
                hist["note"] = "render mask empty"
                info["history"].append(hist)
                break

            du = q_stats["cx"] - r_stats["cx"]
            dv = q_stats["cy"] - r_stats["cy"]

            dtx = alpha_xy * (du / fx) * float(t_cur[2])
            dty = alpha_xy * (dv / fy) * float(t_cur[2])

            t_new = t_cur.copy()
            t_new[0] += dtx
            t_new[1] += dty

            h_ratio = r_stats["h"] / max(q_stats["h"], 1e-8)
            z_scale = (1.0 - alpha_z) + alpha_z * h_ratio
            z_scale = float(np.clip(z_scale, 0.85, 1.15))

            t_new[2] *= z_scale
            t_new = clamp_tz_by_prior(t_new, tz_bbox, prior_range=prior_range)

            hist["du_px"] = float(du)
            hist["dv_px"] = float(dv)
            hist["dtx"] = float(dtx)
            hist["dty"] = float(dty)
            hist["h_ratio"] = float(h_ratio)
            hist["z_scale"] = float(z_scale)
            hist["t_after_update"] = t_new.tolist()

            info["history"].append(hist)
            t_cur = t_new

    info["status"] = "ok"
    info["t_refined"] = best_t.tolist()
    info["best_score"] = float(best_score) if best_metrics is not None else None
    info["best_metrics"] = best_metrics
    return best_t, info


def render_single_pose_gs(
    gs_python,
    gs_repo,
    gs_model_dir,
    gs_iter,
    intrinsics_path,
    pose_json_path,
    output_png_path,
    width,
    height,
    bg_color="0,0,0",
    gs_mode="2dgs",
    save_xyz=False,
    xyz_output_path=None,
):
    gs_python = str(gs_python)
    gs_repo = Path(gs_repo)
    gs_model_dir = str(gs_model_dir)

    script_path = gs_repo / "scripts" / "render_single_pose.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"render_single_pose.py not found: {script_path}\n"
            f"Please add this script under {gs_repo}/scripts/."
        )

    cmd = [
        gs_python,
        str(script_path),
        "--model_dir", gs_model_dir,
        "--iteration", str(gs_iter),
        "--intrinsics_path", str(intrinsics_path),
        "--pose_json", str(pose_json_path),
        "--output_png", str(output_png_path),
        "--width", str(width),
        "--height", str(height),
        "--bg_color", str(bg_color),
        "--gs_mode", str(gs_mode),
    ]

    if save_xyz:
        cmd.append("--save_xyz")
        if xyz_output_path is not None:
            cmd.extend(["--xyz_output", str(xyz_output_path)])

    print("  [GS render] command:")
    print("   " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def sample_valid_xyz_points_from_map(xyz_map, max_points=2000, seed=42):
    xyz_map = np.asarray(xyz_map, dtype=np.float64)
    valid = np.all(np.isfinite(xyz_map), axis=2) & (np.abs(xyz_map).sum(axis=2) > 1e-6)

    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    n = min(int(max_points), len(xs))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(xs), size=n, replace=False)

    ys = ys[pick]
    xs = xs[pick]

    pts3d_obj = xyz_map[ys, xs].astype(np.float64)
    pts2d_uv = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    return pts3d_obj, pts2d_uv


def full_to_query_masked_coords(pts2d_full, step1_crop_offset_xy):
    pts2d_full = np.asarray(pts2d_full, dtype=np.float64)
    off = np.asarray(step1_crop_offset_xy, dtype=np.float64).reshape(1, 2)
    return pts2d_full - off


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def run_step6_translation(args):
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    npz_path = out_dir / "loftr_best_match_data.npz"
    meta_path = out_dir / "loftr_best_match_meta.json"
    loftr_json_path = out_dir / "loftr_scores.json"

    if not npz_path.exists():
        raise FileNotFoundError(
            f"loftr_best_match_data.npz not found: {npz_path}\n"
            "step4 (dino_loftr)를 먼저 실행하세요."
        )

    match_data = np.load(str(npz_path))
    mkpts0_840 = match_data["mkpts0_inlier_840"].astype(np.float64)
    mkpts1_840 = match_data["mkpts1_inlier_840"].astype(np.float64)
    conf_inlier = match_data["conf_inlier"].astype(np.float64)

    meta = load_json(meta_path)
    loftr_json = load_json(loftr_json_path)
    best_render = loftr_json["best_render"]

    print(f"  Loaded {len(mkpts0_840)} inlier matches for '{best_render}'")

    if len(mkpts0_840) == 0:
        raise RuntimeError(
            f"No LoFTR inlier matches for '{best_render}'. "
            "step5(dino_loftr)의 LOFTR_CONF_THRESH를 낮추거나 "
            "LOFTR_RANSAC_THRESH를 높여서 step5를 재실행하세요."
        )

    resize_target = int(meta["loftr_resize_target"])

    q_crop_hw = tuple(meta["query_crop_hw"])
    q_bbox    = meta["query_nonblack_bbox_xyxy"]
    pts_q_crop = unmap_from_square_resize(mkpts0_840, q_crop_hw, resize_target)
    pts_q_full = to_full_image_coords(pts_q_crop, q_bbox)

    if len(pts_q_full) > 0:
        print(f"  pts_q_full range: x=[{pts_q_full[:,0].min():.0f},{pts_q_full[:,0].max():.0f}], y=[{pts_q_full[:,1].min():.0f},{pts_q_full[:,1].max():.0f}]")

    g_crop_hw = tuple(meta["gallery_crop_hw"])
    g_bbox = meta["gallery_nonblack_bbox_xyxy"]
    pts_g_crop = unmap_from_square_resize(mkpts1_840, g_crop_hw, resize_target)
    pts_g_full = to_full_image_coords(pts_g_crop, g_bbox)
    if len(pts_g_full) > 0:
        print(f"  pts_g_full range: x=[{pts_g_full[:,0].min():.0f},{pts_g_full[:,0].max():.0f}], y=[{pts_g_full[:,1].min():.0f},{pts_g_full[:,1].max():.0f}]")

    xyz_dir = Path(args.gallery_xyz_dir)
    xyz_map_path = find_xyz_map_path(xyz_dir, best_render)
    xyz_map = np.load(str(xyz_map_path)).astype(np.float64)

    print(f"  XYZ map: {xyz_map_path.name}  shape={xyz_map.shape}")

    canonical_ply_path = getattr(args, "canonical_ply_path", None)
    if canonical_ply_path is not None and Path(canonical_ply_path).exists():
        _ply_xyz = load_ply_xyz(Path(canonical_ply_path))
        _ply_norms = np.linalg.norm(_ply_xyz, axis=1)
        _ply_med_norm = float(np.median(_ply_norms))

        _xyz_valid_mask = np.abs(xyz_map).sum(axis=-1) > 1e-6
        if _xyz_valid_mask.any():
            _xyz_valid = xyz_map[_xyz_valid_mask]
            _xyz_norms = np.linalg.norm(_xyz_valid, axis=1)
            _p5, _p95 = np.percentile(_xyz_norms, [5, 95])
            _core_mask = (_xyz_norms >= _p5) & (_xyz_norms <= _p95)
            if _core_mask.sum() > 100:
                _xyz_med_norm = float(np.median(_xyz_norms[_core_mask]))
                _scale_correction = _ply_med_norm / _xyz_med_norm
                if 0.8 < _scale_correction < 1.25:
                    xyz_map = xyz_map * _scale_correction
                    print(f"  [XYZ scale fix] PLY median norm={_ply_med_norm:.6f}, "
                          f"XYZ median norm={_xyz_med_norm:.6f}, "
                          f"correction={_scale_correction:.6f}")
                else:
                    print(f"  [XYZ scale fix] correction={_scale_correction:.4f} out of range, skipped")
            else:
                print(f"  [XYZ scale fix] insufficient core points, skipped")
        else:
            print(f"  [XYZ scale fix] no valid XYZ map points")
    else:
        print(f"  [XYZ scale fix] canonical_ply_path not provided, skipped")

    pts3d, valid_mask = lookup_xyz_at_pixels(xyz_map, pts_g_full, bilinear=True)

    n_valid = int(valid_mask.sum())
    print(f"  Valid 2D-3D correspondences: {n_valid} / {len(pts_g_full)}")

    if n_valid < 4:
        raise RuntimeError(
            f"Not enough valid 2D-3D correspondences ({n_valid}). "
            "XYZ map이 비어있거나 gallery_xyz_dir 경로를 확인하세요."
        )

    pts2d_corr = pts_q_full[valid_mask]
    pts3d_corr = pts3d[valid_mask]
    pts_g_corr = pts_g_full[valid_mask]

    print(f"  Before query-mask filtering: {len(pts2d_corr)} correspondences")

    query_mask_path = getattr(args, "query_mask_path", None)
    n_inside_query_mask = None

    if query_mask_path:
        query_mask = cv2.imread(str(query_mask_path), cv2.IMREAD_GRAYSCALE)
        if query_mask is None:
            raise FileNotFoundError(f"Failed to load query mask: {query_mask_path}")

        inside_mask = filter_points_by_binary_mask(pts2d_corr, query_mask)
        n_inside_query_mask = int(inside_mask.sum())

        print(f"  Query-mask inside correspondences: {n_inside_query_mask} / {len(pts2d_corr)}")

        pts2d_corr = pts2d_corr[inside_mask]
        pts3d_corr = pts3d_corr[inside_mask]
        pts_g_corr = pts_g_corr[inside_mask]

        if len(pts2d_corr) < 4:
            raise RuntimeError(
                f"Not enough correspondences after query-mask filtering: {len(pts2d_corr)}"
            )
    else:
        print("  [WARN] --query_mask_path not provided, skipping query-mask filtering")

    pts2d_before_uniform = pts2d_corr.copy()
    pts3d_before_uniform = pts3d_corr.copy()

    conf_corr = conf_inlier[valid_mask]
    if query_mask_path:
        conf_corr = conf_corr[inside_mask]

    use_uniform_sampling = True
    uniform_keep_idx = None
    uniform_cell_stats = None

    if use_uniform_sampling:
        uniform_keep_idx, uniform_cell_stats = uniform_sample_points_2d(
            pts2d_corr,
            scores=conf_corr,
            grid_rows=10,
            grid_cols=10,
            max_per_cell=60,
        )

        print(f"  Uniform sampling: {len(uniform_keep_idx)} / {len(pts2d_corr)} kept")
        print(f"  Uniform cell stats: {uniform_cell_stats}")

        pts2d_corr = pts2d_corr[uniform_keep_idx]
        pts3d_corr = pts3d_corr[uniform_keep_idx]
        conf_corr  = conf_corr[uniform_keep_idx]
        pts_g_corr = pts_g_corr[uniform_keep_idx]

        if len(pts2d_corr) < 4:
            raise RuntimeError(
                f"Not enough correspondences after uniform sampling: {len(pts2d_corr)}"
            )
    else:
        print("  Uniform sampling disabled")

    _norms = np.linalg.norm(pts3d_corr, axis=1)
    _median_norm = np.median(_norms)
    _outlier_thresh = _median_norm * 5.0
    _outlier_mask = _norms < _outlier_thresh
    n_before_outlier = len(pts3d_corr)
    pts2d_corr = pts2d_corr[_outlier_mask]
    pts3d_corr = pts3d_corr[_outlier_mask]
    conf_corr  = conf_corr[_outlier_mask]
    pts_g_corr = pts_g_corr[_outlier_mask]
    print(f"  3D outlier filter (norm < {_outlier_thresh:.4f} = 5*median): "
          f"{len(pts3d_corr)} / {n_before_outlier} kept")

    if len(pts2d_corr) < 4:
        raise RuntimeError(
            f"Not enough correspondences after 3D outlier filtering: {len(pts2d_corr)}"
        )

    gallery = load_json(args.gallery_pose_json)
    render_idx = int(Path(best_render).stem)
    pose_record = None
    for pose in gallery["poses"]:
        if int(pose["index"]) == render_idx:
            pose_record = pose
            break
    if pose_record is None:
        raise KeyError(f"Pose for render '{best_render}' not found in gallery_poses.json")

    R_gallery = np.array(pose_record["R_obj_to_cam"], dtype=np.float64)
    t_gallery = np.array(pose_record["t_obj_to_cam"], dtype=np.float64)

    K = load_intrinsics(args.intrinsics_path)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    print(f"  K: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    reproj_thresh = float(getattr(args, "pnp_reproj_error", 200.0))

    print(f"  pts2d_corr are already in full-image coordinates: shape={pts2d_corr.shape}")

    query_img = load_image(args.query_img)
    axis_len = float(getattr(args, "axis_len_m", 0.04))
    canonical_ply_path = getattr(args, "canonical_ply_path", None)

    render_width = int(getattr(args, "render_width", query_img.shape[1]))
    render_height = int(getattr(args, "render_height", query_img.shape[0]))
    gs_model_dir = getattr(args, "gs_model_dir", None)
    gs_iter = getattr(args, "gs_iter", None)
    gs_repo = getattr(args, "gs_repo", None)
    gs_python = getattr(args, "gs_python", None) or sys.executable
    bg_color = getattr(args, "bg_color", "0,0,0")
    gs_mode = getattr(args, "gs_mode", "2dgs")

    query_masked_img = None
    if getattr(args, "query_masked_path", None):
        qmp = Path(args.query_masked_path)
        if qmp.exists():
            query_masked_img = load_image(qmp)
        else:
            print(f"  [WARN] query_masked_path not found: {qmp}")

    if canonical_ply_path is not None:
        canonical_ply_path = Path(canonical_ply_path)
        if canonical_ply_path.exists():
            ply_xyz = load_ply_xyz(canonical_ply_path)
        else:
            print(f"  [Pre-PnP PLY overlay] canonical_ply_path not found: {canonical_ply_path}")

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
            nonblack_thresh=8,
        )
        if _qmask_init is not None:
            tz_init = estimate_tz_from_mask_bbox(_qmask_init, fy=fy, object_height_m=object_height_m_init)
            q_stats_init = binary_mask_bbox_stats(_qmask_init)
            if tz_init is not None and q_stats_init is not None:
                tx_init = (q_stats_init["cx"] - cx) / fx * tz_init
                ty_init = (q_stats_init["cy"] - cy) / fy * tz_init
                t_out = np.array([tx_init, ty_init, tz_init], dtype=np.float64)
                print(f"  [no_pnp] bbox-based t_init: [{tx_init:.4f}, {ty_init:.4f}, {tz_init:.4f}]")
            else:
                t_out = t_gallery.copy()
                print(f"  [no_pnp] mask bbox estimation failed -> using gallery t")
        else:
            t_out = t_gallery.copy()
            print(f"  [no_pnp] no mask -> using gallery t")

        print(f"  [no_pnp] R = R_gallery, t = bbox_prior  (PnP skipped)")
    else:
        R_out, t_out, pose_method, inlier_count, reproj_err, inlier_idx = solve_pose_pnp(
            pts2d_corr,
            pts3d_corr,
            K,
            R_init=R_gallery,
            reproj_thresh=reproj_thresh,
            use_ransac=not bool(getattr(args, "no_pnp_ransac", False)),
        )

    _gallery_render_path = xyz_dir.parent / "gallery_renders_gs_ds" / best_render
    if not _gallery_render_path.exists():
        _gallery_render_path = xyz_dir.parent / "gallery_renders" / best_render
    _save_correspondence_debug(
        out_dir=out_dir,
        query_img=query_img,
        gallery_render_path=_gallery_render_path,
        pts2d_query=pts2d_corr,
        pts3d_obj=pts3d_corr,
        pts2d_gallery=pts_g_corr,
        R=R_out, t=t_out, K=K,
        inlier_idx=inlier_idx,
    )

    xyz_refine_info = {"status": "not_run"}
    skip_t_refine    = bool(getattr(args, "skip_t_refine", False))
    iou_accept_thresh = float(getattr(args, "t_refine_iou_thresh", 0.30))

    t_pnp = t_out.copy()

    if skip_t_refine:
        print("  [xyz refine] skipped (--skip_t_refine)")
        xyz_refine_info["status"] = "skipped_by_flag"

    elif gs_model_dir and gs_iter is not None and gs_repo and gs_python:
        query_ref_mask = make_query_reference_mask(
            query_masked_img=query_masked_img,
            query_mask_path=(out_dir / "query_mask.png"),
            nonblack_thresh=8,
        )

        object_height_m = float(getattr(args, "object_height_m", 0.125))

        if query_ref_mask is not None:
            tz_bbox = estimate_tz_from_mask_bbox(
                query_ref_mask,
                fy=K[1, 1],
                object_height_m=object_height_m,
            )

            if tz_bbox is not None:
                rel_err = abs(float(t_out[2]) - float(tz_bbox)) / max(float(tz_bbox), 1e-8)
                print(f"  [tz prior] bbox-based tz={tz_bbox:.4f} m, "
                      f"PnP tz={float(t_out[2]):.4f} m, rel_err={rel_err * 100.0:.1f}%")

            t_refined, xyz_refine_info = refine_translation_xyz_with_mask(
                R=R_out,
                t_init=t_out,
                K=K,
                width=render_width,
                height=render_height,
                query_mask_img=query_ref_mask,
                object_height_m=object_height_m,
                gs_python=gs_python,
                gs_repo=gs_repo,
                gs_model_dir=gs_model_dir,
                gs_iter=gs_iter,
                intrinsics_path=args.intrinsics_path,
                bg_color=bg_color,
                nonblack_thresh=8,
                prior_range=(0.80, 1.20),
                num_iters=4,
                alpha_xy=0.65,
                alpha_z=0.55,
            )

            best_score = xyz_refine_info.get("best_score") or -1.0
            best_iou   = (xyz_refine_info.get("best_metrics") or {}).get("iou", 0.0)
            t_ref      = np.asarray(xyz_refine_info["t_refined"], dtype=np.float64)

            print(f"  [xyz refine] status={xyz_refine_info['status']}, "
                  f"best_score={best_score:.4f}, best_iou={best_iou:.4f} "
                  f"(threshold={iou_accept_thresh:.2f})")
            print(f"  [xyz refine] t_pnp=[{t_pnp[0]:.4f}, {t_pnp[1]:.4f}, {t_pnp[2]:.4f}]")
            print(f"  [xyz refine] t_ref=[{t_ref[0]:.4f}, {t_ref[1]:.4f}, {t_ref[2]:.4f}]")

            if best_iou >= iou_accept_thresh:
                t_out = t_refined
                xyz_refine_info["t_applied"] = True
                print(f"  [xyz refine] IoU={best_iou:.4f} >= {iou_accept_thresh:.2f} -> t_refined applied")
            else:
                t_out = t_pnp
                xyz_refine_info["t_applied"] = False
                print(f"  [xyz refine] IoU={best_iou:.4f} < {iou_accept_thresh:.2f} -> PnP t kept")
        else:
            print("  [xyz refine] query reference mask unavailable, skipping.")
            xyz_refine_info["status"] = "skipped_no_query_mask"
    else:
        print("  [xyz refine] GS render args missing, skipping.")
        xyz_refine_info["status"] = "skipped_no_gs_args"

    q_out = rotation_matrix_to_quaternion(R_out)

    if len(inlier_idx) > 0:
        pts2d_inliers = pts2d_corr[inlier_idx]
        pts3d_inliers = pts3d_corr[inlier_idx]
    else:
        pts2d_inliers = np.empty((0, 2), dtype=np.float64)
        pts3d_inliers = np.empty((0, 3), dtype=np.float64)

    t_before_vis = estimate_t_linear(pts2d_corr, pts3d_corr, K, R_gallery)

    stats_after_all = compute_mean_reprojection_error(
        pts2d=pts2d_corr,
        pts3d=pts3d_corr,
        K=K,
        R=R_out,
        t=t_out,
    )
    print(f"  [All Corr] after  mean={stats_after_all['mean_px']:.2f}px, "
        f"median={stats_after_all['median_px']:.2f}px, max={stats_after_all['max_px']:.2f}px")

    if len(pts2d_inliers) > 0:
        stats_after_inliers = compute_mean_reprojection_error(
            pts2d=pts2d_inliers,
            pts3d=pts3d_inliers,
            K=K,
            R=R_out,
            t=t_out,
        )
        print(f"  [Inliers Only] after mean={stats_after_inliers['mean_px']:.2f}px, "
            f"median={stats_after_inliers['median_px']:.2f}px, max={stats_after_inliers['max_px']:.2f}px")
    else:
        stats_after_inliers = {
            "mean_px": None,
            "median_px": None,
            "max_px": None,
            "num_points": 0,
        }

    reproj_stats_after_all = compute_mean_reprojection_error(
        pts2d_corr, pts3d_corr, K, R_out, t_out
    )

    reproj_stats_after_inliers = compute_mean_reprojection_error(
        pts2d_inliers, pts3d_inliers, K, R_out, t_out
    )

    print(f"  [Reproj after PnP / all]    mean={reproj_stats_after_all['mean_px']:.2f}px, "
        f"median={reproj_stats_after_all['median_px']:.2f}px, "
        f"max={reproj_stats_after_all['max_px']:.2f}px, "
        f"N={reproj_stats_after_all['num_points']}")

    print(f"  [Reproj after PnP / inlier] mean={reproj_stats_after_inliers['mean_px']:.2f}px, "
        f"median={reproj_stats_after_inliers['median_px']:.2f}px, "
        f"max={reproj_stats_after_inliers['max_px']:.2f}px, "
        f"N={reproj_stats_after_inliers['num_points']}")

    reproj_debug_path = out_dir / "step6_reproj_debug_inliers.png"
    reproj_debug_stats_path = out_dir / "step6_reproj_debug_inliers_stats.json"

    if len(pts2d_inliers) > 0:
        N_in = min(len(pts2d_inliers), 120)
        rng_in = np.random.default_rng(42)
        dbg_idx_in = rng_in.choice(len(pts2d_inliers), size=N_in, replace=False)

        draw_correspondence_debug_single_pose(
            query_img=query_img,
            pts2d_query=pts2d_inliers,
            pts3d_obj=pts3d_inliers,
            K=K,
            R=R_out,
            t=t_out,
            out_path=reproj_debug_path,
            draw_idx=dbg_idx_in,
            title_prefix="PnP inliers only (after pose)",
            stats_json_path=reproj_debug_stats_path,
        )
    else:
        cv2.imwrite(str(reproj_debug_path), query_img.copy())
        save_json(reproj_debug_stats_path, {
            "title_prefix": "PnP inliers only (after pose)",
            "num_points_total": 0,
            "num_points_drawn": 0,
        })

    post_gs_render_path = None
    post_gs_overlay_path = None
    post_gs_xyz_path = None

    if gs_model_dir and gs_iter is not None and gs_repo and gs_python:
        post_pose_json = out_dir / "step6_pose_after_pnp.json"
        post_gs_render_path = out_dir / "step6_gs_render_after_pnp.png"
        post_gs_overlay_path = out_dir / "step6_gs_render_overlay_after_pnp.png"
        post_gs_xyz_path = out_dir / "step6_gs_render_after_pnp_xyz.npy"

        save_camera_pose_json(
            post_pose_json,
            K=K,
            R=R_out,
            t=t_out,
            width=render_width,
            height=render_height,
        )

        try:
            render_single_pose_gs(
                gs_python=gs_python,
                gs_repo=gs_repo,
                gs_model_dir=gs_model_dir,
                gs_iter=gs_iter,
                intrinsics_path=args.intrinsics_path,
                pose_json_path=post_pose_json,
                output_png_path=post_gs_render_path,
                width=render_width,
                height=render_height,
                bg_color=bg_color,
                gs_mode=gs_mode,
                save_xyz=True,
                xyz_output_path=post_gs_xyz_path,
            )
            post_render_img = load_image(post_gs_render_path)
            overlay_render_on_query(
                query_img=query_img,
                render_img=post_render_img,
                out_path=post_gs_overlay_path,
                alpha=0.45,
                nonblack_thresh=8,
            )
        except Exception as e:
            print(f"  [WARN] Failed to render GS after PnP: {e}")
            post_gs_render_path = None
            post_gs_overlay_path = None
            post_gs_xyz_path = None
    else:
        print("  [WARN] GS render args missing for after-PnP render overlay")

    bad_inlier_local_idx = np.array([], dtype=np.int32)
    bad_corr_global_idx = np.array([], dtype=np.int32)

    if len(pts2d_inliers) > 0 and post_gs_render_path is not None and Path(post_gs_render_path).exists():
        post_render_img = load_image(post_gs_render_path)

        bad_inlier_local_idx, proj_after_all_inliers = get_outside_after_inlier_indices(
            pts3d_inliers=pts3d_inliers,
            K=K,
            R_after=R_out,
            t_after=t_out,
            render_img_after=post_render_img,
            nonblack_thresh=8,
        )

        if len(bad_inlier_local_idx) > 0:
            bad_corr_global_idx = inlier_idx[bad_inlier_local_idx].astype(np.int32)

        print(f"  [Prune candidates] outside-after-render inliers: {len(bad_inlier_local_idx)}")

        save_json(out_dir / "step6_bad_after_render_inliers.json", {
            "bad_inlier_local_idx": bad_inlier_local_idx.tolist(),
            "bad_corr_global_idx": bad_corr_global_idx.tolist(),
            "num_bad_after_render": int(len(bad_inlier_local_idx)),
        })
    else:
        print("  [Prune candidates] skipped (no inliers or no post render)")

    if len(bad_corr_global_idx) > 0:
        keep_corr_mask = np.ones(len(pts2d_corr), dtype=bool)
        keep_corr_mask[bad_corr_global_idx] = False

        pts2d_corr_pruned = pts2d_corr[keep_corr_mask]
        pts3d_corr_pruned = pts3d_corr[keep_corr_mask]

        keep_inlier_mask = np.ones(len(pts2d_inliers), dtype=bool)
        keep_inlier_mask[bad_inlier_local_idx] = False

        pts2d_inliers_pruned = pts2d_inliers[keep_inlier_mask]
        pts3d_inliers_pruned = pts3d_inliers[keep_inlier_mask]
    else:
        pts2d_corr_pruned = pts2d_corr.copy()
        pts3d_corr_pruned = pts3d_corr.copy()
        pts2d_inliers_pruned = pts2d_inliers.copy()
        pts3d_inliers_pruned = pts3d_inliers.copy()

    try:
        if post_gs_xyz_path is not None and Path(post_gs_xyz_path).exists():
            post_xyz_map_samecorr = np.load(str(post_gs_xyz_path)).astype(np.float64)

            same_corr_trace = build_same_corr_motion_trace(
                pts2d_query_obs=pts2d_corr,
                pts2d_gallery_obs=pts_g_corr,
                pts3d_obj=pts3d_corr,
                K=K,
                R_before=R_gallery,
                t_before=t_gallery,
                R_after=R_out,
                t_after=t_out,
            )

            same_corr_surface_trace = augment_same_corr_trace_with_postrender_surface(
                trace=same_corr_trace,
                post_xyz_map=post_xyz_map_samecorr,
                xyz_err_bad_thresh_m=0.005,
            )

            save_json(out_dir / "step6_same_corr_postrender_surface_trace.json", same_corr_surface_trace)
            save_same_corr_surface_csv(out_dir / "step6_same_corr_postrender_surface_trace.csv", same_corr_surface_trace)

            ss = same_corr_surface_trace["summary"]
            print(
                f"  [Same corr post-render surface] "
                f"valid={ss['num_post_xyz_valid']}/{same_corr_surface_trace['num_points']}  "
                f"bad={ss['num_surface_bad']}  "
                f"mean_surface_err={None if ss['mean_surface_err_m'] is None else ss['mean_surface_err_m']*1000:.2f}mm"
            )
            print("  [Same corr post-render surface] saved: step6_same_corr_postrender_surface_trace.json/csv")
        else:
            print("  [Same corr post-render surface] skipped (post XYZ unavailable)")
    except Exception as e:
        print(f"  [Same corr post-render surface] failed: {e}")

    pose_out = {
        "stage": "step6",
        "best_render": best_render,
        "n_loftr_inliers_input": int(len(mkpts0_840)),
        "n_valid_2d3d_corr": n_valid,
        "pnp_reproj_error_px": reproj_err if np.isfinite(reproj_err) else None,
        "gallery_pose_source": args.gallery_pose_json,
        "xyz_map_used": str(xyz_map_path),
        "R_gallery_init": R_gallery.tolist(),
        "R_obj_to_cam": R_out.tolist(),
        "t_obj_to_cam": t_out.tolist(),
        "quat_wxyz": q_out.tolist(),
        "intrinsics_path": str(args.intrinsics_path),
        "query_mask_path": str(query_mask_path) if query_mask_path else None,
        "n_after_query_mask_filter": int(len(pts2d_corr)),
        "n_inside_query_mask": n_inside_query_mask,
        "debug_reproj_inliers": str(reproj_debug_path),
        "debug_gs_render_after_pnp": str(post_gs_render_path) if post_gs_render_path else None,
        "debug_gs_render_overlay_after_pnp": str(post_gs_overlay_path) if post_gs_overlay_path else None,
        "n_before_uniform_sampling": int(len(pts2d_before_uniform)),
        "n_after_uniform_sampling": int(len(pts2d_corr)),
        "uniform_sampling_cell_stats": uniform_cell_stats,
        "outputs": {},
    }

    pose_out["query_img"] = str(args.query_img)
    pose_out["query_masked_path"] = str(getattr(args, "query_masked_path", ""))
    pose_out["intrinsics_path"] = str(args.intrinsics_path)

    save_json(out_dir / "step6_pose.json", pose_out)
    save_json(out_dir / "initial_pose.json", pose_out)

    print("=" * 60)
    print("[Step 6] PnP translation estimation complete")
    print(f"  best_render     : {best_render}")
    print(f"  n_corr          : {n_valid}")
    print(f"  reproj_err      : {reproj_err:.2f}px" if np.isfinite(reproj_err) else "  reproj_err      : N/A")
    print(f"  R_obj_to_cam    :\n{R_out}")
    print(f"  t_obj_to_cam    : [{t_out[0]:.4f}, {t_out[1]:.4f}, {t_out[2]:.4f}]")
    print(f"  initial_pose.json: {out_dir / 'initial_pose.json'}")
    print("=" * 60)
