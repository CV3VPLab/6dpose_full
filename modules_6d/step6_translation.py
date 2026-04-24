"""
step45_translation.py
=====================
step4 (dino_loftr) 후에 LoFTR inlier 매칭점 + gallery XYZ map을 사용해서
정밀한 translation(T)을 추정하는 단계.

입력:
  - loftr_best_match_data.npz   (step5에서 저장)
  - loftr_best_match_meta.json  (step5에서 저장)
  - loftr_scores.json           (best_render 이름)
  - gallery_poses.json          (best_render에 해당하는 R)
  - gallery_xyz_gs/*.npy        (step3에서 저장한 XYZ map)
  - intrinsics.txt

출력:
  - step6_pose.json            → step5 또는 step6의 initial_pose.json으로 사용 가능

좌표 변환 체계:
  mkpts0 (840px query_crop)   → unmap → query_crop → + query_bbox[:2] → query_full_image (2D)
  mkpts1 (840px gallery_crop) → unmap → gallery_crop → + gallery_bbox[:2] → gallery_full_image
                              → XYZ map[v, u] → 3D canonical 좌표

  2D(query) ↔ 3D(canonical) → solvePnPRansac
  R는 gallery pose에서 초기값, T는 자유 추정
"""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import csv

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


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate unmapping
# ─────────────────────────────────────────────────────────────────────────────

def unmap_from_square_resize(pts_resized, orig_hw, resize_target=840):
    """
    square_pad_resize(img, resize_target) 공간의 픽셀 좌표를
    원본 이미지(padding 전) 좌표로 역변환.

    pts_resized : (N, 2) float, [u, v] in resize_target space
    orig_hw     : (h, w) of the image BEFORE square padding
    Returns     : (N, 2) float in original crop image space
    """
    h, w = orig_hw
    side = max(h, w)
    x0 = (side - w) // 2
    y0 = (side - h) // 2

    pts_square = np.asarray(pts_resized, dtype=np.float64) * (side / resize_target)
    pts_crop = pts_square - np.array([[x0, y0]], dtype=np.float64)
    return pts_crop


def to_full_image_coords(pts_crop, nonblack_bbox_xyxy):
    """
    tight_crop 후의 좌표를 full image 좌표로 변환.
    nonblack_bbox_xyxy: (x1, y1, x2, y2) — tight_crop_nonblack이 반환한 bbox
    """
    x1, y1 = float(nonblack_bbox_xyxy[0]), float(nonblack_bbox_xyxy[1])
    pts = np.asarray(pts_crop, dtype=np.float64) + np.array([[x1, y1]])
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# XYZ map lookup
# ─────────────────────────────────────────────────────────────────────────────

def find_xyz_map_path(xyz_dir: Path, render_filename: str):
    """
    gallery_xyz_gs 디렉토리에서 render 파일명에 대응하는 XYZ map 경로를 찾는다.
    시도 순서:
      {stem}.npy
      {stem}_xyz.npy
      {stem}_xyz_map.npy
    """
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
    """
    XYZ map에서 pts_uv 위치의 3D 좌표를 조회.

    xyz_map : (H, W, 3) float32/float64
    pts_uv  : (N, 2) float [u, v] — full gallery image 좌표
    bilinear: True면 쌍선형 보간 (소수점 좌표 정밀도↑)

    Returns:
      pts3d  : (N, 3) float — 유효하지 않은 위치는 NaN
      valid  : (N,) bool
    """
    H, W = xyz_map.shape[:2]
    N = len(pts_uv)
    pts3d = np.full((N, 3), np.nan, dtype=np.float64)

    if not bilinear:
        u = np.round(pts_uv[:, 0]).astype(int)
        v = np.round(pts_uv[:, 1]).astype(int)
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        pts3d[in_bounds] = xyz_map[v[in_bounds], u[in_bounds]].astype(np.float64)
    else:
        # 쌍선형 보간: 더 정밀하지만 느림
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

    # 유효 조건: in bounds AND not all-zero (background)
    finite = np.all(np.isfinite(pts3d), axis=1)
    nonzero = np.abs(pts3d).sum(axis=1) > 1e-6
    valid = finite & nonzero
    return pts3d, valid


# ─────────────────────────────────────────────────────────────────────────────
# Pose estimation
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
    """
    R을 완전히 고정하고 T만 linear least-squares로 추정.

    x_px = K @ (R @ X3d + t)
    → K_inv @ x_px = R @ X3d + t
    → t = mean(K_inv @ x_px - R @ X3d)   (최소제곱 해)

    빠르고 안정적이나, R 오차가 있으면 T도 영향받음.
    correspondence가 많을수록 평균이 안정적.
    """
    K_inv = np.linalg.inv(K)
    N = len(pts2d)
    t_estimates = []

    pts2d_h = np.hstack([pts2d, np.ones((N, 1))])   # (N, 3)
    rays = (K_inv @ pts2d_h.T).T                     # (N, 3) — 정규화 전
    # 각 ray의 depth는 모름 → 직접 t를 구해야 함

    # 대신: R @ X3d + t = depth_i * ray_i
    # depth를 몰라서 직접 못 품 → 아래 방법 사용
    # A @ t = b, A = I (N개 쌓기), b_i = ray_i * depth_i - R @ X3d_i
    # → depth를 lambda로 두면 underdetermined
    # → 실용적 근사: t ≈ mean(pixel_ray * tz - R @ X3d) 방향

    # 더 직접적인 방법: t_z를 grid search or outlier-robust mean
    # 여기서는 solvePnP와 비교용으로 단순 구현
    X3d_cam = (R @ pts3d.T).T   # (N, 3): R*X

    # homogeneous 2D → normalized image coord
    uvh = (K_inv @ np.column_stack([pts2d, np.ones(N)]).T).T  # (N, 3)

    # [u_n, v_n, 1] * tz = R@X3d + t → tz = (Rz*Xz + tz) / 1 ...
    # 가장 robust한 방법: z 방향 t를 먼저 구하고 x,y 유도
    # tz * 1 = (R@X)[z] + tz  → 이건 circular
    # 현실적으로: DLT-style
    # [u_n  0  -1  0  0  0 ] [tx]   [-(R@X)[x]]
    # [0  v_n   0 -1  0  0 ] [ty] = [-(R@X)[y]]
    #                         [tz]
    # 이거면 3D→2D reprojection constraint: u_n * tz - tx = -(R@X)[x] etc.
    A = np.zeros((2 * N, 3), dtype=np.float64)
    b = np.zeros(2 * N, dtype=np.float64)

    for i in range(N):
        un, vn = uvh[i, 0], uvh[i, 1]
        rx, ry, rz = X3d_cam[i]
        A[2*i,   0] = 1;  A[2*i,   2] = -un   # tx - un*tz = rx - un*rz... no
        A[2*i+1, 1] = 1;  A[2*i+1, 2] = -vn

    # 정리: u_n * (R@X + t)[z] = (R@X + t)[x]
    #        → u_n * (rz + tz) = rx + tx
    #        → tx - u_n*tz = -(rx - u_n*rz)
    for i in range(N):
        un, vn = uvh[i, 0], uvh[i, 1]
        rx, ry, rz = X3d_cam[i]
        A[2*i,   0] = 1;  A[2*i,   2] = -un
        b[2*i]   = -(rx - un * rz)
        A[2*i+1, 1] = 1;  A[2*i+1, 2] = -vn
        b[2*i+1] = -(ry - vn * rz)

    t_ls, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return t_ls.astype(np.float64)

# =========================
# [1] solve_pose_pnp() 전체 교체
# 기존:
# def solve_pose_pnp(pts2d, pts3d, K, R_init, t_init, ...)
# =========================


def _save_correspondence_debug(out_dir, query_img, gallery_render_path,
                                pts2d_query, pts3d_obj, pts2d_gallery,
                                R, t, K, inlier_idx=None):
    """
    PnP 후 correspondence 디버그 시각화 저장.
    inlier_idx: RANSAC inlier 인덱스 배열 (None이면 전체 inlier로 처리)

    저장 파일:
      step6_corr_query.png      : query + inliers (색상 원 + 재투영 빨강 X) + outliers (회색)
      step6_corr_gallery.png    : gallery render + inliers (색상) + outliers (회색)
      step6_corr_sidebyside.png : gallery|query 나란히 + inlier 대응선만 표시
    """
    try:
        import cv2 as _cv2
        import numpy as _np

        N = len(pts2d_query)
        if N == 0:
            return

        # ── inlier / outlier 마스크 ─────────────────────────────────────────
        if inlier_idx is None or len(inlier_idx) == 0:
            inlier_mask = _np.ones(N, dtype=bool)
        else:
            inlier_mask = _np.zeros(N, dtype=bool)
            inlier_mask[inlier_idx] = True

        n_inliers  = int(inlier_mask.sum())
        n_outliers = N - n_inliers

        # ── inlier에 대한 색상 팔레트 ───────────────────────────────────────
        inlier_indices = _np.where(inlier_mask)[0]
        inlier_colors = [
            tuple(int(c) for c in _cv2.applyColorMap(
                _np.array([[int(k * 180 / max(n_inliers - 1, 1))]], dtype=_np.uint8),
                _cv2.COLORMAP_HSV
            )[0, 0])
            for k in range(n_inliers)
        ]
        OUTLIER_COLOR = (100, 100, 100)   # 회색

        # ── 1. RANSAC inlier에 대해서만 reprojection 계산 ──────────────────
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

        # ── 2. Query 이미지 ─────────────────────────────────────────────────
        q_vis = query_img.copy()

        # outliers: 회색 작은 원
        for i in _np.where(~inlier_mask)[0]:
            pu, pv = int(round(pts2d_query[i, 0])), int(round(pts2d_query[i, 1]))
            _cv2.circle(q_vis, (pu, pv), 4, OUTLIER_COLOR, -1)

        # inliers: 색상 원 + 재투영 X + 오차 선
        for k, (i, c) in enumerate(zip(inlier_indices, inlier_colors)):
            pu, pv = int(round(pts2d_query[i, 0])), int(round(pts2d_query[i, 1]))
            ru, rv = int(round(pts2d_reproj_in[k, 0])), int(round(pts2d_reproj_in[k, 1]))
            _cv2.circle(q_vis, (pu, pv), 7, c, -1)
            _cv2.drawMarker(q_vis, (ru, rv), (0, 0, 255), _cv2.MARKER_CROSS, 14, 2)
            _cv2.line(q_vis, (pu, pv), (ru, rv), (0, 0, 255), 1)
            # 오차값 텍스트
            _cv2.putText(q_vis, f"{per_point_err[k]:.1f}",
                         (pu + 8, pv - 4), _cv2.FONT_HERSHEY_SIMPLEX,
                         0.4, (255, 255, 255), 1, _cv2.LINE_AA)

        # 범례 텍스트
        _cv2.putText(q_vis,
                     f"inliers={n_inliers}/{N}  reproj_inlier={mean_inlier_err:.2f}px",
                     (10, 30), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, _cv2.LINE_AA)
        _cv2.imwrite(str(out_dir / "step6_corr_query.png"), q_vis)

        # ── 3. Gallery 이미지 ───────────────────────────────────────────────
        gallery_img = _cv2.imread(str(gallery_render_path))
        if gallery_img is not None:
            g_vis = gallery_img.copy()

            # outliers: 회색
            for i in _np.where(~inlier_mask)[0]:
                gu, gv = int(round(pts2d_gallery[i, 0])), int(round(pts2d_gallery[i, 1]))
                _cv2.circle(g_vis, (gu, gv), 4, OUTLIER_COLOR, -1)

            # inliers: 색상
            for k, (i, c) in enumerate(zip(inlier_indices, inlier_colors)):
                gu, gv = int(round(pts2d_gallery[i, 0])), int(round(pts2d_gallery[i, 1]))
                _cv2.circle(g_vis, (gu, gv), 7, c, -1)

            _cv2.putText(g_vis,
                         f"inliers={n_inliers}/{N}",
                         (10, 30), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, _cv2.LINE_AA)
            _cv2.imwrite(str(out_dir / "step6_corr_gallery.png"), g_vis)

            # ── 4. Side-by-side (inlier 대응선만 표시) ───────────────────────
            qh, qw = q_vis.shape[:2]
            gh, gw = g_vis.shape[:2]
            scale = qh / gh
            g_resized = _cv2.resize(g_vis, (int(gw * scale), qh))
            offset = g_resized.shape[1] + 4
            canvas = _np.zeros((qh, qw + offset, 3), dtype=_np.uint8)
            canvas[:, :g_resized.shape[1]] = g_resized
            canvas[:, offset:] = q_vis

            # inlier 대응선
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
    """
    Clean step45 version:
      use_ransac=True  (default):
        Stage 1: EPnP + RANSAC → Stage 2: ITERATIVE refine on inliers
        Fallback: if EPnP fails, use linear T with fixed R_init
      use_ransac=False:
        전체 점으로 ITERATIVE solvePnP (R_init 초기값 사용)
        모든 점이 inlier로 처리됨 → inlier 개수 = N
    """
    N = len(pts2d)
    if N < 4:
        print(f"  [PnP] 2D-3D 대응점 {N}개 < 4, pose 추정 불가")
        t_linear = estimate_t_linear(pts2d, pts3d, K, R_init) if N > 0 else np.zeros(3, dtype=np.float64)
        return R_init, t_linear, "linear_T_fixed_R_insufficient_points", 0, np.inf, np.array([], dtype=np.int32)

    dist = np.zeros((4, 1), dtype=np.float64)

    if not use_ransac:
        # ── RANSAC 없이 전체 점으로 ITERATIVE PnP ──────────────────────────
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
        inlier_idx = np.arange(N, dtype=np.int32)  # 전체가 inlier

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

    # ── RANSAC 있는 기존 경로 ────────────────────────────────────────────────
    # Stage 1: EPnP + RANSAC
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

    # Stage 2: ITERATIVE refine on inliers
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

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

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
    cv2.line(img, o, tuple(imgpts[1]), (0, 0, 255), 4, cv2.LINE_AA)   # X red
    cv2.line(img, o, tuple(imgpts[2]), (0, 255, 0), 4, cv2.LINE_AA)   # Y green
    cv2.line(img, o, tuple(imgpts[3]), (255, 0, 0), 4, cv2.LINE_AA)   # Z blue
    cv2.circle(img, o, 6, (255, 255, 255), -1, cv2.LINE_AA)

    if out_path is not None:
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), img)
    return img


def load_ply_xyz(ply_path):
    """PLY 파일에서 xyz 좌표만 추출."""
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


def find_nearest_ply_vertices_for_xyz_points(pts3d_xyz, ply_xyz, chunk_size=50000):
    """
    각 xyz-map 3D 점에 대해 canonical PLY에서 nearest vertex를 찾음.
    chunked brute-force라 dependency 추가가 없음.
    """
    pts3d_xyz = np.asarray(pts3d_xyz, dtype=np.float64)
    ply_xyz = np.asarray(ply_xyz, dtype=np.float64)

    N = len(pts3d_xyz)
    if N == 0 or len(ply_xyz) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float64),
        )

    best_d2 = np.full((N,), np.inf, dtype=np.float64)
    best_idx = np.full((N,), -1, dtype=np.int32)
    best_xyz = np.zeros((N, 3), dtype=np.float64)

    for s in range(0, len(ply_xyz), int(chunk_size)):
        chunk = ply_xyz[s:s + int(chunk_size)]   # (M,3)
        diff = pts3d_xyz[:, None, :] - chunk[None, :, :]   # (N,M,3)
        d2 = np.sum(diff * diff, axis=2)                   # (N,M)

        local_idx = np.argmin(d2, axis=1)
        local_best_d2 = d2[np.arange(N), local_idx]

        update = local_best_d2 < best_d2
        if np.any(update):
            best_d2[update] = local_best_d2[update]
            best_idx[update] = s + local_idx[update]
            best_xyz[update] = chunk[local_idx[update]]

    return best_xyz, best_idx, np.sqrt(best_d2)


def evaluate_xyz_vs_ply_nn_consistency(
    pts2d_gallery,
    pts3d_xyz,
    ply_xyz,
    K,
    R_gallery,
    t_gallery,
    R_eval=None,
    t_eval=None,
    gallery_render_img=None,
    out_vis_path=None,
):
    """
    Test 1 + Test 2
    - Test 1: xyz-map point ↔ nearest PLY vertex distance
    - Test 2: 같은 pose에서 X_i와 V_i projection 비교
      * gallery pose 기준 reprojection
      * optional current pose 기준 projection delta
    """
    pts2d_gallery = np.asarray(pts2d_gallery, dtype=np.float64)
    pts3d_xyz = np.asarray(pts3d_xyz, dtype=np.float64)

    pts3d_ply_nn, nn_idx, nn_dist_m = find_nearest_ply_vertices_for_xyz_points(
        pts3d_xyz, ply_xyz, chunk_size=50000
    )

    proj_xyz_gallery = project_points_obj_to_img(pts3d_xyz, K, R_gallery, t_gallery)
    proj_ply_gallery = project_points_obj_to_img(pts3d_ply_nn, K, R_gallery, t_gallery)

    err_xyz_to_gallery_px = np.linalg.norm(proj_xyz_gallery - pts2d_gallery, axis=1)
    err_ply_to_gallery_px = np.linalg.norm(proj_ply_gallery - pts2d_gallery, axis=1)
    delta_proj_gallery_px = np.linalg.norm(proj_ply_gallery - proj_xyz_gallery, axis=1)

    result = {
        "pts3d_ply_nn": pts3d_ply_nn,
        "nearest_idx": nn_idx,
        "nn_dist_m": nn_dist_m,
        "proj_xyz_gallery": proj_xyz_gallery,
        "proj_ply_gallery": proj_ply_gallery,
        "err_xyz_to_gallery_px": err_xyz_to_gallery_px,
        "err_ply_to_gallery_px": err_ply_to_gallery_px,
        "delta_proj_gallery_px": delta_proj_gallery_px,
        "mean_nn_dist_m": float(np.mean(nn_dist_m)) if len(nn_dist_m) > 0 else None,
        "median_nn_dist_m": float(np.median(nn_dist_m)) if len(nn_dist_m) > 0 else None,
        "max_nn_dist_m": float(np.max(nn_dist_m)) if len(nn_dist_m) > 0 else None,
        "min_nn_dist_m": float(np.min(nn_dist_m)) if len(nn_dist_m) > 0 else None,
        "mean_delta_proj_gallery_px": float(np.mean(delta_proj_gallery_px)) if len(delta_proj_gallery_px) > 0 else None,
        "median_delta_proj_gallery_px": float(np.median(delta_proj_gallery_px)) if len(delta_proj_gallery_px) > 0 else None,
        "max_delta_proj_gallery_px": float(np.max(delta_proj_gallery_px)) if len(delta_proj_gallery_px) > 0 else None,
    }

    if R_eval is not None and t_eval is not None and len(pts3d_xyz) > 0:
        proj_xyz_eval = project_points_obj_to_img(pts3d_xyz, K, R_eval, t_eval)
        proj_ply_eval = project_points_obj_to_img(pts3d_ply_nn, K, R_eval, t_eval)
        delta_proj_eval_px = np.linalg.norm(proj_ply_eval - proj_xyz_eval, axis=1)

        result["proj_xyz_eval"] = proj_xyz_eval
        result["proj_ply_eval"] = proj_ply_eval
        result["delta_proj_eval_px"] = delta_proj_eval_px
        result["mean_delta_proj_eval_px"] = float(np.mean(delta_proj_eval_px))
        result["median_delta_proj_eval_px"] = float(np.median(delta_proj_eval_px))
        result["max_delta_proj_eval_px"] = float(np.max(delta_proj_eval_px))

    if gallery_render_img is not None and out_vis_path is not None:
        vis = gallery_render_img.copy()
        h, w = vis.shape[:2]

        for i in range(len(pts2d_gallery)):
            ug, vg = int(round(pts2d_gallery[i, 0])), int(round(pts2d_gallery[i, 1]))
            ux, vx = int(round(proj_xyz_gallery[i, 0])), int(round(proj_xyz_gallery[i, 1]))
            up, vp = int(round(proj_ply_gallery[i, 0])), int(round(proj_ply_gallery[i, 1]))

            if 0 <= ug < w and 0 <= vg < h:
                cv2.circle(vis, (ug, vg), 5, (0, 255, 0), -1, cv2.LINE_AA)  # green = original gallery pixel
            if 0 <= ux < w and 0 <= vx < h:
                cv2.drawMarker(vis, (ux, vx), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)  # yellow = xyz reproj
            if 0 <= up < w and 0 <= vp < h:
                cv2.drawMarker(vis, (up, vp), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)    # red = ply reproj

            if (0 <= ug < w and 0 <= vg < h and 0 <= up < w and 0 <= vp < h):
                cv2.line(vis, (ug, vg), (up, vp), (255, 255, 0), 1, cv2.LINE_AA)

            txt = f"{nn_dist_m[i]*1000:.1f}mm"
            if 0 <= ug < w and 0 <= vg < h:
                cv2.putText(vis, txt, (ug + 6, vg - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        header = (
            f"XYZ↔PLY-NN  mean_nn={result['mean_nn_dist_m']*1000:.2f}mm  "
            f"mean_dproj_gallery={result['mean_delta_proj_gallery_px']:.3f}px"
        )
        cv2.putText(vis, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

        ensure_dir(Path(out_vis_path).parent)
        cv2.imwrite(str(out_vis_path), vis)

    return result


def save_xyz_vs_ply_nn_json(path, result):
    save_json(path, {
        "num_points": int(len(result["nearest_idx"])),
        "nearest_ply_idx": result["nearest_idx"].tolist(),
        "nn_dist_mm_each": [float(v * 1000.0) for v in result["nn_dist_m"]],
        "mean_nn_dist_m": result["mean_nn_dist_m"],
        "median_nn_dist_m": result["median_nn_dist_m"],
        "max_nn_dist_m": result["max_nn_dist_m"],
        "min_nn_dist_m": result["min_nn_dist_m"],
        "err_xyz_to_gallery_px_each": [float(v) for v in result["err_xyz_to_gallery_px"]],
        "err_ply_to_gallery_px_each": [float(v) for v in result["err_ply_to_gallery_px"]],
        "delta_proj_gallery_px_each": [float(v) for v in result["delta_proj_gallery_px"]],
        "mean_delta_proj_gallery_px": result["mean_delta_proj_gallery_px"],
        "median_delta_proj_gallery_px": result["median_delta_proj_gallery_px"],
        "max_delta_proj_gallery_px": result["max_delta_proj_gallery_px"],
        "mean_delta_proj_eval_px": result.get("mean_delta_proj_eval_px", None),
        "median_delta_proj_eval_px": result.get("median_delta_proj_eval_px", None),
        "max_delta_proj_eval_px": result.get("max_delta_proj_eval_px", None),
    })

def compute_camera_coords_of_points(pts3d_obj, R, t):
    """
    canonical/object 3D points -> camera coordinates
    X_cam = R @ X_obj + t
    """
    pts3d_obj = np.asarray(pts3d_obj, dtype=np.float64)
    if len(pts3d_obj) == 0:
        return np.empty((0, 3), dtype=np.float64)
    return (R @ pts3d_obj.T).T + np.asarray(t, dtype=np.float64).reshape(1, 3)


def sample_valid_xyz_points_from_map(xyz_map, max_points=2000, seed=42):
    """
    xyz_map의 valid(non-zero finite) 점들 중 일부를 샘플링해서 반환.
    Returns:
      pts3d_obj : (N,3)
      pts2d_uv  : (N,2)  full image pixel coords [u,v]
    """
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


def save_sparse_correspondence_motion_debug(
    query_img,
    pts2d_query_obs,
    pts2d_gallery_obs,
    pts3d_obj,
    K,
    R_before,
    t_before,
    R_after,
    t_after,
    out_png,
    out_json,
    max_draw=200,
):
    """
    sparse matched 3D points(X_i)만 가지고
    - top1/gallery pose에서 어디에 있었는지
    - PnP pose에서 어디로 갔는지
    를 저장/시각화
    """
    pts2d_query_obs = np.asarray(pts2d_query_obs, dtype=np.float64)
    pts2d_gallery_obs = np.asarray(pts2d_gallery_obs, dtype=np.float64)
    pts3d_obj = np.asarray(pts3d_obj, dtype=np.float64)

    proj_before = project_points_obj_to_img(pts3d_obj, K, R_before, t_before)
    proj_after = project_points_obj_to_img(pts3d_obj, K, R_after, t_after)

    cam_before = compute_camera_coords_of_points(pts3d_obj, R_before, t_before)
    cam_after = compute_camera_coords_of_points(pts3d_obj, R_after, t_after)

    delta_uv = proj_after - proj_before
    delta_cam = cam_after - cam_before

    save_json(out_json, {
        "num_points": int(len(pts3d_obj)),
        "pts2d_query_obs": pts2d_query_obs.tolist(),
        "pts2d_gallery_obs": pts2d_gallery_obs.tolist(),
        "pts3d_obj": pts3d_obj.tolist(),
        "proj_before": proj_before.tolist(),
        "proj_after": proj_after.tolist(),
        "delta_uv": delta_uv.tolist(),
        "cam_before": cam_before.tolist(),
        "cam_after": cam_after.tolist(),
        "delta_cam": delta_cam.tolist(),
        "mean_delta_uv_px": float(np.mean(np.linalg.norm(delta_uv, axis=1))) if len(delta_uv) > 0 else None,
        "max_delta_uv_px": float(np.max(np.linalg.norm(delta_uv, axis=1))) if len(delta_uv) > 0 else None,
        "mean_delta_cam_m": float(np.mean(np.linalg.norm(delta_cam, axis=1))) if len(delta_cam) > 0 else None,
        "max_delta_cam_m": float(np.max(np.linalg.norm(delta_cam, axis=1))) if len(delta_cam) > 0 else None,
    })

    img = query_img.copy()
    h, w = img.shape[:2]

    if len(pts3d_obj) > max_draw:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts3d_obj), size=max_draw, replace=False)
    else:
        idx = np.arange(len(pts3d_obj), dtype=np.int32)

    # white = observed query, blue = before(top1/gallery pose), green = after(PnP pose)
    for i in idx:
        q = tuple(np.round(pts2d_query_obs[i]).astype(int))
        b = tuple(np.round(proj_before[i]).astype(int))
        a = tuple(np.round(proj_after[i]).astype(int))

        in_q = 0 <= q[0] < w and 0 <= q[1] < h
        in_b = 0 <= b[0] < w and 0 <= b[1] < h
        in_a = 0 <= a[0] < w and 0 <= a[1] < h

        if in_q:
            cv2.circle(img, q, 3, (255, 255, 255), -1, cv2.LINE_AA)
        if in_b:
            cv2.circle(img, b, 3, (255, 100, 0), -1, cv2.LINE_AA)
        if in_a:
            cv2.circle(img, a, 3, (0, 255, 0), -1, cv2.LINE_AA)

        if in_b and in_a:
            cv2.line(img, b, a, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, "white=query obs, blue=top1 pose, green=PnP pose", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    ensure_dir(Path(out_png).parent)
    cv2.imwrite(str(out_png), img)


def save_dense_xyzmap_motion_debug(
    query_img,
    xyz_map,
    K,
    R_before,
    t_before,
    R_after,
    t_after,
    out_png,
    out_json,
    max_points=200,
):
    """
    top1의 xyz_map 전체(valid points sampled)에 대해
    - gallery pose에서의 projection
    - PnP pose에서의 projection
    을 비교
    """
    pts3d_obj, pts2d_uv = sample_valid_xyz_points_from_map(
        xyz_map,
        max_points=max_points,
        seed=42,
    )

    if len(pts3d_obj) == 0:
        save_json(out_json, {"num_points": 0})
        cv2.imwrite(str(out_png), query_img.copy())
        return

    proj_before = project_points_obj_to_img(pts3d_obj, K, R_before, t_before)
    proj_after = project_points_obj_to_img(pts3d_obj, K, R_after, t_after)

    cam_before = compute_camera_coords_of_points(pts3d_obj, R_before, t_before)
    cam_after = compute_camera_coords_of_points(pts3d_obj, R_after, t_after)

    delta_uv = proj_after - proj_before
    delta_cam = cam_after - cam_before

    save_json(out_json, {
        "num_points": int(len(pts3d_obj)),
        "sampled_uv_from_xyz_map": pts2d_uv.tolist(),
        "proj_before": proj_before.tolist(),
        "proj_after": proj_after.tolist(),
        "delta_uv": delta_uv.tolist(),
        "cam_before": cam_before.tolist(),
        "cam_after": cam_after.tolist(),
        "delta_cam": delta_cam.tolist(),
        "mean_delta_uv_px": float(np.mean(np.linalg.norm(delta_uv, axis=1))),
        "median_delta_uv_px": float(np.median(np.linalg.norm(delta_uv, axis=1))),
        "max_delta_uv_px": float(np.max(np.linalg.norm(delta_uv, axis=1))),
        "mean_delta_cam_m": float(np.mean(np.linalg.norm(delta_cam, axis=1))),
        "median_delta_cam_m": float(np.median(np.linalg.norm(delta_cam, axis=1))),
        "max_delta_cam_m": float(np.max(np.linalg.norm(delta_cam, axis=1))),
    })

    img = query_img.copy()
    h, w = img.shape[:2]

    for i in range(len(pts3d_obj)):
        b = tuple(np.round(proj_before[i]).astype(int))
        a = tuple(np.round(proj_after[i]).astype(int))

        in_b = 0 <= b[0] < w and 0 <= b[1] < h
        in_a = 0 <= a[0] < w and 0 <= a[1] < h

        if in_b:
            cv2.circle(img, b, 1, (255, 100, 0), -1, cv2.LINE_AA)   # blue-ish
        if in_a:
            cv2.circle(img, a, 1, (0, 255, 0), -1, cv2.LINE_AA)     # green
        if in_b and in_a:
            cv2.line(img, b, a, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, "dense xyz-map sampled motion: blue=top1 pose, green=PnP pose", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    ensure_dir(Path(out_png).parent)
    cv2.imwrite(str(out_png), img)

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
    """
    same correspondence points only:
    각 point_id마다 top1/gallery pose -> PnP pose 변화 추적
    """
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
    improvement_px = err_before_px - err_after_px   # +면 PnP가 query에 더 가까워짐
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


def draw_same_corr_ids_on_gallery(
    gallery_img,
    trace,
    out_path,
    font_scale=0.35,
    circle_radius=4,
):
    """
    gallery image 위에 same correspondence point_id를 전부 표시
    """
    img = gallery_img.copy()
    h, w = img.shape[:2]

    for row in trace["rows"]:
        pid = row["point_id"]
        g = tuple(np.round(row["gallery_uv"]).astype(int))
        if 0 <= g[0] < w and 0 <= g[1] < h:
            cv2.circle(img, g, circle_radius, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(
                img, str(pid), (g[0] + 4, g[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 1, cv2.LINE_AA
            )

    cv2.putText(img, "same correspondence point IDs on top1 gallery", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), img)

def draw_same_corr_topk_gallery(
    gallery_img,
    trace,
    out_path,
    select_key="gallery_to_pnp_motion_px",   # or "gallery_reproj_err_px"
    topk=12,
    title="",
):
    """
    gallery 이미지에서 같은 point_id가
    observed gallery -> before(top1 reproj) -> after(PnP reproj)
    로 어떻게 보이는지 top-k만 깔끔하게 그림

    white = observed gallery_uv
    blue  = before_uv   (xyz_obj를 top1 pose로 reprojection)
    green = after_uv    (xyz_obj를 PnP pose로 reprojection)
    yellow arrow = before -> after
    """
    rows = trace["rows"]
    img = gallery_img.copy()
    h, w = img.shape[:2]

    if len(rows) == 0:
        cv2.putText(img, "No same-corr points", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), img)
        return

    vals = np.array([float(r[select_key]) for r in rows], dtype=np.float64)
    order = np.argsort(-vals)   # descending
    chosen = order[:min(int(topk), len(order))]

    for idx in chosen:
        r = rows[int(idx)]
        pid = r["point_id"]

        g = tuple(np.round(r["gallery_uv"]).astype(int))
        b = tuple(np.round(r["before_uv"]).astype(int))
        a = tuple(np.round(r["after_uv"]).astype(int))

        in_g = 0 <= g[0] < w and 0 <= g[1] < h
        in_b = 0 <= b[0] < w and 0 <= b[1] < h
        in_a = 0 <= a[0] < w and 0 <= a[1] < h

        if in_g:
            cv2.circle(img, g, 5, (255, 255, 255), -1, cv2.LINE_AA)   # white
        if in_b:
            cv2.circle(img, b, 5, (255, 100, 0), -1, cv2.LINE_AA)     # blue-ish
        if in_a:
            cv2.circle(img, a, 5, (0, 255, 0), -1, cv2.LINE_AA)       # green

        if in_b and in_a:
            cv2.arrowedLine(img, b, a, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.2)

        anchor = a if in_a else (b if in_b else g)
        if 0 <= anchor[0] < w and 0 <= anchor[1] < h:
            txt = f"id={pid} {select_key}={r[select_key]:.1f}"
            cv2.putText(
                img, txt, (anchor[0] + 6, anchor[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA
            )

    cv2.putText(img, f"{title}  top{len(chosen)} by {select_key}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "white=gallery obs, blue=top1 reproj, green=PnP reproj, yellow=before->after", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), img)

def draw_same_corr_topk_query(
    query_img,
    trace,
    out_path,
    select_key="delta_uv_px",   # or "improvement_px"
    topk=12,
    title="",
):
    """
    query 이미지에서 같은 point_id가
    top1 pose -> PnP pose 로 어떻게 변했는지 top-k만 깔끔하게 그림
    white=query obs, blue=before, green=after, yellow=before->after
    """
    rows = trace["rows"]
    img = query_img.copy()
    h, w = img.shape[:2]

    if len(rows) == 0:
        cv2.putText(img, "No same-corr points", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), img)
        return

    vals = np.array([float(r[select_key]) for r in rows], dtype=np.float64)
    order = np.argsort(-vals)   # descending
    chosen = order[:min(int(topk), len(order))]

    for idx in chosen:
        r = rows[int(idx)]
        pid = r["point_id"]

        q = tuple(np.round(r["query_uv"]).astype(int))
        b = tuple(np.round(r["before_uv"]).astype(int))
        a = tuple(np.round(r["after_uv"]).astype(int))

        in_q = 0 <= q[0] < w and 0 <= q[1] < h
        in_b = 0 <= b[0] < w and 0 <= b[1] < h
        in_a = 0 <= a[0] < w and 0 <= a[1] < h

        if in_q:
            cv2.circle(img, q, 5, (255, 255, 255), -1, cv2.LINE_AA)   # white
        if in_b:
            cv2.circle(img, b, 5, (255, 100, 0), -1, cv2.LINE_AA)     # blue-ish
        if in_a:
            cv2.circle(img, a, 5, (0, 255, 0), -1, cv2.LINE_AA)       # green

        if in_b and in_a:
            cv2.arrowedLine(img, b, a, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.2)

        anchor = a if in_a else (b if in_b else q)
        if 0 <= anchor[0] < w and 0 <= anchor[1] < h:
            txt = f"id={pid} {select_key}={r[select_key]:.1f}"
            cv2.putText(
                img, txt, (anchor[0] + 6, anchor[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA
            )

    cv2.putText(img, f"{title}  top{len(chosen)} by {select_key}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "white=query, blue=top1, green=PnP, yellow=before->after", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), img)


def draw_same_corr_topk_query_with_3d_panel(
    query_img,
    trace,
    out_path,
    select_key="improvement_px",   # or "delta_uv_px"
    topk=6,
    title="",
):
    """
    같은 correspondence point에 대해
    query 위의 before/after 2D 변화 + 우측 패널에 3D 정보까지 같이 표시

    left:
      white = query_uv
      blue  = before_uv
      green = after_uv
      yellow = before -> after

    right panel:
      point_id별로
      xyz_obj
      before_cam
      after_cam
      delta_cam
      before_uv / after_uv
      err_before / err_after / improvement
      를 텍스트로 표시
    """
    rows = trace["rows"]
    if len(rows) == 0:
        img = query_img.copy()
        cv2.putText(img, "No same-corr points", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), img)
        return

    vals = np.array([float(r[select_key]) for r in rows], dtype=np.float64)
    order = np.argsort(-vals)   # descending
    chosen = order[:min(int(topk), len(order))]

    left = query_img.copy()
    h, w = left.shape[:2]

    # 우측 정보 패널
    panel_w = 1100
    panel = np.full((h, panel_w, 3), 245, dtype=np.uint8)

    # 좌측 query 위 점/화살표 표시
    for idx in chosen:
        r = rows[int(idx)]
        pid = r["point_id"]

        q = tuple(np.round(r["query_uv"]).astype(int))
        b = tuple(np.round(r["before_uv"]).astype(int))
        a = tuple(np.round(r["after_uv"]).astype(int))

        in_q = 0 <= q[0] < w and 0 <= q[1] < h
        in_b = 0 <= b[0] < w and 0 <= b[1] < h
        in_a = 0 <= a[0] < w and 0 <= a[1] < h

        if in_q:
            cv2.circle(left, q, 5, (255, 255, 255), -1, cv2.LINE_AA)   # white
        if in_b:
            cv2.circle(left, b, 5, (255, 100, 0), -1, cv2.LINE_AA)     # blue-ish
        if in_a:
            cv2.circle(left, a, 5, (0, 255, 0), -1, cv2.LINE_AA)       # green

        if in_b and in_a:
            cv2.arrowedLine(left, b, a, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.2)

        anchor = a if in_a else (b if in_b else q)
        if 0 <= anchor[0] < w and 0 <= anchor[1] < h:
            cv2.putText(
                left, f"id={pid}", (anchor[0] + 6, anchor[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA
            )

    cv2.putText(left, f"{title}  top{len(chosen)} by {select_key}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(left, "white=query, blue=before, green=after, yellow=before->after", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    # 우측 텍스트 패널
    cv2.putText(panel, "Same-corr 2D/3D paired trace", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(panel, "obj = canonical/object-frame 3D", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(panel, "before_cam = xyz_obj moved by top1/gallery pose", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(panel, "after_cam  = xyz_obj moved by PnP pose", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

    y = 160
    block_h = 120

    for rank, idx in enumerate(chosen):
        r = rows[int(idx)]
        pid = r["point_id"]

        qx, qy = r["query_uv"]
        gu, gv = r["gallery_uv"]
        ox, oy, oz = r["xyz_obj"]
        bu, bv = r["before_uv"]
        au, av = r["after_uv"]

        bcx, bcy, bcz = r["cam_before"]
        acx, acy, acz = r["cam_after"]
        dcx, dcy, dcz = r["delta_cam"]

        err_b = r["err_before_px"]
        err_a = r["err_after_px"]
        impr  = r["improvement_px"]
        dcam  = r["delta_cam_m"]
        duv   = r["delta_uv_px"]

        y0 = y + rank * block_h
        if y0 + 100 > h:
            break

        # block separator
        cv2.line(panel, (15, y0 - 15), (panel_w - 15, y0 - 15), (200, 200, 200), 1, cv2.LINE_AA)

        lines = [
            f"id={pid}   {select_key}={r[select_key]:.2f}   delta_uv_px={duv:.2f}",
            f"query=({qx:.1f},{qy:.1f})   gallery=({gu:.1f},{gv:.1f})",
            f"obj=({ox:.4f}, {oy:.4f}, {oz:.4f})",
            f"before_cam=({bcx:.4f}, {bcy:.4f}, {bcz:.4f})",
            f"after_cam =({acx:.4f}, {acy:.4f}, {acz:.4f})",
            f"delta_cam =({dcx:.4f}, {dcy:.4f}, {dcz:.4f})   |d|={dcam:.4f}m",
            f"before_uv=({bu:.1f},{bv:.1f})   after_uv=({au:.1f},{av:.1f})",
            f"err_before={err_b:.2f}px   err_after={err_a:.2f}px   improvement={impr:.2f}px",
        ]

        for li, txt in enumerate(lines):
            cv2.putText(
                panel, txt, (20, y0 + li * 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (30, 30, 30), 1, cv2.LINE_AA
            )

    canvas = np.concatenate([left, panel], axis=1)
    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), canvas)

def augment_same_corr_trace_with_postrender_surface(
    trace,
    post_xyz_map,
    xyz_err_bad_thresh_m=0.005,
):
    """
    각 same-corr point_id에 대해
    after_uv 위치에서 post-render XYZ map lookup을 수행하고,
    xyz_obj와의 3D 차이를 저장한다.
    """
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


def draw_same_corr_topk_surface_err_postrender(
    post_render_img,
    trace_aug,
    out_path,
    topk=12,
):
    """
    post-render 이미지 위에서
    worst surface error point_id들을 표시.
    left: image overlay
    right: 2D/3D paired text panel
    """
    rows = trace_aug["rows"]
    if len(rows) == 0:
        img = post_render_img.copy()
        cv2.putText(img, "No same-corr points", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), img)
        return

    # valid 점 우선, 그중 surface_err_mm 큰 순
    sortable = []
    for i, r in enumerate(rows):
        val = -1.0 if r["surface_err_mm"] is None else float(r["surface_err_mm"])
        sortable.append((val, i))
    sortable.sort(reverse=True, key=lambda x: x[0])

    chosen_idx = [i for _, i in sortable[:min(int(topk), len(sortable))]]

    left = post_render_img.copy()
    h, w = left.shape[:2]

    panel_w = 1150
    panel = np.full((h, panel_w, 3), 245, dtype=np.uint8)

    for idx in chosen_idx:
        r = rows[idx]
        pid = r["point_id"]
        a = tuple(np.round(r["after_uv"]).astype(int))

        in_a = 0 <= a[0] < w and 0 <= a[1] < h

        if in_a:
            if r["post_xyz_valid"] and not r["surface_bad"]:
                cv2.circle(left, a, 6, (0, 255, 0), -1, cv2.LINE_AA)
            else:
                cv2.drawMarker(left, a, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)

            txt = f"id={pid} err={r['surface_err_mm']:.1f}mm" if r["surface_err_mm"] is not None else f"id={pid} invalid"
            cv2.putText(left, txt, (a[0] + 6, a[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(left, "Same-corr post-render surface error", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(left, "green=good, red=bad/invalid at after_uv", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(panel, "Same-corr post-render XYZ lookup", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(panel, "compare xyz_obj vs post_xyz_lookup at after_uv", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

    y = 110
    block_h = 112

    for rank, idx in enumerate(chosen_idx):
        r = rows[idx]
        y0 = y + rank * block_h
        if y0 + 95 > h:
            break

        cv2.line(panel, (15, y0 - 15), (panel_w - 15, y0 - 15), (200, 200, 200), 1, cv2.LINE_AA)

        ox, oy, oz = r["xyz_obj"]
        px, py, pz = r["post_xyz_lookup"]
        ex, ey, ez = r["surface_err_vec"]

        lines = [
            f"id={r['point_id']}   surface_err_mm={r['surface_err_mm'] if r['surface_err_mm'] is not None else 'invalid'}   valid={r['post_xyz_valid']}",
            f"after_uv=({r['after_uv'][0]:.1f}, {r['after_uv'][1]:.1f})",
            f"obj      =({ox:.4f}, {oy:.4f}, {oz:.4f})",
            f"post_xyz =({px:.4f}, {py:.4f}, {pz:.4f})",
            f"err_vec  =({ex:.4f}, {ey:.4f}, {ez:.4f})",
            f"err_after={r['err_after_px']:.2f}px   improvement={r['improvement_px']:.2f}px",
        ]

        for li, txt in enumerate(lines):
            cv2.putText(panel, txt, (20, y0 + li * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)

    canvas = np.concatenate([left, panel], axis=1)
    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), canvas)

def _is_valid_xyz_value(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.all(np.isfinite(xyz)) and (np.abs(xyz).sum() > 1e-6)


def _lookup_best_xyz_in_window(post_xyz_map, uv, xyz_ref, radius=4):
    """
    uv 주변 (2r+1)x(2r+1) window에서
    xyz_ref와 가장 가까운 post-render XYZ를 찾는다.

    Returns:
      {
        "center_uv": [u,v],
        "center_xyz": [x,y,z],
        "center_valid": bool,
        "center_err_m": float or None,
        "best_uv": [u,v] or None,
        "best_xyz": [x,y,z] or None,
        "best_valid": bool,
        "best_err_m": float or None,
        "best_offset_uv": [du,dv] or None,
        "best_shift_px": float or None,
      }
    """
    h, w = post_xyz_map.shape[:2]
    u0 = int(round(float(uv[0])))
    v0 = int(round(float(uv[1])))

    out = {
        "center_uv": [float(u0), float(v0)],
        "center_xyz": [np.nan, np.nan, np.nan],
        "center_valid": False,
        "center_err_m": None,
        "best_uv": None,
        "best_xyz": None,
        "best_valid": False,
        "best_err_m": None,
        "best_offset_uv": None,
        "best_shift_px": None,
    }

    if 0 <= u0 < w and 0 <= v0 < h:
        center_xyz = post_xyz_map[v0, u0].astype(np.float64)
        out["center_xyz"] = [float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])]
        if _is_valid_xyz_value(center_xyz):
            out["center_valid"] = True
            out["center_err_m"] = float(np.linalg.norm(center_xyz - xyz_ref))

    best_err = np.inf
    best_xyz = None
    best_uv = None

    u_min = max(0, u0 - int(radius))
    u_max = min(w - 1, u0 + int(radius))
    v_min = max(0, v0 - int(radius))
    v_max = min(h - 1, v0 + int(radius))

    for vv in range(v_min, v_max + 1):
        for uu in range(u_min, u_max + 1):
            xyz = post_xyz_map[vv, uu].astype(np.float64)
            if not _is_valid_xyz_value(xyz):
                continue
            err = float(np.linalg.norm(xyz - xyz_ref))
            if err < best_err:
                best_err = err
                best_xyz = xyz
                best_uv = (uu, vv)

    if best_xyz is not None:
        out["best_valid"] = True
        out["best_xyz"] = [float(best_xyz[0]), float(best_xyz[1]), float(best_xyz[2])]
        out["best_uv"] = [float(best_uv[0]), float(best_uv[1])]
        out["best_err_m"] = float(best_err)
        du = float(best_uv[0] - u0)
        dv = float(best_uv[1] - v0)
        out["best_offset_uv"] = [du, dv]
        out["best_shift_px"] = float(np.sqrt(du * du + dv * dv))

    return out


def augment_same_corr_trace_with_postrender_localnn(
    trace_aug_center,
    post_xyz_map,
    radius=4,
    best_err_bad_thresh_m=0.005,
):
    """
    이미 center lookup 결과가 들어간 same-corr trace에 대해,
    after_uv 주변 neighborhood best-match 결과를 추가.
    """
    rows = trace_aug_center["rows"]
    out_rows = []

    for r in rows:
        xyz_ref = np.asarray(r["xyz_obj"], dtype=np.float64)
        uv = np.asarray(r["after_uv"], dtype=np.float64)

        nn = _lookup_best_xyz_in_window(
            post_xyz_map=post_xyz_map,
            uv=uv,
            xyz_ref=xyz_ref,
            radius=radius,
        )

        rr = dict(r)
        rr["localnn_radius"] = int(radius)

        rr["center_err_mm"] = None if r["surface_err_m"] is None else float(r["surface_err_m"] * 1000.0)
        rr["best_xyz_lookup"] = nn["best_xyz"]
        rr["best_uv"] = nn["best_uv"]
        rr["best_err_m"] = nn["best_err_m"]
        rr["best_err_mm"] = None if nn["best_err_m"] is None else float(nn["best_err_m"] * 1000.0)
        rr["best_valid"] = bool(nn["best_valid"])
        rr["best_offset_uv"] = nn["best_offset_uv"]
        rr["best_shift_px"] = nn["best_shift_px"]
        rr["best_surface_bad"] = bool(nn["best_valid"] and nn["best_err_m"] > float(best_err_bad_thresh_m))

        if r["surface_err_m"] is not None and nn["best_err_m"] is not None:
            rr["center_to_best_improvement_mm"] = float((r["surface_err_m"] - nn["best_err_m"]) * 1000.0)
        else:
            rr["center_to_best_improvement_mm"] = None

        out_rows.append(rr)

    out = {
        "num_points": int(len(out_rows)),
        "summary": dict(trace_aug_center.get("summary", {})),
        "rows": out_rows,
    }

    valid_best = np.array(
        [r["best_err_m"] for r in out_rows if r["best_err_m"] is not None],
        dtype=np.float64
    )
    valid_imp = np.array(
        [r["center_to_best_improvement_mm"] for r in out_rows if r["center_to_best_improvement_mm"] is not None],
        dtype=np.float64
    )

    out["summary"].update({
        "localnn_radius": int(radius),
        "num_best_valid": int(np.sum([1 for r in out_rows if r["best_valid"]])),
        "num_best_surface_bad": int(np.sum([1 for r in out_rows if r["best_surface_bad"]])),
        "best_err_bad_thresh_m": float(best_err_bad_thresh_m),
        "mean_best_err_m": float(np.mean(valid_best)) if len(valid_best) > 0 else None,
        "median_best_err_m": float(np.median(valid_best)) if len(valid_best) > 0 else None,
        "max_best_err_m": float(np.max(valid_best)) if len(valid_best) > 0 else None,
        "mean_center_to_best_improvement_mm": float(np.mean(valid_imp)) if len(valid_imp) > 0 else None,
        "median_center_to_best_improvement_mm": float(np.median(valid_imp)) if len(valid_imp) > 0 else None,
        "max_center_to_best_improvement_mm": float(np.max(valid_imp)) if len(valid_imp) > 0 else None,
    })

    return out


def save_same_corr_localnn_csv(csv_path, trace_localnn):
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)

    fieldnames = [
        "point_id",
        "after_u", "after_v",
        "x_obj", "y_obj", "z_obj",
        "center_err_mm",
        "best_u", "best_v",
        "best_err_mm",
        "best_shift_px",
        "best_du", "best_dv",
        "center_to_best_improvement_mm",
        "err_after_px",
        "improvement_px",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in trace_localnn["rows"]:
            best_uv = r["best_uv"] if r["best_uv"] is not None else [None, None]
            best_offset = r["best_offset_uv"] if r["best_offset_uv"] is not None else [None, None]

            writer.writerow({
                "point_id": r["point_id"],
                "after_u": r["after_uv"][0],
                "after_v": r["after_uv"][1],
                "x_obj": r["xyz_obj"][0],
                "y_obj": r["xyz_obj"][1],
                "z_obj": r["xyz_obj"][2],
                "center_err_mm": r["center_err_mm"],
                "best_u": best_uv[0],
                "best_v": best_uv[1],
                "best_err_mm": r["best_err_mm"],
                "best_shift_px": r["best_shift_px"],
                "best_du": best_offset[0],
                "best_dv": best_offset[1],
                "center_to_best_improvement_mm": r["center_to_best_improvement_mm"],
                "err_after_px": r["err_after_px"],
                "improvement_px": r["improvement_px"],
            })


def draw_same_corr_top12_localnn_postrender(
    post_render_img,
    trace_localnn,
    out_path,
    topk=12,
):
    """
    center error worst points를 post-render 위에 표시:
    - red X: original after_uv(center)
    - green O: neighborhood에서 찾은 best_uv
    - yellow line: center -> best
    """
    rows = trace_localnn["rows"]
    if len(rows) == 0:
        img = post_render_img.copy()
        cv2.putText(img, "No same-corr points", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), img)
        return

    sortable = []
    for i, r in enumerate(rows):
        val = -1.0 if r["center_err_mm"] is None else float(r["center_err_mm"])
        sortable.append((val, i))
    sortable.sort(reverse=True, key=lambda x: x[0])

    chosen_idx = [i for _, i in sortable[:min(int(topk), len(sortable))]]

    left = post_render_img.copy()
    h, w = left.shape[:2]

    panel_w = 1200
    panel = np.full((h, panel_w, 3), 245, dtype=np.uint8)

    for idx in chosen_idx:
        r = rows[idx]
        pid = r["point_id"]

        c = tuple(np.round(r["after_uv"]).astype(int))
        b = None if r["best_uv"] is None else tuple(np.round(r["best_uv"]).astype(int))

        in_c = 0 <= c[0] < w and 0 <= c[1] < h
        in_b = b is not None and 0 <= b[0] < w and 0 <= b[1] < h

        if in_c:
            cv2.drawMarker(left, c, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        if in_b:
            cv2.circle(left, b, 6, (0, 255, 0), -1, cv2.LINE_AA)
        if in_c and in_b:
            cv2.line(left, c, b, (0, 255, 255), 2, cv2.LINE_AA)

        anchor = b if in_b else c
        if 0 <= anchor[0] < w and 0 <= anchor[1] < h:
            txt = (
                f"id={pid} center={r['center_err_mm']:.1f}mm "
                f"best={r['best_err_mm']:.1f}mm "
                f"shift={r['best_shift_px']:.1f}px"
            )
            cv2.putText(left, txt, (anchor[0] + 6, anchor[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(left, "Same-corr local neighborhood XYZ check", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(left, "red=center after_uv, green=best_uv in local window, yellow=center->best", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(panel, "Same-corr post-render local NN XYZ", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(panel, "compare center lookup vs best lookup inside local window", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

    y = 110
    block_h = 112

    for rank, idx in enumerate(chosen_idx):
        r = rows[idx]
        y0 = y + rank * block_h
        if y0 + 95 > h:
            break

        cv2.line(panel, (15, y0 - 15), (panel_w - 15, y0 - 15), (200, 200, 200), 1, cv2.LINE_AA)

        best_uv = r["best_uv"] if r["best_uv"] is not None else [np.nan, np.nan]
        best_off = r["best_offset_uv"] if r["best_offset_uv"] is not None else [np.nan, np.nan]
        best_xyz = r["best_xyz_lookup"] if r["best_xyz_lookup"] is not None else [np.nan, np.nan, np.nan]

        lines = [
            f"id={r['point_id']}   center={r['center_err_mm']:.2f}mm   best={r['best_err_mm']:.2f}mm   improve={r['center_to_best_improvement_mm']:.2f}mm",
            f"after_uv=({r['after_uv'][0]:.1f},{r['after_uv'][1]:.1f})   best_uv=({best_uv[0]:.1f},{best_uv[1]:.1f})",
            f"best_offset=(du={best_off[0]:.1f}, dv={best_off[1]:.1f})   shift={r['best_shift_px']:.2f}px",
            f"obj=({r['xyz_obj'][0]:.4f}, {r['xyz_obj'][1]:.4f}, {r['xyz_obj'][2]:.4f})",
            f"best_xyz=({best_xyz[0]:.4f}, {best_xyz[1]:.4f}, {best_xyz[2]:.4f})",
            f"err_after={r['err_after_px']:.2f}px   sparse_improvement={r['improvement_px']:.2f}px",
        ]

        for li, txt in enumerate(lines):
            cv2.putText(panel, txt, (20, y0 + li * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)

    canvas = np.concatenate([left, panel], axis=1)
    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), canvas)

def build_sparse_point_trace(
    pts2d_query_obs,
    pts2d_gallery_obs,
    pts3d_obj,
    K,
    R_before,
    t_before,
    R_after,
    t_after,
):
    """
    각 sparse correspondence 점(point_id)에 대해
    - query/gallery 2D
    - canonical 3D
    - top1/gallery pose projection
    - PnP pose projection
    - before/after query reprojection error
    - camera coords before/after
    를 모두 정리
    """
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
        },
        "rows": rows,
    }


def save_sparse_point_trace_csv(csv_path, trace):
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


def draw_sparse_point_trace_topk(
    query_img,
    trace,
    out_path,
    select_key="delta_uv_px",   # or "improvement_px"
    topk=10,
    title="",
):
    """
    point_id를 보이게 그리는 top-k labeled visualization
    - white : observed query
    - blue  : before(top1/gallery pose)
    - green : after(PnP pose)
    - yellow arrow : before -> after
    """
    rows = trace["rows"]
    img = query_img.copy()
    h, w = img.shape[:2]

    if len(rows) == 0:
        cv2.putText(img, "No sparse trace points", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), img)
        return

    vals = np.array([float(r[select_key]) for r in rows], dtype=np.float64)
    order = np.argsort(-vals)  # descending
    chosen = order[:min(int(topk), len(order))]

    for rank, idx in enumerate(chosen):
        r = rows[int(idx)]

        q = tuple(np.round(r["query_uv"]).astype(int))
        b = tuple(np.round(r["before_uv"]).astype(int))
        a = tuple(np.round(r["after_uv"]).astype(int))

        in_q = 0 <= q[0] < w and 0 <= q[1] < h
        in_b = 0 <= b[0] < w and 0 <= b[1] < h
        in_a = 0 <= a[0] < w and 0 <= a[1] < h

        if in_q:
            cv2.circle(img, q, 5, (255, 255, 255), -1, cv2.LINE_AA)   # white
        if in_b:
            cv2.circle(img, b, 5, (255, 100, 0), -1, cv2.LINE_AA)     # blue-ish
        if in_a:
            cv2.circle(img, a, 5, (0, 255, 0), -1, cv2.LINE_AA)       # green

        if in_b and in_a:
            cv2.arrowedLine(img, b, a, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.2)

        label_anchor = a if in_a else (b if in_b else q)
        if 0 <= label_anchor[0] < w and 0 <= label_anchor[1] < h:
            txt = f"id={r['point_id']} {select_key}={r[select_key]:.1f}"
            cv2.putText(
                img, txt,
                (label_anchor[0] + 6, label_anchor[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA
            )

    header = f"{title}  top{len(chosen)} by {select_key}"
    cv2.putText(img, header, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "white=query, blue=top1, green=PnP, yellow=before->after", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), img)


def draw_ply_overlay(query_img, ply_xyz, K, R, t,
                     max_points=8000, alpha=0.5,
                     point_color=(0, 255, 255), point_radius=2,
                     out_path=None):
    """
    canonical PLY 점들을 추정된 pose로 query 이미지에 반투명 투영.

    alpha  : 점 overlay 불투명도 (0=원본만, 1=점만)
    Returns: (overlay_img, n_points_in_frame)
    """
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
        print(f"  [PLY overlay] projected {kept}/{len(imgpts)} pts → {Path(out_path).name}")

    return result, kept


# def draw_correspondence_debug(query_img, pts2d_query, pts3d_canonical,
#                                K, R, t, out_path, max_draw=100):
#     """
#     Query 이미지에 2D-3D 대응점을 시각화.
#     - 파란 원: 실제 query 픽셀
#     - 노란 원: 추정된 pose로 3D를 재투영한 픽셀
#     """
#     img = query_img.copy()
#     rvec, _ = cv2.Rodrigues(R.astype(np.float64))
#     tvec = t.reshape(3, 1).astype(np.float64)
#     dist = np.zeros((4, 1), dtype=np.float64)

#     N = min(len(pts2d_query), max_draw)
#     idx = np.random.choice(len(pts2d_query), size=N, replace=False)

#     proj, _ = cv2.projectPoints(
#         pts3d_canonical[idx].astype(np.float32),
#         rvec, tvec, K.astype(np.float64), dist
#     )
#     proj = proj.reshape(-1, 2)

#     h, w = img.shape[:2]
#     for i in range(N):
#         px_q = tuple(np.round(pts2d_query[idx[i]]).astype(int))
#         px_p = tuple(np.round(proj[i]).astype(int))

#         in_bounds_q = 0 <= px_q[0] < w and 0 <= px_q[1] < h
#         in_bounds_p = 0 <= px_p[0] < w and 0 <= px_p[1] < h

#         if in_bounds_q:
#             cv2.circle(img, px_q, 4, (255, 100, 0), -1, cv2.LINE_AA)   # blue: observed
#         if in_bounds_p:
#             cv2.circle(img, px_p, 4, (0, 255, 255), -1, cv2.LINE_AA)  # yellow: reprojected
#         if in_bounds_q and in_bounds_p:
#             cv2.line(img, px_q, px_p, (200, 200, 200), 1, cv2.LINE_AA)

#     # Legend
#     cv2.circle(img, (20, 20), 6, (255, 100, 0), -1)
#     cv2.putText(img, "observed (query)", (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
#     cv2.circle(img, (20, 45), 6, (0, 255, 255), -1)
#     cv2.putText(img, "reprojected (3D→pose)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

#     ensure_dir(Path(out_path).parent)
#     cv2.imwrite(str(out_path), img)

def project_points_obj_to_img(pts3d_obj, K, R, t):
    """
    object/canonical 3D points -> image 2D projection
    R, t are assumed to be object-to-camera extrinsics.
    """
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
    observed_color=(255, 100, 0),   # blue-ish in BGR
    reproj_color=(0, 255, 255),     # yellow
    line_color=(180, 180, 180),
    title_prefix="",
    stats_json_path=None,
):
    """
    하나의 pose에 대해 동일 correspondence 집합을 시각화.
    - 파란 원: observed query 2D
    - 노란 원: reprojected 3D->2D
    - 회색 선: observed <-> reprojected
    """
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


def draw_correspondence_debug_before_after(
    query_img,
    pts2d_query,
    pts3d_obj,
    K,
    R_before,
    t_before,
    R_after,
    t_after,
    out_path,
    draw_idx=None,
    max_draw=100,
    color_obs=(255, 255, 255),        # white
    color_before=(255, 100, 0),       # blue-ish
    color_after=(0, 255, 0),          # green
    line_before=(180, 120, 80),
    line_after=(120, 220, 120),
    title_prefix="same correspondences: before vs after",
    stats_json_path=None,
):
    """
    같은 correspondence 집합에 대해 before / after pose reprojection을 한 장에 같이 표시.
    - 흰색: observed query 2D
    - 파랑계열: before reprojection
    - 초록: after reprojection
    """
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
            })
        return

    proj_before_all = project_points_obj_to_img(pts3d_obj, K, R_before, t_before)
    proj_after_all = project_points_obj_to_img(pts3d_obj, K, R_after, t_after)

    stats_before_all = compute_reprojection_stats(pts2d_query, proj_before_all)
    stats_after_all = compute_reprojection_stats(pts2d_query, proj_after_all)

    if draw_idx is None:
        N = min(len(pts2d_query), max_draw)
        rng = np.random.default_rng(42)
        draw_idx = rng.choice(len(pts2d_query), size=N, replace=False)
    else:
        draw_idx = np.asarray(draw_idx, dtype=np.int32)

    pts2d_draw = pts2d_query[draw_idx]
    proj_before_draw = proj_before_all[draw_idx]
    proj_after_draw = proj_after_all[draw_idx]

    stats_before_draw = compute_reprojection_stats(pts2d_draw, proj_before_draw)
    stats_after_draw = compute_reprojection_stats(pts2d_draw, proj_after_draw)

    for i in range(len(draw_idx)):
        px_q = tuple(np.round(pts2d_draw[i]).astype(int))
        px_b = tuple(np.round(proj_before_draw[i]).astype(int))
        px_a = tuple(np.round(proj_after_draw[i]).astype(int))

        in_q = 0 <= px_q[0] < w and 0 <= px_q[1] < h
        in_b = 0 <= px_b[0] < w and 0 <= px_b[1] < h
        in_a = 0 <= px_a[0] < w and 0 <= px_a[1] < h

        if in_q:
            cv2.circle(img, px_q, 3, color_obs, -1, cv2.LINE_AA)

        if in_b:
            cv2.circle(img, px_b, 3, color_before, -1, cv2.LINE_AA)
        if in_q and in_b:
            cv2.line(img, px_q, px_b, line_before, 1, cv2.LINE_AA)

        if in_a:
            cv2.circle(img, px_a, 3, color_after, -1, cv2.LINE_AA)
        if in_q and in_a:
            cv2.line(img, px_q, px_a, line_after, 1, cv2.LINE_AA)

    y = 25
    cv2.putText(img, title_prefix, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    y += 28

    cv2.circle(img, (20, y - 5), 5, color_obs, -1)
    cv2.putText(img, "observed query 2D", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    y += 24

    cv2.circle(img, (20, y - 5), 5, color_before, -1)
    cv2.putText(img, "before pose reprojection", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    y += 24

    cv2.circle(img, (20, y - 5), 5, color_after, -1)
    cv2.putText(img, "after pose reprojection", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    y += 28

    cv2.putText(img,
                f"before mean={stats_before_draw['mean_px']:.2f}px  med={stats_before_draw['median_px']:.2f}px  max={stats_before_draw['max_px']:.2f}px",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(img,
                f"after  mean={stats_after_draw['mean_px']:.2f}px  med={stats_after_draw['median_px']:.2f}px  max={stats_after_draw['max_px']:.2f}px",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), img)

    if stats_json_path is not None:
        save_json(stats_json_path, {
            "title_prefix": title_prefix,
            "num_points_total": int(len(pts2d_query)),
            "num_points_drawn": int(len(draw_idx)),
            "draw_idx": draw_idx.tolist(),
            "before_all": stats_before_all,
            "after_all": stats_after_all,
            "before_drawn_subset": stats_before_draw,
            "after_drawn_subset": stats_after_draw,
        })

def draw_points_only(query_img, pts2d, out_path, color=(255, 100, 0), radius=3, max_draw=None, label=None):
    """
    2D 점만 원본 query 이미지 위에 시각화.
    """
    img = query_img.copy()
    h, w = img.shape[:2]

    pts2d = np.asarray(pts2d, dtype=np.float64)
    if len(pts2d) == 0:
        if label is not None:
            cv2.putText(img, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), img)
        return

    if max_draw is not None and len(pts2d) > max_draw:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts2d), size=max_draw, replace=False)
        pts = pts2d[idx]
    else:
        pts = pts2d

    kept = 0
    for p in pts:
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
            kept += 1

    if label is not None:
        cv2.putText(img, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, f"drawn={kept}/{len(pts)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), img)


def filter_points_by_binary_mask(pts2d, mask_img):
    """
    full-image 좌표계의 2D 점들 중에서 binary mask 내부 점만 남김.
    mask_img: single-channel uint8, foreground > 0
    Returns:
      inside_mask: (N,) bool
    """
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
    """
    query image 상의 2D 분포가 한쪽에 몰리지 않도록 grid 기반 샘플링.
    scores가 있으면 높은 점수 우선 유지.

    Returns:
      keep_idx: 선택된 원본 인덱스 배열
      cell_stats: 각 cell별 선택 개수 dict
    """
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
        order = np.argsort(-scores[idxs])  # 높은 score 우선
        chosen = idxs[order[:max_per_cell]]
        keep.append(chosen)
        cell_stats[str(key)] = int(len(chosen))

    if len(keep) == 0:
        return np.array([], dtype=np.int32), cell_stats

    keep_idx = np.concatenate(keep, axis=0)
    keep_idx = np.unique(keep_idx)
    return keep_idx.astype(np.int32), cell_stats

def build_fixed_xyzobj_projection_trace(
    pts3d_obj,
    K,
    R_pose,
    t_pose,
    post_xyz_map=None,
):
    """
    고정된 xyz_obj 점들을 새 pose(R_pose, t_pose)로 projection.
    필요하면 post-render XYZ map에서 같은 픽셀 위치의 3D도 lookup.
    """
    pts3d_obj = np.asarray(pts3d_obj, dtype=np.float64)

    proj_uv = project_points_obj_to_img(pts3d_obj, K, R_pose, t_pose)
    cam_xyz = compute_camera_coords_of_points(pts3d_obj, R_pose, t_pose)

    rows = []
    post_xyz_lookup = None
    valid_lookup = None
    xyz_err_m = None

    if post_xyz_map is not None:
        post_xyz_lookup, valid_lookup = lookup_xyz_at_pixels(
            post_xyz_map,
            proj_uv,
            bilinear=True,
        )
        xyz_err_m = np.full((len(pts3d_obj),), np.nan, dtype=np.float64)
        if np.any(valid_lookup):
            xyz_err_m[valid_lookup] = np.linalg.norm(
                post_xyz_lookup[valid_lookup] - pts3d_obj[valid_lookup],
                axis=1
            )

    for i in range(len(pts3d_obj)):
        row = {
            "point_id": int(i),
            "xyz_obj": [
                float(pts3d_obj[i, 0]),
                float(pts3d_obj[i, 1]),
                float(pts3d_obj[i, 2]),
            ],
            "proj_uv": [
                float(proj_uv[i, 0]),
                float(proj_uv[i, 1]),
            ],
            "cam_xyz": [
                float(cam_xyz[i, 0]),
                float(cam_xyz[i, 1]),
                float(cam_xyz[i, 2]),
            ],
        }

        if post_xyz_map is not None:
            row["post_xyz_lookup"] = [
                float(post_xyz_lookup[i, 0]),
                float(post_xyz_lookup[i, 1]),
                float(post_xyz_lookup[i, 2]),
            ]
            row["post_xyz_valid"] = bool(valid_lookup[i])
            row["xyz_err_m"] = None if not np.isfinite(xyz_err_m[i]) else float(xyz_err_m[i])
            row["xyz_err_mm"] = None if not np.isfinite(xyz_err_m[i]) else float(xyz_err_m[i] * 1000.0)

        rows.append(row)

    out = {
        "num_points": int(len(rows)),
        "rows": rows,
        "summary": {},
    }

    if post_xyz_map is not None and xyz_err_m is not None:
        valid_err = xyz_err_m[np.isfinite(xyz_err_m)]
        out["summary"].update({
            "num_post_xyz_valid": int(np.sum(valid_lookup)),
            "mean_xyz_err_m": float(np.mean(valid_err)) if len(valid_err) > 0 else None,
            "median_xyz_err_m": float(np.median(valid_err)) if len(valid_err) > 0 else None,
            "max_xyz_err_m": float(np.max(valid_err)) if len(valid_err) > 0 else None,
        })

    return out


def save_fixed_xyzobj_projection_csv(csv_path, trace):
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)

    fieldnames = [
        "point_id",
        "proj_u", "proj_v",
        "x_obj", "y_obj", "z_obj",
        "cam_x", "cam_y", "cam_z",
        "post_x", "post_y", "post_z",
        "post_xyz_valid",
        "xyz_err_mm",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in trace["rows"]:
            post_xyz = r.get("post_xyz_lookup", [None, None, None])
            writer.writerow({
                "point_id": r["point_id"],
                "proj_u": r["proj_uv"][0],
                "proj_v": r["proj_uv"][1],
                "x_obj": r["xyz_obj"][0],
                "y_obj": r["xyz_obj"][1],
                "z_obj": r["xyz_obj"][2],
                "cam_x": r["cam_xyz"][0],
                "cam_y": r["cam_xyz"][1],
                "cam_z": r["cam_xyz"][2],
                "post_x": post_xyz[0],
                "post_y": post_xyz[1],
                "post_z": post_xyz[2],
                "post_xyz_valid": r.get("post_xyz_valid", None),
                "xyz_err_mm": r.get("xyz_err_mm", None),
            })


def draw_fixed_xyzobj_projection_on_postrender(
    post_render_img,
    trace,
    out_path,
    topk=20,
    select_key="xyz_err_mm",   # None이면 point_id 순서
    title="Fixed xyz_obj projected onto post-render",
):
    """
    고정된 xyz_obj를 새 pose로 projection한 결과를
    post-render RGB 위에 직접 표시.
    """
    rows = trace["rows"]
    img = post_render_img.copy()
    h, w = img.shape[:2]

    if len(rows) == 0:
        cv2.putText(img, "No fixed xyz_obj points", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), img)
        return

    if select_key is None:
        chosen_rows = rows[:min(int(topk), len(rows))]
    else:
        def _score(r):
            v = r.get(select_key, None)
            if v is None:
                return -1.0
            return float(v)
        chosen_rows = sorted(rows, key=_score, reverse=True)[:min(int(topk), len(rows))]

    for r in chosen_rows:
        pid = r["point_id"]
        uv = tuple(np.round(r["proj_uv"]).astype(int))

        if 0 <= uv[0] < w and 0 <= uv[1] < h:
            cv2.circle(img, uv, 5, (0, 255, 255), -1, cv2.LINE_AA)
            txt = f"id={pid}"
            if r.get("xyz_err_mm", None) is not None:
                txt += f" err={r['xyz_err_mm']:.1f}mm"
            cv2.putText(img, txt, (uv[0] + 6, uv[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, title, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "yellow = fixed xyz_obj projected by PnP pose", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), img)

def draw_points_before_after(query_img, pts_before, pts_after, out_path,
                             color_before=(255, 100, 0), color_after=(0, 255, 0),
                             radius_before=2, radius_after=2,
                             label_before="before", label_after="after"):
    """
    before / after 점 분포를 같은 query 이미지 위에 겹쳐서 시각화.
    """
    img = query_img.copy()
    h, w = img.shape[:2]

    pts_before = np.asarray(pts_before, dtype=np.float64)
    pts_after = np.asarray(pts_after, dtype=np.float64)

    for p in pts_before:
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), radius_before, color_before, -1, cv2.LINE_AA)

    for p in pts_after:
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), radius_after, color_after, -1, cv2.LINE_AA)

    cv2.circle(img, (20, 20), 5, color_before, -1)
    cv2.putText(img, label_before, (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    cv2.circle(img, (20, 45), 5, color_after, -1)
    cv2.putText(img, label_after, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), img)


def compute_mean_reprojection_error(pts2d, pts3d, K, R, t):
    """
    2D(query)와 3D(canonical) 대응에 대해,
    주어진 pose (R, t)로 3D를 query 이미지에 재투영했을 때의
    pixel distance 평균/중앙값을 계산.
    """
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
    """
    step45에서 단일 pose를 저장하기 위한 간단한 json.
    """
    data = {
        "width": int(width),
        "height": int(height),
        "K": np.asarray(K, dtype=np.float64).tolist(),
        "R_obj_to_cam": np.asarray(R, dtype=np.float64).tolist(),
        "t_obj_to_cam": np.asarray(t, dtype=np.float64).reshape(3).tolist(),
    }
    save_json(path, data)


def overlay_render_on_query(query_img, render_img, out_path, alpha=0.45, nonblack_thresh=8):
    """
    render 이미지의 non-black 영역만 query 위에 alpha blend.
    """
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
    """
    render 이미지에서 non-black 영역을 binary mask로 변환.
    """
    gray = cv2.cvtColor(render_img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > nonblack_thresh).astype(np.uint8) * 255
    return mask


def points_inside_binary_mask(pts2d, mask_img):
    """
    pts2d: (N, 2) float
    mask_img: uint8 single-channel, fg > 0
    Returns:
      inside: (N,) bool
    """
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
    """
    after-pose 기준으로 PnP inlier들 중
    render(non-black) 영역 밖에 있는 local index를 반환.
    """
    if len(pts3d_inliers) == 0:
        return np.array([], dtype=np.int32), np.empty((0, 2), dtype=np.float64)

    after_mask = render_to_binary_mask(render_img_after, nonblack_thresh=nonblack_thresh)
    proj_after = project_points_obj_to_img(pts3d_inliers, K, R_after, t_after)
    inside_after = points_inside_binary_mask(proj_after, after_mask)
    bad_local_idx = np.where(~inside_after)[0].astype(np.int32)
    return bad_local_idx, proj_after

def draw_inlier_motion_with_region_check(
    query_img,
    pts2d_obs,
    pts3d_obj,
    K,
    R_before,
    t_before,
    R_after,
    t_after,
    out_path,
    draw_idx=None,
    max_draw=120,
    region_mask_before=None,
    region_mask_after=None,
    title_prefix="same inliers: before -> after",
    stats_json_path=None,
):
    """
    같은 inlier set에 대해:
      - observed query point
      - before reprojection
      - after reprojection
    을 동시에 그리고,
    region_mask_before / after 가 있으면
      - render 영역 inside/outside 여부도 같이 표시한다.

    색상 규칙:
      흰색  : observed query 2D
      주황  : before reprojection
      초록  : after reprojection (region inside)
      빨강  : after reprojection (region outside)
      회색선: observed -> before
      연두선: observed -> after
    """
    img = query_img.copy()
    h, w = img.shape[:2]

    pts2d_obs = np.asarray(pts2d_obs, dtype=np.float64)
    pts3d_obj = np.asarray(pts3d_obj, dtype=np.float64)

    if len(pts2d_obs) != len(pts3d_obj):
        raise ValueError(
            f"Length mismatch: len(pts2d_obs)={len(pts2d_obs)} vs len(pts3d_obj)={len(pts3d_obj)}"
        )

    if len(pts2d_obs) == 0:
        cv2.putText(img, "No inliers", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), img)
        if stats_json_path is not None:
            save_json(stats_json_path, {
                "num_points_total": 0,
                "num_points_drawn": 0,
            })
        return

    proj_before_all = project_points_obj_to_img(pts3d_obj, K, R_before, t_before)
    proj_after_all  = project_points_obj_to_img(pts3d_obj, K, R_after,  t_after)

    if draw_idx is None:
        N = min(len(pts2d_obs), max_draw)
        rng = np.random.default_rng(42)
        draw_idx = rng.choice(len(pts2d_obs), size=N, replace=False)
    else:
        draw_idx = np.asarray(draw_idx, dtype=np.int32)

    obs_draw = pts2d_obs[draw_idx]
    bef_draw = proj_before_all[draw_idx]
    aft_draw = proj_after_all[draw_idx]

    before_inside = None
    after_inside = None
    if region_mask_before is not None:
        before_inside = points_inside_binary_mask(bef_draw, region_mask_before)
    if region_mask_after is not None:
        after_inside = points_inside_binary_mask(aft_draw, region_mask_after)

    # draw
    n_after_outside = 0
    for i in range(len(draw_idx)):
        pq = tuple(np.round(obs_draw[i]).astype(int))
        pb = tuple(np.round(bef_draw[i]).astype(int))
        pa = tuple(np.round(aft_draw[i]).astype(int))

        in_q = 0 <= pq[0] < w and 0 <= pq[1] < h
        in_b = 0 <= pb[0] < w and 0 <= pb[1] < h
        in_a = 0 <= pa[0] < w and 0 <= pa[1] < h

        # observed
        if in_q:
            cv2.circle(img, pq, 3, (255, 255, 255), -1, cv2.LINE_AA)

        # before
        if in_b:
            cv2.circle(img, pb, 3, (255, 140, 0), -1, cv2.LINE_AA)
        if in_q and in_b:
            cv2.line(img, pq, pb, (180, 140, 80), 1, cv2.LINE_AA)

        # after
        after_color = (0, 255, 0)
        if after_inside is not None and not after_inside[i]:
            after_color = (0, 0, 255)   # outside region => red
            n_after_outside += 1

        if in_a:
            cv2.circle(img, pa, 4, after_color, -1, cv2.LINE_AA)
        if in_q and in_a:
            cv2.line(img, pq, pa, (120, 220, 120), 1, cv2.LINE_AA)

        # motion vector before -> after
        if in_b and in_a:
            cv2.line(img, pb, pa, (255, 0, 255), 1, cv2.LINE_AA)

    # legend
    y = 25
    cv2.putText(img, title_prefix, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
    y += 28

    cv2.circle(img, (20, y-5), 5, (255,255,255), -1)
    cv2.putText(img, "observed query 2D", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
    y += 22

    cv2.circle(img, (20, y-5), 5, (255,140,0), -1)
    cv2.putText(img, "before reprojection", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
    y += 22

    cv2.circle(img, (20, y-5), 5, (0,255,0), -1)
    cv2.putText(img, "after reprojection (inside render region)", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
    y += 22

    cv2.circle(img, (20, y-5), 5, (0,0,255), -1)
    cv2.putText(img, "after reprojection (outside render region)", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
    y += 22

    cv2.putText(img, f"drawn={len(draw_idx)}  after_outside={n_after_outside}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)

    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), img)

    if stats_json_path is not None:
        stats = {
            "num_points_total": int(len(pts2d_obs)),
            "num_points_drawn": int(len(draw_idx)),
            "draw_idx": draw_idx.tolist(),
            "before_reproj_stats_drawn": compute_reprojection_stats(obs_draw, bef_draw),
            "after_reproj_stats_drawn": compute_reprojection_stats(obs_draw, aft_draw),
            "after_outside_render_count": int(n_after_outside),
        }
        if before_inside is not None:
            stats["before_inside_render_count"] = int(np.sum(before_inside))
            stats["before_outside_render_count"] = int(len(before_inside) - np.sum(before_inside))
        if after_inside is not None:
            stats["after_inside_render_count"] = int(np.sum(after_inside))
            stats["after_outside_render_count"] = int(len(after_inside) - np.sum(after_inside))
        save_json(stats_json_path, stats)


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
    # 1) query_mask.png 우선
    if query_mask_path is not None:
        query_mask_path = Path(query_mask_path)
        if query_mask_path.exists():
            qmask = cv2.imread(str(query_mask_path), cv2.IMREAD_GRAYSCALE)
            if qmask is not None:
                return ((qmask > 0).astype(np.uint8) * 255)

    # 2) query_masked_full/non-black fallback
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
    """
    object origin의 image center를 크게 바꾸지 않기 위해
    tx, ty를 tz에 비례해서 같이 스케일.
    u = fx * tx / tz + cx, v = fy * ty / tz + cy 를 대략 유지.
    """
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

    # 화면 대각선 기준 정규화
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
    """
    R 고정, t=[tx,ty,tz]만 mask 기준으로 refine.
    - tx,ty: bbox center 차이를 pixel -> world 로 환산
    - tz   : bbox height ratio로 업데이트
    """
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

    # prior sanity clamp (강제 init이 아니라 과도한 tz 방지)
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

            # --------------------------------------------------
            # 1) center alignment -> tx, ty update
            # render center를 query center로 이동시키는 방향
            # --------------------------------------------------
            du = q_stats["cx"] - r_stats["cx"]   # pixel
            dv = q_stats["cy"] - r_stats["cy"]   # pixel

            # image-space -> camera translation
            dtx = alpha_xy * (du / fx) * float(t_cur[2])
            dty = alpha_xy * (dv / fy) * float(t_cur[2])

            t_new = t_cur.copy()
            t_new[0] += dtx
            t_new[1] += dty

            # --------------------------------------------------
            # 2) scale alignment -> tz update
            # render가 query보다 작으면 h_render < h_query -> tz 줄여야 함
            # --------------------------------------------------
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
    """
    단일 pose에서 3DGS 렌더를 생성.
    save_xyz=True이면 XYZ map(.npy)도 함께 저장.
    """
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

def full_to_query_masked_coords(pts2d_full, step1_crop_offset_xy):
    """
    full original image 좌표 -> query_masked image 좌표
    step1_crop_offset_xy: (2,) or (1,2), usually [x1, y1] of step1 bbox.
    """
    pts2d_full = np.asarray(pts2d_full, dtype=np.float64)
    off = np.asarray(step1_crop_offset_xy, dtype=np.float64).reshape(1, 2)
    return pts2d_full - off


def draw_query_render_correspondence(
    query_img,
    render_img,
    pts_query,
    pts_render,
    out_path,
    draw_idx=None,
    max_draw=200,
    title_prefix="query <-> render correspondences",
    stats_json_path=None,
):
    """
    masked query 이미지와 rendered 이미지 사이 correspondence 시각화.
    왼쪽: query (원본 크기)
    오른쪽: render (query 높이에 맞게 리사이즈 → 좌표도 스케일)
    선으로 1:1 대응 연결

    색상:
      녹색 선/점 : 양쪽 모두 이미지 안
      빨간 선/점 : 한쪽이라도 이미지 밖
    """
    query_img = np.asarray(query_img)
    render_img = np.asarray(render_img)
    pts_query = np.asarray(pts_query, dtype=np.float64)
    pts_render = np.asarray(pts_render, dtype=np.float64)

    if len(pts_query) != len(pts_render):
        raise ValueError(
            f"Length mismatch: len(pts_query)={len(pts_query)} vs len(pts_render)={len(pts_render)}"
        )

    hq, wq = query_img.shape[:2]
    hr, wr = render_img.shape[:2]

    # render를 query 높이에 맞게 리사이즈해서 시각적 스케일 통일
    if hr != hq:
        scale_r = hq / hr
        new_wr = int(wr * scale_r)
        render_img_vis = cv2.resize(render_img, (new_wr, hq), interpolation=cv2.INTER_AREA)
    else:
        scale_r = 1.0
        new_wr = wr
        render_img_vis = render_img.copy()

    # render 좌표도 동일 비율로 스케일
    pts_render_vis = pts_render * scale_r

    H = hq
    W = wq + new_wr

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:hq, :wq] = query_img
    canvas[:hq, wq:wq+new_wr] = render_img_vis

    # 이후 코드에서 wr → new_wr, pts_render → pts_render_vis 사용
    wr = new_wr
    hr = hq
    pts_render = pts_render_vis

    if len(pts_query) == 0:
        cv2.putText(canvas, "No correspondences", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), canvas)
        if stats_json_path is not None:
            save_json(stats_json_path, {
                "title_prefix": title_prefix,
                "num_points_total": 0,
                "num_points_drawn": 0,
                "num_inside_both": 0,
                "num_outside_any": 0,
            })
        return

    if draw_idx is None:
        N = min(len(pts_query), max_draw)
        rng = np.random.default_rng(42)
        draw_idx = rng.choice(len(pts_query), size=N, replace=False)
    else:
        draw_idx = np.asarray(draw_idx, dtype=np.int32)

    q_draw = pts_query[draw_idx]
    r_draw = pts_render[draw_idx]

    inside_both = 0
    outside_any = 0

    for i in range(len(draw_idx)):
        qx, qy = np.round(q_draw[i]).astype(int)
        rx, ry = np.round(r_draw[i]).astype(int)

        in_q = (0 <= qx < wq) and (0 <= qy < hq)
        in_r = (0 <= rx < wr) and (0 <= ry < hr)

        color = (0, 255, 0) if (in_q and in_r) else (0, 0, 255)

        p0 = (int(qx), int(qy))
        p1 = (int(rx + wq), int(ry))

        if in_q:
            cv2.circle(canvas, p0, 3, (0, 255, 255), -1, cv2.LINE_AA)
        if in_r:
            cv2.circle(canvas, p1, 3, (0, 255, 255), -1, cv2.LINE_AA)

        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)

        if in_q and in_r:
            inside_both += 1
        else:
            outside_any += 1

    y = 25
    cv2.putText(canvas, title_prefix, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
    y += 28
    cv2.putText(canvas, f"drawn={len(draw_idx)}  inside_both={inside_both}  outside_any={outside_any}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(canvas, "green: inside both query/render, red: outside any image",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)

    cv2.putText(canvas, "query", (20, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "render", (wq + 20, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    ensure_dir(Path(out_path).parent)
    # canvas가 너무 크면 1920px 너비로 리사이즈해서 저장 (시각화 용도)
    MAX_VIS_W = 3840
    if canvas.shape[1] > MAX_VIS_W:
        vis_scale = MAX_VIS_W / canvas.shape[1]
        vis_h = int(canvas.shape[0] * vis_scale)
        canvas_save = cv2.resize(canvas, (MAX_VIS_W, vis_h), interpolation=cv2.INTER_AREA)
    else:
        canvas_save = canvas
    cv2.imwrite(str(out_path), canvas_save)

    if stats_json_path is not None:
        save_json(stats_json_path, {
            "title_prefix": title_prefix,
            "num_points_total": int(len(pts_query)),
            "num_points_drawn": int(len(draw_idx)),
            "draw_idx": draw_idx.tolist(),
            "num_inside_both": int(inside_both),
            "num_outside_any": int(outside_any),
        })

def draw_points_on_render(render_img, pts2d, out_path,
                          color=(0, 255, 255), radius=3,
                          label=None):
    img = render_img.copy()
    h, w = img.shape[:2]

    pts2d = np.asarray(pts2d, dtype=np.float64)
    kept = 0
    for p in pts2d:
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
            kept += 1

    if label is not None:
        cv2.putText(img, label, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, f"drawn={kept}/{len(pts2d)}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), img)

# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def run_step6_translation(args):
    """
    args에서 사용하는 필드:
      --out_dir
      --gallery_pose_json
      --gallery_xyz_dir          (step4에서 저장한 XYZ map 디렉토리)
      --intrinsics_path
      --query_img
      --pnp_reproj_error         (default: 4.0)
      --axis_len_m               (axes 시각화용, default: 0.04)
      --device                   (사용 안 함, 호환성용)
    """
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # ── 1. step4 결과 로드 ────────────────────────────────────────────────────
    npz_path = out_dir / "loftr_best_match_data.npz"
    meta_path = out_dir / "loftr_best_match_meta.json"
    loftr_json_path = out_dir / "loftr_scores.json"

    if not npz_path.exists():
        raise FileNotFoundError(
            f"loftr_best_match_data.npz not found: {npz_path}\n"
            "step4 (dino_loftr)를 먼저 실행하세요."
        )

    match_data = np.load(str(npz_path))
    mkpts0_840 = match_data["mkpts0_inlier_840"].astype(np.float64)   # (N, 2) query
    mkpts1_840 = match_data["mkpts1_inlier_840"].astype(np.float64)   # (N, 2) gallery
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

    # ── 2. 좌표 역변환 ────────────────────────────────────────────────────────
    resize_target = int(meta["loftr_resize_target"])

    # Query: 840px → full image 좌표 (crop 없음)
    # meta["query_crop_hw"] = full image (H, W)
    # meta["query_nonblack_bbox_xyxy"] = [0, 0, W, H]
    # → unmap 후 그대로 full image 좌표
    q_crop_hw = tuple(meta["query_crop_hw"])
    q_bbox    = meta["query_nonblack_bbox_xyxy"]   # [0, 0, W, H]
    pts_q_crop = unmap_from_square_resize(mkpts0_840, q_crop_hw, resize_target)
    pts_q_full = to_full_image_coords(pts_q_crop, q_bbox)   # full image 좌표 (직접)

    # q_bbox = [x1, y1, x2, y2] (full image 기준 crop bbox)
    # to_full_image_coords가 [x1, y1]을 더해서 full image 좌표로 변환함
    # → 추가 offset 불필요
    if len(pts_q_full) > 0:
        print(f"  pts_q_full range: x=[{pts_q_full[:,0].min():.0f},{pts_q_full[:,0].max():.0f}], y=[{pts_q_full[:,1].min():.0f},{pts_q_full[:,1].max():.0f}]")

    # Gallery: 840px → full gallery render image (crop 없음)
    # meta["gallery_crop_hw"] = full image (H, W)
    # meta["gallery_nonblack_bbox_xyxy"] = [0, 0, W, H]
    g_crop_hw = tuple(meta["gallery_crop_hw"])
    g_bbox = meta["gallery_nonblack_bbox_xyxy"]  # [0, 0, W, H] (full)
    pts_g_crop = unmap_from_square_resize(mkpts1_840, g_crop_hw, resize_target)
    pts_g_full = to_full_image_coords(pts_g_crop, g_bbox)   # (N, 2) in full gallery image
    if len(pts_g_full) > 0:
        print(f"  pts_g_full range: x=[{pts_g_full[:,0].min():.0f},{pts_g_full[:,0].max():.0f}], y=[{pts_g_full[:,1].min():.0f},{pts_g_full[:,1].max():.0f}]")

    # ── 3. XYZ map 로드 및 3D 좌표 lookup ────────────────────────────────────
    xyz_dir = Path(args.gallery_xyz_dir)
    xyz_map_path = find_xyz_map_path(xyz_dir, best_render)
    xyz_map = np.load(str(xyz_map_path)).astype(np.float64)            # (H, W, 3)

    print(f"  XYZ map: {xyz_map_path.name}  shape={xyz_map.shape}")

    # ── Runtime XYZ map scale 보정 ────────────────────────────────────────
    # render_gallery.py의 depth_scale heuristic(median 기반)에 ~6% 오차가 있으므로,
    # canonical PLY의 실제 범위와 비교하여 runtime에서 보정.
    # 방법: XYZ map의 유효 점을 canonical PLY와 비교하여 scale factor 추정.
    canonical_ply_path = getattr(args, "canonical_ply_path", None)
    if canonical_ply_path is not None and Path(canonical_ply_path).exists():
        _ply_xyz = load_ply_xyz(Path(canonical_ply_path))
        _ply_norms = np.linalg.norm(_ply_xyz, axis=1)
        _ply_med_norm = float(np.median(_ply_norms))

        _xyz_valid_mask = np.abs(xyz_map).sum(axis=-1) > 1e-6
        if _xyz_valid_mask.any():
            _xyz_valid = xyz_map[_xyz_valid_mask]
            _xyz_norms = np.linalg.norm(_xyz_valid, axis=1)
            # PLY 범위 밖의 극단 outlier 제외 (상위/하위 5%)
            _p5, _p95 = np.percentile(_xyz_norms, [5, 95])
            _core_mask = (_xyz_norms >= _p5) & (_xyz_norms <= _p95)
            if _core_mask.sum() > 100:
                _xyz_med_norm = float(np.median(_xyz_norms[_core_mask]))
                _scale_correction = _ply_med_norm / _xyz_med_norm
                # 보정이 합리적 범위 (0.8 ~ 1.2) 일 때만 적용
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
    pts_g_corr = pts_g_full[valid_mask]   # gallery 픽셀 tracking

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

    # mask filtering 끝난 뒤의 상태를 before_uniform으로 저장
    pts2d_before_uniform = pts2d_corr.copy()
    pts3d_before_uniform = pts3d_corr.copy()

    # confidence도 동일하게 줄이기
    conf_corr = conf_inlier[valid_mask]
    if query_mask_path:
        conf_corr = conf_corr[inside_mask]

    # ── query-side spatial uniform sampling ────────────────────────────────────
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

    # ── 4. Gallery pose에서 R 로드 ────────────────────────────────────────────
    # ── 3D outlier 필터 ─────────────────────────────────────────────────────
    # XYZ map lookup된 pts3d 중 norm이 극단적으로 큰 점 제거
    # (배경/노이즈 픽셀이 섞인 경우 재투영 시 극단값 발생)
    _norms = np.linalg.norm(pts3d_corr, axis=1)
    _median_norm = np.median(_norms)
    _outlier_thresh = _median_norm * 5.0   # median의 5배 초과 = outlier
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
    t_gallery = np.array(pose_record["t_obj_to_cam"], dtype=np.float64)  # gallery render 생성에 사용된 원래 t

    # ── 5. Intrinsics ─────────────────────────────────────────────────────────
    K = load_intrinsics(args.intrinsics_path)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # render_gallery.py와 render_single_pose.py 모두 cx/cy를 projection matrix에
    # 반영하도록 수정됨 → K 하나로 통일 사용
    print(f"  K: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    
    # ── 7. solvePnP: pts2d_corr는 이미 full image 좌표계 ──────────────────────
    reproj_thresh = float(getattr(args, "pnp_reproj_error", 200.0))

    print(f"  pts2d_corr are already in full-image coordinates: shape={pts2d_corr.shape}")

    # ── 공통 시각화 변수 미리 준비 ────────────────────────────────────────────
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

    # ── PnP 전 pose 시각화 (best render pose + coarse t_init) ─────────────────
    pre_ply_overlay_path = None
    if canonical_ply_path is not None:
        canonical_ply_path = Path(canonical_ply_path)
        if canonical_ply_path.exists():
            ply_xyz = load_ply_xyz(canonical_ply_path)
        else:
            print(f"  [Pre-PnP PLY overlay] canonical_ply_path not found: {canonical_ply_path}")

    no_pnp = bool(getattr(args, "no_pnp", False))

    if no_pnp:
        # ── PnP 완전 건너뜀: R_gallery 그대로, t는 mask bbox로 초기 추정 ──────
        R_out = R_gallery.copy()
        inlier_idx = np.arange(len(pts2d_corr), dtype=np.int32)
        inlier_count = len(pts2d_corr)
        reproj_err = np.inf
        pose_method = "no_pnp_gallery_R"

        # mask bbox에서 tz 추정 → tx, ty는 mask center로 추정
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
                # image center → camera coords: tx = (u - cx) / fx * tz
                tx_init = (q_stats_init["cx"] - cx) / fx * tz_init
                ty_init = (q_stats_init["cy"] - cy) / fy * tz_init
                t_out = np.array([tx_init, ty_init, tz_init], dtype=np.float64)
                print(f"  [no_pnp] bbox-based t_init: [{tx_init:.4f}, {ty_init:.4f}, {tz_init:.4f}]")
            else:
                t_out = t_gallery.copy()
                print(f"  [no_pnp] mask bbox 추정 실패 → gallery t 사용")
        else:
            t_out = t_gallery.copy()
            print(f"  [no_pnp] mask 없음 → gallery t 사용")

        print(f"  [no_pnp] R = R_gallery, t = bbox_prior  (PnP 건너뜀)")
    else:
        R_out, t_out, pose_method, inlier_count, reproj_err, inlier_idx = solve_pose_pnp(
            pts2d_corr,
            pts3d_corr,
            K,
            R_init=R_gallery,
            reproj_thresh=reproj_thresh,
            use_ransac=not bool(getattr(args, "no_pnp_ransac", False)),
        )

    # ── Correspondence debug visualization ────────────────────────────────────
    # gallery render 경로: xyz_dir 옆 renders 디렉토리 또는 dino_scores_json 위치에서 추론
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

    # ------------------------------------------------------------
    # Debug A: sparse matched 3D points motion
    #   - same matched 3D points X_i
    #   - top1/gallery pose 에서 어디였는지
    #   - PnP pose 에서 어디로 갔는지
    # ------------------------------------------------------------
    save_sparse_correspondence_motion_debug(
        query_img=query_img,
        pts2d_query_obs=pts2d_corr,
        pts2d_gallery_obs=pts_g_corr,
        pts3d_obj=pts3d_corr,
        K=K,
        R_before=R_gallery,
        t_before=t_gallery,
        R_after=R_out,
        t_after=t_out,
        out_png=out_dir / "step6_sparse_points_before_after_query.png",
        out_json=out_dir / "step6_sparse_points_before_after.json",
        max_draw=200,
    )
    print("  [Sparse motion debug] saved: step6_sparse_points_before_after_query.png/json")

    # ------------------------------------------------------------
    # Debug B: dense xyz-map motion sampled from top1 XYZ map
    #   - top1 xyz_map valid points 전체(샘플링)
    #   - gallery pose 에서 어디였는지
    #   - PnP pose 에서 어디로 갔는지
    # ------------------------------------------------------------
    try:
        xyz_map_path_dbg = find_xyz_map_path(xyz_dir, best_render)
        xyz_map_dbg = np.load(str(xyz_map_path_dbg)).astype(np.float64)

        save_dense_xyzmap_motion_debug(
            query_img=query_img,
            xyz_map=xyz_map_dbg,
            K=K,
            R_before=R_gallery,
            t_before=t_gallery,
            R_after=R_out,
            t_after=t_out,
            out_png=out_dir / "step6_xyzmap_motion_sampled_query.png",
            out_json=out_dir / "step6_xyzmap_motion_sampled.json",
            max_points=2000,
        )
        print("  [Dense xyz-map motion] saved: step6_xyzmap_motion_sampled_query.png/json")
    except Exception as e:
        print(f"  [Dense xyz-map motion] failed: {e}")

    # ------------------------------------------------------------
    # Same correspondence points only motion trace
    # ------------------------------------------------------------
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

    save_json(out_dir / "step6_same_corr_motion_trace.json", same_corr_trace)
    save_same_corr_motion_csv(out_dir / "step6_same_corr_motion_trace.csv", same_corr_trace)

    try:
        gallery_img_samecorr = load_image(_gallery_render_path)
        draw_same_corr_ids_on_gallery(
            gallery_img=gallery_img_samecorr,
            trace=same_corr_trace,
            out_path=out_dir / "step6_same_corr_ids_gallery.png",
        )

        draw_same_corr_topk_gallery(
            gallery_img=gallery_img_samecorr,
            trace=same_corr_trace,
            out_path=out_dir / "step6_same_corr_top12_gallery_motion.png",
            select_key="gallery_to_pnp_motion_px",
            topk=12,
            title="Same correspondence motion on gallery",
        )

        draw_same_corr_topk_gallery(
            gallery_img=gallery_img_samecorr,
            trace=same_corr_trace,
            out_path=out_dir / "step6_same_corr_top12_gallery_reproj_err.png",
            select_key="gallery_reproj_err_px",
            topk=12,
            title="Gallery reprojection sanity",
        )

        draw_same_corr_topk_query_with_3d_panel(
            query_img=query_img,
            trace=same_corr_trace,
            out_path=out_dir / "step6_same_corr_top6_improvement_query_3dpanel.png",
            select_key="improvement_px",
            topk=6,
            title="Same corr improvement with 3D pairing",
        )

        draw_same_corr_topk_query_with_3d_panel(
            query_img=query_img,
            trace=same_corr_trace,
            out_path=out_dir / "step6_same_corr_top6_motion_query_3dpanel.png",
            select_key="delta_uv_px",
            topk=6,
            title="Same corr motion with 3D pairing",
        )


    except Exception as e:
        print(f"  [Same corr ids gallery] failed: {e}")

    draw_same_corr_topk_query(
        query_img=query_img,
        trace=same_corr_trace,
        out_path=out_dir / "step6_same_corr_top12_motion_query.png",
        select_key="delta_uv_px",
        topk=12,
        title="Same correspondence motion",
    )

    draw_same_corr_topk_query(
        query_img=query_img,
        trace=same_corr_trace,
        out_path=out_dir / "step6_same_corr_top12_improvement_query.png",
        select_key="improvement_px",
        topk=12,
        title="Same correspondence improvement",
    )

    s = same_corr_trace["summary"]
    print(
        f"  [Same corr trace] N={same_corr_trace['num_points']}  "
        f"mean_delta_uv={s['mean_delta_uv_px']:.2f}px  "
        f"mean_err_before={s['mean_err_before_px']:.2f}px  "
        f"mean_err_after={s['mean_err_after_px']:.2f}px  "
        f"mean_improvement={s['mean_improvement_px']:.2f}px"
    )
    print("  [Same corr trace] saved: step6_same_corr_motion_trace.json/csv")
    print("  [Same corr trace] saved: step6_same_corr_ids_gallery.png")
    print("  [Same corr trace] saved: step6_same_corr_top12_motion_query.png")
    print("  [Same corr trace] saved: step6_same_corr_top12_improvement_query.png")
    print("  [Same corr trace] saved: step6_same_corr_top12_gallery_motion.png")
    print("  [Same corr trace] saved: step6_same_corr_top12_gallery_reproj_err.png")

    # ------------------------------------------------------------
    # Point-ID trace for sparse correspondences
    #   what each sparse point did from top1 -> PnP
    # ------------------------------------------------------------
    sparse_trace = build_sparse_point_trace(
        pts2d_query_obs=pts2d_corr,
        pts2d_gallery_obs=pts_g_corr,
        pts3d_obj=pts3d_corr,
        K=K,
        R_before=R_gallery,
        t_before=t_gallery,
        R_after=R_out,
        t_after=t_out,
    )

    save_json(out_dir / "step6_sparse_point_trace.json", sparse_trace)
    save_sparse_point_trace_csv(out_dir / "step6_sparse_point_trace.csv", sparse_trace)

    draw_sparse_point_trace_topk(
        query_img=query_img,
        trace=sparse_trace,
        out_path=out_dir / "step6_sparse_top10_motion_query.png",
        select_key="delta_uv_px",
        topk=10,
        title="Sparse point trace",
    )

    draw_sparse_point_trace_topk(
        query_img=query_img,
        trace=sparse_trace,
        out_path=out_dir / "step6_sparse_top10_improvement_query.png",
        select_key="improvement_px",
        topk=10,
        title="Sparse point trace",
    )

    s = sparse_trace["summary"]
    print(
        f"  [Sparse point trace] N={sparse_trace['num_points']}  "
        f"mean_delta_uv={s['mean_delta_uv_px']:.2f}px  "
        f"mean_err_before={s['mean_err_before_px']:.2f}px  "
        f"mean_err_after={s['mean_err_after_px']:.2f}px  "
        f"mean_improvement={s['mean_improvement_px']:.2f}px"
    )
    print("  [Sparse point trace] saved: step6_sparse_point_trace.json/csv")
    print("  [Sparse point trace] saved: step6_sparse_top10_motion_query.png")
    print("  [Sparse point trace] saved: step6_sparse_top10_improvement_query.png")
    # ------------------------------------------------------------
    # Test 1 + Test 2:
    # XYZ-map point ↔ nearest PLY vertex consistency
    # ------------------------------------------------------------
    pts3d_ply_nn = None
    pts3d_ply_nn_idx = None
    pts3d_ply_nn_dist = None

    if canonical_ply_path is not None and "ply_xyz" in locals():
        gallery_render_img = None
        if _gallery_render_path.exists():
            gallery_render_img = load_image(_gallery_render_path)

        xyz_ply_nn_result = evaluate_xyz_vs_ply_nn_consistency(
            pts2d_gallery=pts_g_corr,
            pts3d_xyz=pts3d_corr,
            ply_xyz=ply_xyz,
            K=K,
            R_gallery=R_gallery,
            t_gallery=t_gallery,
            R_eval=R_out,
            t_eval=t_out,
            gallery_render_img=gallery_render_img,
            out_vis_path=out_dir / "step6_xyz_vs_ply_nn_gallery.png",
        )

        pts3d_ply_nn = xyz_ply_nn_result["pts3d_ply_nn"]
        pts3d_ply_nn_idx = xyz_ply_nn_result["nearest_idx"]
        pts3d_ply_nn_dist = xyz_ply_nn_result["nn_dist_m"]

        save_xyz_vs_ply_nn_json(
            out_dir / "step6_xyz_vs_ply_nn_gallery.json",
            xyz_ply_nn_result,
        )

        print(
            f"  [XYZ↔PLY NN] N={len(pts3d_corr)}  "
            f"mean_nn={xyz_ply_nn_result['mean_nn_dist_m']*1000:.2f}mm  "
            f"mean_dproj_gallery={xyz_ply_nn_result['mean_delta_proj_gallery_px']:.3f}px  "
            f"mean_dproj_eval={xyz_ply_nn_result.get('mean_delta_proj_eval_px', float('nan')):.3f}px"
        )
    else:
        print("  [XYZ↔PLY NN] skipped (canonical_ply_path / ply_xyz unavailable)")

     # ------------------------------------------------------------
    # Test 3:
    # Re-run PnP with nearest-PLY 3D points instead of XYZ-map points
    # ------------------------------------------------------------
    if pts3d_ply_nn is not None and len(pts3d_ply_nn) == len(pts2d_corr):
        R_ply_test, t_ply_test, pose_method_ply, inlier_count_ply, reproj_err_ply, inlier_idx_ply = solve_pose_pnp(
            pts2d_corr,
            pts3d_ply_nn,
            K,
            R_init=R_gallery,
            reproj_thresh=reproj_thresh,
            use_ransac=not bool(getattr(args, "no_pnp_ransac", False)),
        )

        rot_diff_deg_xyz_vs_ply = rotation_geodesic_deg(R_out, R_ply_test)
        t_diff_norm_m_xyz_vs_ply = float(np.linalg.norm(np.asarray(t_out) - np.asarray(t_ply_test)))

        save_json(out_dir / "step6_ply_nn_repnp_pose.json", {
            "pose_method_ply": pose_method_ply,
            "inlier_count_ply": int(inlier_count_ply),
            "reproj_err_ply": None if not np.isfinite(reproj_err_ply) else float(reproj_err_ply),
            "R_ply_test": np.asarray(R_ply_test, dtype=np.float64).tolist(),
            "t_ply_test": np.asarray(t_ply_test, dtype=np.float64).tolist(),
            "rot_diff_deg_xyz_vs_ply": rot_diff_deg_xyz_vs_ply,
            "t_diff_norm_m_xyz_vs_ply": t_diff_norm_m_xyz_vs_ply,
        })

        print(
            f"  [PLY-nearest rePnP] inliers={inlier_count_ply}/{len(pts2d_corr)}  "
            f"reproj={reproj_err_ply:.2f}px  "
            f"rot_diff_vs_xyz={rot_diff_deg_xyz_vs_ply:.3f}deg  "
            f"t_diff_vs_xyz={t_diff_norm_m_xyz_vs_ply:.4f}m"
        )

        draw_correspondence_debug_single_pose(
            query_img=query_img,
            pts2d_query=pts2d_corr,
            pts3d_obj=pts3d_ply_nn,
            K=K,
            R=R_ply_test,
            t=t_ply_test,
            out_path=out_dir / "step6_corr_query_ply_nn_repnp.png",
            draw_idx=None,
            max_draw=100,
            observed_color=(255, 100, 0),
            reproj_color=(0, 255, 255),
            line_color=(180, 180, 180),
            title_prefix="PLY-nearest rePnP on same query correspondences",
            stats_json_path=out_dir / "step6_corr_query_ply_nn_repnp_stats.json",
        )
    else:
        print("  [PLY-nearest rePnP] skipped (pts3d_ply_nn unavailable)")
               
    # ------------------------------------------------------------
    # after-PnP translation-only refinement (R fixed, t=[x,y,z] refined)
    # --skip_t_refine 플래그로 완전 비활성화 가능
    # IoU 기반 조건부 적용: best_score 가 iou_accept_thresh 미만이면
    #   refinement 결과를 버리고 PnP 원본 t 유지
    # ------------------------------------------------------------
    xyz_refine_info = {"status": "not_run"}
    skip_t_refine    = bool(getattr(args, "skip_t_refine", False))
    iou_accept_thresh = float(getattr(args, "t_refine_iou_thresh", 0.30))

    t_pnp = t_out.copy()   # PnP 결과 보존 (fallback용)

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

            # ── 조건부 적용 ─────────────────────────────────────────────────
            if best_iou >= iou_accept_thresh:
                t_out = t_refined
                xyz_refine_info["t_applied"] = True
                print(f"  [xyz refine] ✓ IoU={best_iou:.4f} >= {iou_accept_thresh:.2f} → t_refined 적용")
            else:
                t_out = t_pnp   # PnP 원본 t 유지
                xyz_refine_info["t_applied"] = False
                print(f"  [xyz refine] ✗ IoU={best_iou:.4f} < {iou_accept_thresh:.2f} → PnP t 유지 (t_refined 버림)")
        else:
            print("  [xyz refine] query reference mask unavailable, skipping.")
            xyz_refine_info["status"] = "skipped_no_query_mask"
    else:
        print("  [xyz refine] GS render args missing, skipping.")
        xyz_refine_info["status"] = "skipped_no_gs_args"

    # q_out은 translation refine 이후 계산
    q_out = rotation_matrix_to_quaternion(R_out)

    if len(inlier_idx) > 0:
        pts2d_inliers = pts2d_corr[inlier_idx]
        pts3d_inliers = pts3d_corr[inlier_idx]
    else:
        pts2d_inliers = np.empty((0, 2), dtype=np.float64)
        pts3d_inliers = np.empty((0, 3), dtype=np.float64)

    # ------------------------------------------------------------
    # Before-PnP visualization pose
    #   coarse bbox t_init 대신, 현재 2D-3D correspondence와
    #   R_gallery만으로 linear T를 구해서 디버그용 before pose로 사용
    # ------------------------------------------------------------
    t_before_vis = estimate_t_linear(pts2d_corr, pts3d_corr, K, R_gallery)

    pre_axes_path = out_dir / "step6_axes_before_pnp.png"
    project_axes_overlay(query_img, K, R_gallery, t_before_vis, axis_len, out_path=pre_axes_path)

    pre_ply_overlay_path = None
    if canonical_ply_path is not None:
        canonical_ply_path = Path(canonical_ply_path)
        if canonical_ply_path.exists():
            ply_xyz = load_ply_xyz(canonical_ply_path)
            pre_ply_overlay_path = out_dir / "step6_ply_overlay_before_pnp.png"
            _, _ = draw_ply_overlay(
                query_img=query_img,
                ply_xyz=ply_xyz,
                K=K, R=R_gallery, t=t_before_vis,
                max_points=8000,
                alpha=0.5,
                point_color=(255, 0, 255),
                point_radius=2,
                out_path=pre_ply_overlay_path,
            )
        else:
            print(f"  [Pre-PnP PLY overlay] canonical_ply_path not found: {canonical_ply_path}")

    # ------------------------------------------------------------
    # Numeric reprojection sanity check
    # ------------------------------------------------------------
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

    # ── 8. 시각화 ─────────────────────────────────────────────────────────────
    # axes_path = out_dir / "step6_axes_on_query.png"
    # project_axes_overlay(query_img, K, R_out, t_out, axis_len, out_path=axes_path)

    # (A) step6에서 실제 PnP에 넣은 전체 점
    points_used_all_path = out_dir / "step6_points_used_all.png"
    draw_points_only(
        query_img=query_img,
        pts2d=pts2d_corr,
        out_path=points_used_all_path,
        color=(255, 100, 0),
        radius=3,
        max_draw=None,
        label=f"step6 used points (all): {len(pts2d_corr)}",
    )

    # (B) PnP inlier만
    points_used_inliers_path = out_dir / "step6_points_used_pnp_inliers.png"

    # ── reprojection error 통계 ───────────────────────────────────────────────
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

    draw_points_only(
        query_img=query_img,
        pts2d=pts2d_inliers,
        out_path=points_used_inliers_path,
        color=(0, 255, 0),
        radius=3,
        max_draw=None,
        label=f"PnP inliers: {len(pts2d_inliers)}",
    )

    # (C) PnP inlier의 observed vs reprojection
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
        draw_points_only(
            query_img=query_img,
            pts2d=np.empty((0, 2), dtype=np.float64),
            out_path=reproj_debug_path,
            color=(255, 255, 255),
            radius=3,
            label="No PnP inliers",
        )
        save_json(reproj_debug_stats_path, {
            "title_prefix": "PnP inliers only (after pose)",
            "num_points_total": 0,
            "num_points_drawn": 0,
        })

    points_before_uniform_path = out_dir / "step6_points_before_uniform_sampling.png"
    draw_points_only(
        query_img=query_img,
        pts2d=pts2d_before_uniform,
        out_path=points_before_uniform_path,
        color=(255, 100, 0),
        radius=3,
        max_draw=None,
        label=f"before uniform sampling: {len(pts2d_before_uniform)}",
    )

    points_before_after_uniform_path = out_dir / "step6_points_before_after_uniform_sampling.png"
    draw_points_before_after(
        query_img=query_img,
        pts_before=pts2d_before_uniform,
        pts_after=pts2d_corr,
        out_path=points_before_after_uniform_path,
        color_before=(255, 100, 0),
        color_after=(0, 255, 0),
        radius_before=2,
        radius_after=2,
        label_before=f"before ({len(pts2d_before_uniform)})",
        label_after=f"after uniform ({len(pts2d_corr)})",
    )

    # ------------------------------------------------------------
    # Reprojection debug: 반드시 같은 correspondence 집합으로 비교
    # ------------------------------------------------------------
    if len(inlier_idx) > 0:
        pts2d_dbg = pts2d_corr[inlier_idx]
        pts3d_dbg = pts3d_corr[inlier_idx]
        dbg_name = "pnp_inliers"
    else:
        pts2d_dbg = pts2d_corr
        pts3d_dbg = pts3d_corr
        dbg_name = "all_corr_fallback"

    # before / after / compare 모두 같은 draw_idx를 공유
    N_dbg = min(len(pts2d_dbg), 120)
    if N_dbg > 0:
        rng_dbg = np.random.default_rng(42)
        dbg_idx = rng_dbg.choice(len(pts2d_dbg), size=N_dbg, replace=False)
    else:
        dbg_idx = np.array([], dtype=np.int32)

    # (2) after only
    reproj_after_pnp_path = out_dir / "step6_reproj_debug_after_pnp.png"
    reproj_after_stats_path = out_dir / "step6_reproj_debug_after_pnp_stats.json"
    draw_correspondence_debug_single_pose(
        query_img=query_img,
        pts2d_query=pts2d_dbg,
        pts3d_obj=pts3d_dbg,
        K=K,
        R=R_out,
        t=t_out,
        out_path=reproj_after_pnp_path,
        draw_idx=dbg_idx,
        title_prefix=f"after PnP ({dbg_name})",
        stats_json_path=reproj_after_stats_path,
    )

    print(f"  [Reproj Debug] using {len(pts2d_dbg)} correspondences ({dbg_name})")
    print(f"  [Reproj Debug] draw subset size: {len(dbg_idx)}")

    # ------------------------------------------------------------
    # Reprojection debug: same correspondences before vs after
    # ------------------------------------------------------------
    reproj_compare_path = out_dir / "step6_reproj_debug_before_after_same_corr.png"
    reproj_compare_stats_path = out_dir / "step6_reproj_debug_before_after_same_corr_stats.json"

    draw_correspondence_debug_before_after(
        query_img=query_img,
        pts2d_query=pts2d_dbg,
        pts3d_obj=pts3d_dbg,
        K=K,
        R_before=R_gallery,
        t_before=t_before_vis,
        R_after=R_out,
        t_after=t_out,
        out_path=reproj_compare_path,
        draw_idx=dbg_idx,
        title_prefix=f"same correspondences before/after ({dbg_name})",
        stats_json_path=reproj_compare_stats_path,
    )

    # ── PnP 후 pose 시각화 ─────────────────────────────────────────────────────
    post_axes_path = out_dir / "step6_axes_after_pnp.png"
    project_axes_overlay(query_img, K, R_out, t_out, axis_len, out_path=post_axes_path)

    post_ply_overlay_path = None
    if canonical_ply_path is not None:
        canonical_ply_path = Path(canonical_ply_path)
        if canonical_ply_path.exists():
            ply_xyz = load_ply_xyz(canonical_ply_path)
            post_ply_overlay_path = out_dir / "step6_ply_overlay_after_pnp.png"
            _, post_ply_kept = draw_ply_overlay(
                query_img=query_img,
                ply_xyz=ply_xyz,
                K=K, R=R_out, t=t_out,
                max_points=8000,
                alpha=0.5,
                point_color=(0, 255, 0),
                point_radius=2,
                out_path=post_ply_overlay_path,
            )
        else:
            print(f"  [Post-PnP PLY overlay] canonical_ply_path not found: {canonical_ply_path}")

        # ── PnP 후 3DGS render + overlay ────────────────────────────────────────
    post_gs_render_path = None
    post_gs_overlay_path = None
    post_gs_xyz_path = None   # R_out+t_out pose의 XYZ map

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

    # ------------------------------------------------------------
    # Fixed xyz_obj -> projection on post-render RGB / XYZ
    # ------------------------------------------------------------
    try:
        if post_gs_render_path is not None and post_gs_xyz_path is not None \
           and Path(post_gs_render_path).exists() and Path(post_gs_xyz_path).exists():

            post_render_img_fixed = load_image(post_gs_render_path)
            post_xyz_map_fixed = np.load(str(post_gs_xyz_path)).astype(np.float64)

            fixed_xyzobj_proj_trace = build_fixed_xyzobj_projection_trace(
                pts3d_obj=pts3d_corr,   # 고정된 xyz_obj
                K=K,
                R_pose=R_out,
                t_pose=t_out,
                post_xyz_map=post_xyz_map_fixed,
            )

            save_json(out_dir / "step6_fixed_xyzobj_projection_trace.json", fixed_xyzobj_proj_trace)
            save_fixed_xyzobj_projection_csv(
                out_dir / "step6_fixed_xyzobj_projection_trace.csv",
                fixed_xyzobj_proj_trace,
            )

            draw_fixed_xyzobj_projection_on_postrender(
                post_render_img=post_render_img_fixed,
                trace=fixed_xyzobj_proj_trace,
                out_path=out_dir / "step6_fixed_xyzobj_projection_on_postrender.png",
                topk=20,
                select_key="xyz_err_mm",
                title="Fixed xyz_obj projected on post-render",
            )

            ss = fixed_xyzobj_proj_trace["summary"]
            if len(ss) > 0:
                print(
                    f"  [Fixed xyz_obj projection] "
                    f"valid={ss['num_post_xyz_valid']}/{fixed_xyzobj_proj_trace['num_points']}  "
                    f"mean_xyz_err={None if ss['mean_xyz_err_m'] is None else ss['mean_xyz_err_m']*1000:.2f}mm"
                )
            print("  [Fixed xyz_obj projection] saved: step6_fixed_xyzobj_projection_trace.json/csv")
            print("  [Fixed xyz_obj projection] saved: step6_fixed_xyzobj_projection_on_postrender.png")

        else:
            print("  [Fixed xyz_obj projection] skipped (post-render RGB/XYZ not found)")
    except Exception as e:
        print(f"  [Fixed xyz_obj projection] failed: {e}")
    # ------------------------------------------------------------
    # Same-inlier motion debug on top of query + render-region check
    # ------------------------------------------------------------
    if len(pts2d_inliers) > 0:
        N_move = min(len(pts2d_inliers), 120)
        rng_move = np.random.default_rng(42)
        move_idx = rng_move.choice(len(pts2d_inliers), size=N_move, replace=False)

        before_region_mask = None
        after_region_mask = None

        # after GS render region
        if post_gs_render_path is not None and Path(post_gs_render_path).exists():
            post_render_img = load_image(post_gs_render_path)
            after_region_mask = render_to_binary_mask(post_render_img, nonblack_thresh=8)

    else:
        print("  [Inlier motion debug] No PnP inliers, skipping motion visualization.")

    inlier_motion_path = out_dir / "step6_inlier_motion_before_after_on_query.png"
    inlier_motion_stats_path = out_dir / "step6_inlier_motion_before_after_on_query_stats.json"

    draw_inlier_motion_with_region_check(
        query_img=query_img,
        pts2d_obs=pts2d_inliers,
        pts3d_obj=pts3d_inliers,
        K=K,
        R_before=R_gallery,
        t_before=t_before_vis,
        R_after=R_out,
        t_after=t_out,
        out_path=inlier_motion_path,
        draw_idx=move_idx,
        max_draw=120,
        region_mask_before=before_region_mask if 'before_region_mask' in locals() else None,
        region_mask_after=after_region_mask,
        title_prefix="same inliers: before -> after",
        stats_json_path=inlier_motion_stats_path,
    )

    # ------------------------------------------------------------
    # Detect problematic after-PnP inliers (outside render region)
    # ------------------------------------------------------------
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
            # local inlier index -> global corr index
            bad_corr_global_idx = inlier_idx[bad_inlier_local_idx].astype(np.int32)

        print(f"  [Prune candidates] outside-after-render inliers: {len(bad_inlier_local_idx)}")
        if len(bad_inlier_local_idx) > 0:
            print(f"  [Prune candidates] local inlier idx: {bad_inlier_local_idx.tolist()}")
            print(f"  [Prune candidates] global corr idx : {bad_corr_global_idx.tolist()}")

        save_json(out_dir / "step6_bad_after_render_inliers.json", {
            "bad_inlier_local_idx": bad_inlier_local_idx.tolist(),
            "bad_corr_global_idx": bad_corr_global_idx.tolist(),
            "num_bad_after_render": int(len(bad_inlier_local_idx)),
        })
    else:
        print("  [Prune candidates] skipped (no inliers or no post render)")

    # ------------------------------------------------------------
    # Pruned correspondence sets for visualization
    #   - before side: remove the same global correspondences from pts2d_corr/pts3d_corr
    #   - after side : remove the problematic local inliers from pts2d_inliers/pts3d_inliers
    # ------------------------------------------------------------
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
    
    # ------------------------------------------------------------
    # After-PnP: query_masked <-> rendered correspondence visualization
    #   사용 점 = PnP가 채택한 최종 inliers (pts2d_inliers / pts3d_inliers)
    # ------------------------------------------------------------
    if post_gs_render_path is not None and Path(post_gs_render_path).exists():
        post_render_img = load_image(post_gs_render_path)

        # query_img (3840x2160 full) + pts2d_inliers (full image 좌표) 그대로 사용
        pts2d_query_masked_after = pts2d_inliers  # full image 좌표

        pts2d_render_after = project_points_obj_to_img(
            pts3d_inliers,
            K=K,
            R=R_out,
            t=t_out,
        )

        N_vis_after = min(len(pts2d_inliers), 200)
        rng_vis_after = np.random.default_rng(42)
        vis_idx_after = rng_vis_after.choice(len(pts2d_inliers), size=N_vis_after, replace=False) \
            if len(pts2d_inliers) > 0 else np.array([], dtype=np.int32)

        after_corr_vis_path = out_dir / "step6_query_render_corr_after_pnp.png"
        after_corr_vis_stats = out_dir / "step6_query_render_corr_after_pnp_stats.json"

        draw_query_render_correspondence(
            query_img=query_img,
            render_img=post_render_img,
            pts_query=pts2d_query_masked_after,
            pts_render=pts2d_render_after,
            out_path=after_corr_vis_path,
            draw_idx=vis_idx_after,
            title_prefix=f"after PnP: PnP inliers only (N={len(pts2d_inliers)})",
            stats_json_path=after_corr_vis_stats,
        )
    else:
        print("  [WARN] Skipping after-PnP query/render correspondence visualization.")


    # (4) post GS render (R_out + t_out pose로 렌더) 위에
    #     R_out+t_out pose의 XYZ map에서 pts_g_full로 새로 lookup → 재투영
    #     → 점이 캔에 잘 올라오면 PnP 결과가 정확하다는 의미
    if post_gs_render_path is not None and Path(post_gs_render_path).exists():
        post_render_img = load_image(post_gs_render_path)

        if post_gs_xyz_path is not None and Path(post_gs_xyz_path).exists():
            # R_out+t_out pose의 XYZ map에서 pts_g_full로 새로 lookup
            post_xyz_map = np.load(str(post_gs_xyz_path)).astype(np.float64)
            post_pts3d, post_valid = lookup_xyz_at_pixels(post_xyz_map, pts_g_full, bilinear=True)
            post_pts3d_valid = post_pts3d[post_valid]

            if len(post_pts3d_valid) > 0:
                pts2d_render_reproj_after = project_points_obj_to_img(
                    post_pts3d_valid, K, R_out, t_out,
                )
                draw_points_on_render(
                    render_img=post_render_img,
                    pts2d=pts2d_render_reproj_after,
                    out_path=out_dir / "step6_render_points_reprojected_after.png",
                    color=(0, 255, 0),
                    radius=6,
                    label=f"Reprojected from R_out+t_out XYZ map (N={len(post_pts3d_valid)})",
                )
                print(f"  [Debug] PnP XYZ map reproj → step6_render_points_reprojected_after.png  (N={len(post_pts3d_valid)})")
        else:
            # XYZ map 없으면 기존 pts3d_inliers로 fallback
            pts3d_for_reproj = pts3d_inliers if len(pts3d_inliers) > 0 else pts3d_corr
            pts2d_render_reproj_after = project_points_obj_to_img(
                pts3d_for_reproj, K, R_out, t_out,
            )
            draw_points_on_render(
                render_img=post_render_img,
                pts2d=pts2d_render_reproj_after,
                out_path=out_dir / "step6_render_points_reprojected_after.png",
                color=(255, 165, 0),   # 주황 = XYZ map 없어서 부정확
                radius=6,
                label=f"[fallback] Reprojected pts3d_inliers with PnP pose (N={len(pts3d_for_reproj)})",
            )
            print("  [WARN] post XYZ map not found, using gallery pts3d_inliers as fallback")

        # ------------------------------------------------------------
    # Same-corr point_id별 post-render same-surface check
    # ------------------------------------------------------------
    try:
        if 'same_corr_trace' in locals() and post_gs_xyz_path is not None and Path(post_gs_xyz_path).exists():
            post_xyz_map_samecorr = np.load(str(post_gs_xyz_path)).astype(np.float64)
            post_render_img_samecorr = load_image(post_gs_render_path)

            same_corr_surface_trace = augment_same_corr_trace_with_postrender_surface(
                trace=same_corr_trace,
                post_xyz_map=post_xyz_map_samecorr,
                xyz_err_bad_thresh_m=0.005,
            )

            save_json(out_dir / "step6_same_corr_postrender_surface_trace.json", same_corr_surface_trace)
            save_same_corr_surface_csv(out_dir / "step6_same_corr_postrender_surface_trace.csv", same_corr_surface_trace)

            draw_same_corr_topk_surface_err_postrender(
                post_render_img=post_render_img_samecorr,
                trace_aug=same_corr_surface_trace,
                out_path=out_dir / "step6_same_corr_top12_surface_err_postrender.png",
                topk=12,
            )

            ss = same_corr_surface_trace["summary"]
            print(
                f"  [Same corr post-render surface] "
                f"valid={ss['num_post_xyz_valid']}/{same_corr_surface_trace['num_points']}  "
                f"bad={ss['num_surface_bad']}  "
                f"mean_surface_err={None if ss['mean_surface_err_m'] is None else ss['mean_surface_err_m']*1000:.2f}mm"
            )
            print("  [Same corr post-render surface] saved: step6_same_corr_postrender_surface_trace.json/csv")
            print("  [Same corr post-render surface] saved: step6_same_corr_top12_surface_err_postrender.png")
        else:
            print("  [Same corr post-render surface] skipped (same_corr_trace / post XYZ unavailable)")
    except Exception as e:
        print(f"  [Same corr post-render surface] failed: {e}")
    # ------------------------------------------------------------
    # Same-corr post-render local neighborhood best-match check
    # ------------------------------------------------------------
    try:
        if 'same_corr_surface_trace' in locals() and post_gs_xyz_path is not None and Path(post_gs_xyz_path).exists():
            post_xyz_map_localnn = np.load(str(post_gs_xyz_path)).astype(np.float64)

            same_corr_localnn_trace = augment_same_corr_trace_with_postrender_localnn(
                trace_aug_center=same_corr_surface_trace,
                post_xyz_map=post_xyz_map_localnn,
                radius=4,                 # 9x9 window
                best_err_bad_thresh_m=0.005,
            )

            save_json(out_dir / "step6_same_corr_postrender_localnn_trace.json", same_corr_localnn_trace)
            save_same_corr_localnn_csv(out_dir / "step6_same_corr_postrender_localnn_trace.csv", same_corr_localnn_trace)

            draw_same_corr_top12_localnn_postrender(
                post_render_img=post_render_img_samecorr,
                trace_localnn=same_corr_localnn_trace,
                out_path=out_dir / "step6_same_corr_top12_localnn_postrender.png",
                topk=12,
            )

            ss = same_corr_localnn_trace["summary"]
            print(
                f"  [Same corr localNN] "
                f"best_valid={ss['num_best_valid']}/{same_corr_localnn_trace['num_points']}  "
                f"best_bad={ss['num_best_surface_bad']}  "
                f"mean_best_err={None if ss['mean_best_err_m'] is None else ss['mean_best_err_m']*1000:.2f}mm  "
                f"mean_center_to_best_improve={ss['mean_center_to_best_improvement_mm']:.2f}mm"
            )
            print("  [Same corr localNN] saved: step6_same_corr_postrender_localnn_trace.json/csv")
            print("  [Same corr localNN] saved: step6_same_corr_top12_localnn_postrender.png")
        else:
            print("  [Same corr localNN] skipped (surface trace / post XYZ unavailable)")
    except Exception as e:
        print(f"  [Same corr localNN] failed: {e}")

    # ── 9. 저장 ───────────────────────────────────────────────────────────────
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
        "debug_points_used_all": str(points_used_all_path),
        "debug_points_used_pnp_inliers": str(points_used_inliers_path),
        "debug_reproj_inliers": str(reproj_debug_path),
        "debug_axes_after_pnp": str(post_axes_path),
        "debug_ply_overlay_before_pnp": str(pre_ply_overlay_path) if pre_ply_overlay_path else None,
        "debug_ply_overlay_after_pnp": str(post_ply_overlay_path) if post_ply_overlay_path else None,
        "n_before_uniform_sampling": int(len(pts2d_before_uniform)),
        "n_after_uniform_sampling": int(len(pts2d_corr)),
        "uniform_sampling_cell_stats": uniform_cell_stats,
        "debug_points_before_uniform_sampling": str(points_before_uniform_path),
        "debug_points_before_after_uniform_sampling": str(points_before_after_uniform_path),
        "debug_gs_render_after_pnp": str(post_gs_render_path) if post_gs_render_path else None,
        "debug_gs_render_overlay_after_pnp": str(post_gs_overlay_path) if post_gs_overlay_path else None,
        "debug_axes_before_pnp": str(pre_axes_path) if pre_axes_path else None,
        "debug_ply_overlay_before_pnp": str(pre_ply_overlay_path) if pre_ply_overlay_path else None,
        "debug_reproj_before_after_same_corr": str(reproj_compare_path) if reproj_compare_path else None,
        "outputs": {
            # "axes_on_query":        str(axes_path),
            "ply_overlay":          str(post_ply_overlay_path) if post_ply_overlay_path else None,
        },
    }

    # initial_pose.json과 동일한 필드명으로도 저장 (step6 호환)
    pose_out["query_img"] = str(args.query_img)
    pose_out["query_masked_path"] = str(getattr(args, "query_masked_path", ""))
    pose_out["intrinsics_path"] = str(args.intrinsics_path)

    save_json(out_dir / "step6_pose.json", pose_out)
    # step5/step6이 읽는 initial_pose.json에도 덮어쓰기
    save_json(out_dir / "initial_pose.json", pose_out)

    print("=" * 60)
    print("[Step 6] PnP translation estimation complete")
    print(f"  best_render     : {best_render}")
    print(f"  n_corr          : {n_valid}")
    print(f"  reproj_err      : {reproj_err:.2f}px" if np.isfinite(reproj_err) else "  reproj_err      : N/A")
    print(f"  R_obj_to_cam    :\n{R_out}")
    print(f"  t_obj_to_cam    : [{t_out[0]:.4f}, {t_out[1]:.4f}, {t_out[2]:.4f}]")
    # print(f"  axes_vis        : {axes_path}")
    # if post_ply_overlay_path:
    #     print(f"  ply_overlay     : {post_ply_overlay_path}  (kept={ply_kept})")
    print(f"  initial_pose.json: {out_dir / 'initial_pose.json'}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# helpers (re-exported for other modules)
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
    return np.asarray(pts_crop, dtype=np.float64) + np.array([[x1, y1]])