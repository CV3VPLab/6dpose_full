
import json
from pathlib import Path
import math

import cv2
import numpy as np
from plyfile import PlyData


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


def load_image(path, color=True):
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


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
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported intrinsics format: {path}")
    return K, fx, fy, cx, cy


def square_pad_resize(img_bgr: np.ndarray, size: int = 320) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0+h, x0:x0+w] = img_bgr
    out = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return out


def compute_nonblack_bbox(img_bgr: np.ndarray, thresh: int = 8):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > thresh)
    h, w = gray.shape
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, w, h
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return int(x1), int(y1), int(x2), int(y2)


def crop_with_bbox(img_bgr, bbox):
    x1, y1, x2, y2 = bbox
    return img_bgr[y1:y2, x1:x2].copy()


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


def rotation_matrix_to_euler_xyz_deg(R):
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x, y, z])

def load_ply_xyz(ply_path):
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    return xyz


def sample_points_for_projection(xyz, max_points=4000, seed=0):
    xyz = np.asarray(xyz, dtype=np.float32)
    if len(xyz) <= max_points:
        return xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xyz), size=max_points, replace=False)
    return xyz[idx]


def project_points_cv(obj_pts_3d, K, rvec, tvec):
    obj_pts_3d = np.asarray(obj_pts_3d, dtype=np.float32)
    dist = np.zeros((4, 1), dtype=np.float64)
    imgpts, _ = cv2.projectPoints(obj_pts_3d, rvec, tvec, K.astype(np.float64), dist)
    return imgpts.reshape(-1, 2)


def draw_projected_ply_overlay(
    img_bgr,
    ply_xyz,
    K,
    R,
    t,
    max_points=5000,
    alpha=0.45,
    point_color=(0, 255, 255),
    point_radius=1,
):
    base = img_bgr.copy()
    overlay = img_bgr.copy()

    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec = t.reshape(3, 1).astype(np.float64)

    xyz_sampled = sample_points_for_projection(ply_xyz, max_points=max_points, seed=0)
    imgpts = project_points_cv(xyz_sampled, K, rvec, tvec)

    h, w = overlay.shape[:2]
    kept = 0
    for p in imgpts:
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(overlay, (x, y), point_radius, point_color, -1, cv2.LINE_AA)
            kept += 1

    vis = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)
    return vis, kept


def estimate_axis_endpoints_from_xyz(xyz):
    xyz = np.asarray(xyz, dtype=np.float32)

    zmin = float(xyz[:, 2].min())
    zmax = float(xyz[:, 2].max())
    zspan = max(1e-6, zmax - zmin)
    band = 0.05 * zspan

    bot = xyz[xyz[:, 2] <= zmin + band]
    top = xyz[xyz[:, 2] >= zmax - band]

    if len(bot) == 0:
        bot = xyz[np.argsort(xyz[:, 2])[:100]]
    if len(top) == 0:
        top = xyz[np.argsort(-xyz[:, 2])[:100]]

    bot_center = bot.mean(axis=0)
    top_center = top.mean(axis=0)
    return bot_center.astype(np.float32), top_center.astype(np.float32)


def draw_projected_axis_line(img_bgr, K, R, t, p0_obj, p1_obj, color=(255, 255, 0), thickness=3):
    vis = img_bgr.copy()

    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec = t.reshape(3, 1).astype(np.float64)
    obj_pts = np.asarray([p0_obj, p1_obj], dtype=np.float32)

    imgpts = project_points_cv(obj_pts, K, rvec, tvec)
    imgpts = np.round(imgpts).astype(int)

    p0 = tuple(imgpts[0])
    p1 = tuple(imgpts[1])

    cv2.line(vis, p0, p1, color, thickness, cv2.LINE_AA)
    cv2.circle(vis, p0, 6, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(vis, p1, 6, (0, 0, 255), -1, cv2.LINE_AA)
    return vis

def compute_projected_bbox_from_points(obj_xyz, K, R, t, img_shape_hw, max_points=4000):
    """
    canonical object points를 현재 pose로 투영해서 2D bbox 계산
    """
    xyz = sample_points_for_projection(obj_xyz, max_points=max_points, seed=0)

    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec = t.reshape(3, 1).astype(np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    imgpts, _ = cv2.projectPoints(xyz.astype(np.float32), rvec, tvec, K.astype(np.float64), dist)
    imgpts = imgpts.reshape(-1, 2)

    h, w = img_shape_hw[:2]
    keep = []
    for p in imgpts:
        x, y = float(p[0]), float(p[1])
        if np.isfinite(x) and np.isfinite(y):
            keep.append([x, y])

    if len(keep) == 0:
        return None

    keep = np.asarray(keep, dtype=np.float32)
    xs = keep[:, 0]
    ys = keep[:, 1]

    x1 = max(0.0, float(xs.min()))
    y1 = max(0.0, float(ys.min()))
    x2 = min(float(w - 1), float(xs.max()))
    y2 = min(float(h - 1), float(ys.max()))

    if x2 <= x1 or y2 <= y1:
        return None

    return np.array([x1, y1, x2, y2], dtype=np.float64)


def bbox_center_and_size(bbox_xyxy):
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return cx, cy, w, h


def refine_translation_with_projected_bbox(
    obj_xyz,
    query_bbox_xyxy,
    K,
    R,
    t_init,
    img_shape_hw,
    max_iters=12,
    max_points=4000,
    damping_xy=0.7,
    damping_z=0.5,
):
    """
    R은 고정하고 t만 보정.
    목표:
      - projected bbox center ~= query bbox center
      - projected bbox height ~= query bbox height
    """
    t = np.asarray(t_init, dtype=np.float64).reshape(3).copy()

    fx = float(K[0, 0])
    fy = float(K[1, 1])

    qcx, qcy, qw, qh = bbox_center_and_size(query_bbox_xyxy)

    history = []

    for it in range(max_iters):
        proj_bbox = compute_projected_bbox_from_points(
            obj_xyz=obj_xyz,
            K=K,
            R=R,
            t=t,
            img_shape_hw=img_shape_hw,
            max_points=max_points,
        )

        if proj_bbox is None:
            history.append({
                "iter": int(it),
                "status": "proj_bbox_none",
                "t": t.tolist(),
            })
            break

        pcx, pcy, pw, ph = bbox_center_and_size(proj_bbox)

        # center error in pixels
        du = qcx - pcx
        dv = qcy - pcy

        # height ratio
        scale_h = qh / max(ph, 1e-6)

        # 현재 depth 기준으로 x,y 평행이동 업데이트
        # image plane 오차를 camera x,y 오차로 근사 환산
        tz_new = t[2] * (1.0 + damping_z * (scale_h - 1.0))

        # depth가 바뀌면 x,y 환산도 새 depth로 하는 편이 조금 안정적
        tx_new = t[0] + damping_xy * (du * tz_new / fx)
        ty_new = t[1] + damping_xy * (dv * tz_new / fy)

        history.append({
            "iter": int(it),
            "proj_bbox": proj_bbox.tolist(),
            "proj_center": [float(pcx), float(pcy)],
            "proj_size": [float(pw), float(ph)],
            "query_center": [float(qcx), float(qcy)],
            "query_size": [float(qw), float(qh)],
            "delta_uv": [float(du), float(dv)],
            "scale_h": float(scale_h),
            "t_before": t.tolist(),
            "t_after": [float(tx_new), float(ty_new), float(tz_new)],
        })

        t[:] = [tx_new, ty_new, tz_new]

    return t.astype(np.float64), history


def draw_bbox(img_bgr, bbox_xyxy, color=(0, 255, 255), thickness=2, label=None):
    vis = img_bgr.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if label:
        cv2.putText(vis, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return vis


def draw_query_and_projected_bbox(img_bgr, query_bbox_xyxy, proj_bbox_xyxy, out_path):
    vis = img_bgr.copy()
    vis = draw_bbox(vis, query_bbox_xyxy, color=(0, 255, 0), thickness=3, label="query_bbox")
    if proj_bbox_xyxy is not None:
        vis = draw_bbox(vis, proj_bbox_xyxy, color=(0, 255, 255), thickness=3, label="projected_bbox")
    cv2.imwrite(str(out_path), vis)

def project_axes_overlay(query_img, K, R, t, axis_len_m, out_path):
    img = query_img.copy()
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
    x = tuple(imgpts[1])
    y = tuple(imgpts[2])
    z = tuple(imgpts[3])

    cv2.line(img, o, x, (0, 0, 255), 4, cv2.LINE_AA)   # X red
    cv2.line(img, o, y, (0, 255, 0), 4, cv2.LINE_AA)   # Y green
    cv2.line(img, o, z, (255, 0, 0), 4, cv2.LINE_AA)   # Z blue
    cv2.circle(img, o, 6, (255, 255, 255), -1, cv2.LINE_AA)

    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), img)
    return img


def make_summary_panel(width, height, lines, title="Initial Pose"):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    y = 85
    for line in lines:
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 2, cv2.LINE_AA)
        y += 34
    return img


def make_query_vs_best_image(query_crop, best_crop, out_size=320):
    q = square_pad_resize(query_crop, out_size)
    b = square_pad_resize(best_crop, out_size)
    canvas = np.zeros((out_size, out_size * 2, 3), dtype=np.uint8)
    canvas[:, :out_size] = q
    canvas[:, out_size:] = b

    cv2.putText(canvas, "query", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "best", (out_size + 12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def extract_best_render_info(args):
    # prefer reranked result
    rerank_json = Path(args.out_dir) / "loftr_scores.json"
    if rerank_json.exists():
        data = load_json(rerank_json)
        return {
            "source": "step4_rerank",
            "best_render": data["best_render"],
            "score": data["best_loftr_score"],
            "full": data,
        }

    dino_json = Path(args.out_dir) / "retrieval_scores.json"
    if dino_json.exists():
        data = load_json(dino_json)
        return {
            "source": "step4_dino",
            "best_render": data["best_render"],
            "score": data["best_score"],
            "full": data,
        }

    raise FileNotFoundError("Neither loftr_scores.json nor retrieval_scores.json found in out_dir.")


def find_pose_for_render(gallery_json, render_name):
    idx = int(Path(render_name).stem)
    for pose in gallery_json["poses"]:
        if int(pose["index"]) == idx:
            return pose
    raise KeyError(f"Pose for render {render_name} not found in gallery_poses.json")


def estimate_translation_from_bbox(bbox_xyxy, fx, fy, cx, cy, object_height_m):
    x1, y1, x2, y2 = bbox_xyxy
    bbox_h = max(1.0, float(y2 - y1))
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)

    tz = (fy * object_height_m) / bbox_h
    tx = ((u - cx) * tz) / fx
    ty = ((v - cy) * tz) / fy
    return np.array([tx, ty, tz], dtype=np.float64), {
        "bbox_h_px": bbox_h,
        "bbox_center_uv": [float(u), float(v)],
        "estimated_tz_from_height": float(tz),
    }


def build_pose_visualization_v2(query_img, query_crop, best_render, axes_overlay,
                                loftr_match_img, summary_lines, out_path):
    W = 960
    H = 720
    GAP = 20

    def pad_to_panel(img, w, h, pad=16):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        inner_w = w - 2 * pad
        inner_h = h - 2 * pad
        resized = cv2.resize(img, (inner_w, inner_h), interpolation=cv2.INTER_AREA)
        canvas[pad:pad+inner_h, pad:pad+inner_w] = resized
        return canvas

    def hstack_with_gap(a, b, gap=20):
        spacer = np.zeros((a.shape[0], gap, 3), dtype=np.uint8)
        return np.concatenate([a, spacer, b], axis=1)

    def make_summary_panel_loose(width, height, lines, title="Pose Result"):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(img, title, (24, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 2, cv2.LINE_AA)
        y = 105
        for line in lines:
            cv2.putText(img, line, (24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (230, 230, 230), 2, cv2.LINE_AA)
            y += 44
        return img

    # Row 1
    row1_l = pad_to_panel(query_img, W, H, pad=12)
    row1_r = make_summary_panel_loose(W, H, summary_lines, title="Pose Result")
    row1 = hstack_with_gap(row1_l, row1_r, GAP)

    # Row 2
    q = square_pad_resize(query_crop, H - 32)
    b = square_pad_resize(best_render, H - 32)

    row2_l = np.zeros((H, W, 3), dtype=np.uint8)
    row2_r = np.zeros((H, W, 3), dtype=np.uint8)

    qh, qw = q.shape[:2]
    bh, bw = b.shape[:2]

    qx = (W - qw) // 2
    qy = (H - qh) // 2
    bx = (W - bw) // 2
    by = (H - bh) // 2

    row2_l[qy:qy+qh, qx:qx+qw] = q
    row2_r[by:by+bh, bx:bx+bw] = b

    cv2.putText(row2_l, "query", (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(row2_r, "best", (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    row2 = hstack_with_gap(row2_l, row2_r, GAP)

    # Row 3
    row3_l = pad_to_panel(axes_overlay, W, H, pad=12)

    if loftr_match_img is None:
        row3_r = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.putText(row3_r, "No LoFTR match image", (40, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    else:
        row3_r = pad_to_panel(loftr_match_img, W, H, pad=12)

    row3 = hstack_with_gap(row3_l, row3_r, GAP)

    # Vertical gaps
    vgap = np.zeros((GAP, row1.shape[1], 3), dtype=np.uint8)
    canvas = np.concatenate([row1, vgap, row2, vgap, row3], axis=0)

    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), canvas)


def run_step5_initial_pose(args):
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    best_info = extract_best_render_info(args)
    gallery = load_json(args.gallery_pose_json)
    pose = find_pose_for_render(gallery, best_info["best_render"])

    R = np.array(pose["R_obj_to_cam"], dtype=np.float64)
    K, fx, fy, cx, cy = load_intrinsics(args.intrinsics_path)

    step1_json_path = Path(args.step1_json)
    if not step1_json_path.exists():
        raise FileNotFoundError(f"step1 json not found: {step1_json_path}")
    step1 = load_json(step1_json_path)

    bbox = step1.get("mask_bbox_xyxy", None) or step1.get("bbox_xyxy", None)
    if bbox is None:
        raise KeyError("step1_result.json has neither mask_bbox_xyxy nor bbox_xyxy")

    t, trans_meta = estimate_translation_from_bbox(
        bbox_xyxy=bbox,
        fx=fx, fy=fy, cx=cx, cy=cy,
        object_height_m=float(args.object_height_m),
    )

    q = rotation_matrix_to_quaternion(R)
    euler_deg = rotation_matrix_to_euler_xyz_deg(R)

    query_img = load_image(args.query_img)
    query_masked = load_image(args.query_masked_path)
    best_render = load_image(str(Path(args.gallery_dir) / best_info["best_render"]))

    canonical_ply_path = getattr(args, "canonical_ply_path", None)
    canonical_xyz = None
    if canonical_ply_path is not None:
        canonical_ply_path = Path(canonical_ply_path)
        if canonical_ply_path.exists():
            canonical_xyz = load_ply_xyz(canonical_ply_path)
        else:
            print(f"[Step5 Viz] canonical_ply_path not found: {canonical_ply_path}")

    t_init = t.copy()
    refinement_history = []
    refined_proj_bbox = None

    if canonical_xyz is not None:
        t_refined, refinement_history = refine_translation_with_projected_bbox(
            obj_xyz=canonical_xyz,
            query_bbox_xyxy=bbox,
            K=K,
            R=R,
            t_init=t_init,
            img_shape_hw=query_img.shape[:2],
            max_iters=12,
            max_points=4000,
            damping_xy=0.7,
            damping_z=0.5,
        )
        t = t_refined

        refined_proj_bbox = compute_projected_bbox_from_points(
            obj_xyz=canonical_xyz,
            K=K,
            R=R,
            t=t,
            img_shape_hw=query_img.shape[:2],
            max_points=4000,
        )

        bbox_compare_path = out_dir / "step5_query_vs_projected_bbox.png"
        draw_query_and_projected_bbox(
            img_bgr=query_img,
            query_bbox_xyxy=bbox,
            proj_bbox_xyxy=refined_proj_bbox,
            out_path=bbox_compare_path,
        )

        save_json(out_dir / "step5_translation_refinement.json", {
            "t_init": t_init.tolist(),
            "t_refined": t.tolist(),
            "query_bbox_xyxy": [float(v) for v in bbox],
            "projected_bbox_xyxy_final": None if refined_proj_bbox is None else refined_proj_bbox.tolist(),
            "history": refinement_history,
        })

        print(f"[Step5] t_init    : {t_init.tolist()}")
        print(f"[Step5] t_refined : {t.tolist()}")
        print(f"[Step5] bbox compare saved: {bbox_compare_path}")

    axes_overlay_path = out_dir / "step5_initial_axes_refined_on_query.png"
    axes_overlay = project_axes_overlay(
        query_img=query_img,
        K=K,
        R=R,
        t=t,
        axis_len_m=float(args.axis_len_m),
        out_path=axes_overlay_path,
    )

    ply_overlay_path = None
    canonical_axis_path = None
    kept_proj = None

    if getattr(args, "canonical_ply_path", None):
        canonical_ply_path = Path(args.canonical_ply_path)
        if canonical_ply_path.exists():
            ply_xyz = load_ply_xyz(canonical_ply_path)

            ply_overlay_img, kept_proj = draw_projected_ply_overlay(
                img_bgr=query_img,
                ply_xyz=ply_xyz,
                K=K,
                R=R,
                t=t,
                max_points=5000,
                alpha=0.45,
                point_color=(0, 255, 255),
                point_radius=1,
            )
            ply_overlay_path = out_dir / "step5_initial_ply_overlay_refined_on_query.png"
            cv2.imwrite(str(ply_overlay_path), ply_overlay_img)

            bot_center, top_center = estimate_axis_endpoints_from_xyz(ply_xyz)
            canonical_axis_img = draw_projected_axis_line(
                img_bgr=query_img,
                K=K,
                R=R,
                t=t,
                p0_obj=bot_center,
                p1_obj=top_center,
                color=(255, 255, 0),
                thickness=3,
            )
            canonical_axis_path = out_dir / "step5_initial_canonical_axis_refined_on_query.png"
            cv2.imwrite(str(canonical_axis_path), canonical_axis_img)

            print(f"[Step5 Viz] ply overlay         : {ply_overlay_path} (kept_proj={kept_proj})")
            print(f"[Step5 Viz] canonical axis line : {canonical_axis_path}")
        else:
            print(f"[Step5 Viz] canonical_ply_path not found: {canonical_ply_path}")

    query_crop = query_masked
    summary_lines = [
        f"best render: {best_info['best_render']}",
        f"source: {best_info['source']}",
        f"score: {best_info['score']:.4f}",
        f"bbox h(px): {trans_meta['bbox_h_px']:.1f}",
        f"t_init = [{t_init[0]:.4f}, {t_init[1]:.4f}, {t_init[2]:.4f}]",
        f"t_refn = [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]",
        f"quat = [{q[0]:.4f}, {q[1]:.4f},",
        f"        {q[2]:.4f}, {q[3]:.4f}]",
    ]

    pose_vis_path = out_dir / "pose_visualization.png"

    loftr_match_path = out_dir / "loftr_matches_best.png"
    loftr_match_img = None
    if loftr_match_path.exists():
        loftr_match_img = load_image(loftr_match_path)

    build_pose_visualization_v2(
        query_img=query_img,
        query_crop=query_crop,
        best_render=best_render,
        axes_overlay=axes_overlay,
        loftr_match_img=loftr_match_img,
        summary_lines=summary_lines,
        out_path=pose_vis_path,
    )

    initial_pose = {
        "stage": "step5",
        "best_render": best_info["best_render"],
        "best_source": best_info["source"],
        "best_score": float(best_info["score"]),
        "query_img": str(args.query_img),
        "query_masked_path": str(args.query_masked_path),
        "gallery_pose_json": str(args.gallery_pose_json),
        "intrinsics_path": str(args.intrinsics_path),
        "step1_json": str(args.step1_json),
        "bbox_xyxy_used": [int(v) for v in bbox],
        "translation_estimation": trans_meta,
        "object_height_m": float(args.object_height_m),
        "axis_len_m": float(args.axis_len_m),
        "R_obj_to_cam": R.tolist(),
        "t_obj_to_cam": t.tolist(),
        "quat_wxyz": q.tolist(),
        "euler_xyz_deg": euler_deg.tolist(),
        "render_pose_record": pose,
        "t_obj_to_cam_init": t_init.tolist(),
        "t_obj_to_cam_refined": t.tolist(),
        "translation_refinement_json": str(out_dir / "step5_translation_refinement.json") if canonical_xyz is not None else None,
        "outputs": {
            "axes_overlay": str(axes_overlay_path),
            "pose_visualization": str(pose_vis_path),
            "initial_ply_overlay_on_query": None if ply_overlay_path is None else str(ply_overlay_path),
            "initial_canonical_axis_on_query": None if canonical_axis_path is None else str(canonical_axis_path),
            "query_vs_projected_bbox": str(out_dir / "step5_query_vs_projected_bbox.png") if canonical_xyz is not None else None,
        }
    }

    save_json(out_dir / "initial_pose.json", initial_pose)

    print("=" * 60)
    print("[Step 5] Initial pose estimation complete")
    print(f"  best_render : {best_info['best_render']}")
    print(f"  source      : {best_info['source']}")
    print(f"  score       : {best_info['score']:.4f}")
    print(f"  t_obj_to_cam: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    print(f"  json        : {out_dir / 'initial_pose.json'}")
    print(f"  axes        : {axes_overlay_path}")
    print(f"  panel       : {pose_vis_path}")
    print("=" * 60)
