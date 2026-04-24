import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def copy_if_exists(src, dst):
    src = Path(src)
    dst = Path(dst)
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def find_iteration_dir(gs_model_dir: Path, gs_iter=None) -> Path:
    point_root = gs_model_dir / "point_cloud"
    if not point_root.exists():
        raise FileNotFoundError(f"point_cloud directory not found: {point_root}")

    iter_dirs = sorted([p for p in point_root.iterdir() if p.is_dir() and p.name.startswith("iteration_")])
    if not iter_dirs:
        raise FileNotFoundError(f"No iteration_* dirs found in: {point_root}")

    if gs_iter is None:
        return iter_dirs[-1]

    target = point_root / f"iteration_{gs_iter}"
    if not target.exists():
        raise FileNotFoundError(f"Requested iteration dir not found: {target}")
    return target


def quaternion_to_matrix(q):
    # q = [w, x, y, z]
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q = q / n
    w, x, y, z = q

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def matrix_to_quaternion(R):
    # returns [w, x, y, z]
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    qn = np.linalg.norm(q)
    if qn < 1e-12:
        return np.array([1, 0, 0, 0], dtype=np.float64)
    return q / qn


def rotation_matrix_from_a_to_b(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    if s < 1e-12:
        if c > 0:
            return np.eye(3, dtype=np.float64)
        # 180 deg rotation
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - np.dot(axis, a) * a
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        x, y, z = axis
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
        return np.eye(3) + 2 * (K @ K)

    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=np.float64)

    R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
    return R


def load_vertex_data(ply_path):
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    data = vertex.data
    names = data.dtype.names
    return ply, data, names


def structured_to_xyz(data):
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    return xyz


def compute_pca_axis(xyz):
    centroid = xyz.mean(axis=0)
    xc = xyz - centroid
    cov = np.cov(xc.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    axis = eigvecs[:, 0]
    return centroid, eigvals, axis


def choose_up_vector(axis, flip_axis="auto"):
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    if flip_axis == "flip":
        axis = -axis
    elif flip_axis == "auto":
        # prefer +z
        if axis[2] < 0:
            axis = -axis
    # keep => do nothing
    return axis


def measure_height_and_radius(xyz_rotated, percentile=90):
    """
    Measure height (Z-extent) and radius (XY distance) of an
    already-rotated point cloud where Z is the up axis.

    Uses percentile-based trimming to reject outlier Gaussians.
    Returns (height, radius) in the same units as xyz_rotated.
    """
    z = xyz_rotated[:, 2]
    z_lo = np.percentile(z, 100 - percentile)
    z_hi = np.percentile(z, percentile)
    height = float(z_hi - z_lo)

    xy_dist = np.sqrt(xyz_rotated[:, 0] ** 2 + xyz_rotated[:, 1] ** 2)
    radius = float(np.percentile(xy_dist, percentile))

    return height, radius


def build_preview(before_xyz, after_xyz, out_path, max_points=25000,
                  scale_applied=False, scale_factor=1.0,
                  measured_h=None, measured_r=None,
                  target_h=None, target_r=None):
    def normalize_xy(pts, size=900, pad=40):
        xy = pts[:, :2]
        mn = xy.min(axis=0)
        mx = xy.max(axis=0)
        span = np.maximum(mx - mn, 1e-6)
        scale = (size - 2 * pad) / max(span[0], span[1])
        xy2 = (xy - mn) * scale + pad
        return xy2.astype(np.int32)

    def draw_cloud(img, pts2d, color):
        for p in pts2d:
            cv2.circle(img, (int(p[0]), int(p[1])), 1, color, -1, lineType=cv2.LINE_AA)

    rng = np.random.default_rng(0)
    if len(before_xyz) > max_points:
        idx = rng.choice(len(before_xyz), size=max_points, replace=False)
        before_xyz = before_xyz[idx]
    if len(after_xyz) > max_points:
        idx = rng.choice(len(after_xyz), size=max_points, replace=False)
        after_xyz = after_xyz[idx]

    canvas_h, canvas_w = 900, 1800
    img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    left = np.zeros((900, 900, 3), dtype=np.uint8)
    right = np.zeros((900, 900, 3), dtype=np.uint8)

    draw_cloud(left, normalize_xy(before_xyz), (0, 255, 255))
    draw_cloud(right, normalize_xy(after_xyz), (0, 255, 0))

    cv2.putText(left, "Before", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    after_label = "After (scaled)" if scale_applied else "After"
    cv2.putText(right, after_label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Annotate scale info on right panel
    y_text = 100
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, fw = 0.55, 1
    if measured_h is not None:
        cv2.putText(right, f"Measured H: {measured_h:.4f} units", (20, y_text), font, fs, (200, 200, 200), fw)
        y_text += 28
    if measured_r is not None:
        cv2.putText(right, f"Measured R: {measured_r:.4f} units", (20, y_text), font, fs, (200, 200, 200), fw)
        y_text += 28
    if scale_applied:
        cv2.putText(right, f"Scale factor: {scale_factor:.5f}", (20, y_text), font, fs, (100, 255, 100), fw)
        y_text += 28
        if target_h is not None:
            cv2.putText(right, f"Target H: {target_h:.4f} m", (20, y_text), font, fs, (100, 255, 100), fw)
            y_text += 28
        if target_r is not None:
            cv2.putText(right, f"Target R: {target_r:.4f} m", (20, y_text), font, fs, (100, 255, 100), fw)
    else:
        cv2.putText(right, "Scale NOT applied (apply_scale=0)", (20, y_text), font, fs, (100, 100, 255), fw)

    img[:, :900] = left
    img[:, 900:] = right

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), img)


def copy_model_metadata(src_dir: Path, dst_dir: Path):
    for name in ["cameras.json", "cfg_args", "exposure.json", "input.ply"]:
        copy_if_exists(src_dir / name, dst_dir / name)


def write_transformed_ply(in_ply_path, out_ply_path, centroid, R_align, scale_factor=1.0):
    """
    Apply: xyz_out = (xyz_in - centroid) @ R_align.T * scale_factor

    For 3DGS Gaussian rotation quaternions (rot_0..3):
        R_gaussian_new = R_align @ R_gaussian_old   (scale-invariant, rotation only)

    For 3DGS Gaussian scale fields (scale_0..2, stored as log(sigma)):
        log_scale_new = log_scale_old + log(scale_factor)
        i.e. actual sigma is multiplied by scale_factor uniformly
    """
    ply, data, names = load_vertex_data(in_ply_path)

    xyz = structured_to_xyz(data)
    xyz_rotated = ((xyz - centroid) @ R_align.T)
    xyz_new = (xyz_rotated * scale_factor).astype(np.float32)

    has_rot = all(k in names for k in ["rot_0", "rot_1", "rot_2", "rot_3"])
    # Works for both 3DGS (scale_0/1/2) and 2DGS (scale_0/1 only)
    gs_scale_names = {n for n in ["scale_0", "scale_1", "scale_2"] if n in names}
    has_gs_scale = len(gs_scale_names) > 0

    log_scale_offset = float(np.log(scale_factor)) if scale_factor != 1.0 else 0.0

    rows = []
    for i in range(len(data)):
        row = []
        for name in names:
            if name == "x":
                row.append(float(xyz_new[i, 0]))
            elif name == "y":
                row.append(float(xyz_new[i, 1]))
            elif name == "z":
                row.append(float(xyz_new[i, 2]))
            elif has_rot and name in ["rot_0", "rot_1", "rot_2", "rot_3"]:
                row.append(None)  # filled below
            elif has_gs_scale and name in gs_scale_names:
                # log space: add log(scale_factor) — works for 3DGS (3 scales) and 2DGS (2 scales)
                row.append(float(data[name][i]) + log_scale_offset)
            else:
                row.append(data[name][i])
        rows.append(row)

    if has_rot:
        name_to_idx = {n: idx for idx, n in enumerate(names)}
        i0 = name_to_idx["rot_0"]
        i1 = name_to_idx["rot_1"]
        i2 = name_to_idx["rot_2"]
        i3 = name_to_idx["rot_3"]

        for i in range(len(data)):
            q = np.array([
                float(data["rot_0"][i]),
                float(data["rot_1"][i]),
                float(data["rot_2"][i]),
                float(data["rot_3"][i]),
            ], dtype=np.float64)

            Rg = quaternion_to_matrix(q)
            Rg_new = R_align @ Rg   # scale does NOT affect rotation
            q_new = matrix_to_quaternion(Rg_new).astype(np.float32)

            rows[i][i0] = float(q_new[0])
            rows[i][i1] = float(q_new[1])
            rows[i][i2] = float(q_new[2])
            rows[i][i3] = float(q_new[3])

    dtype = [(name, data.dtype[name]) for name in names]
    out_arr = np.empty(len(rows), dtype=dtype)

    for j, name in enumerate(names):
        out_arr[name] = [row[j] for row in rows]

    out_el = PlyElement.describe(out_arr, "vertex")
    out_ply = PlyData([out_el], text=ply.text)

    out_ply_path = Path(out_ply_path)
    ensure_dir(out_ply_path.parent)
    out_ply.write(str(out_ply_path))

    return xyz, xyz_rotated, xyz_new  # (original, rotated-only, rotated+scaled)


def run_step2_canonicalize_gs_model(args):
    gs_model_dir = Path(args.gs_model_dir)
    canonical_model_dir = Path(args.canonical_model_dir)

    iter_dir = find_iteration_dir(gs_model_dir, args.gs_iter)
    iter_name = iter_dir.name

    in_ply = iter_dir / "point_cloud.ply"
    if not in_ply.exists():
        raise FileNotFoundError(f"point_cloud.ply not found: {in_ply}")

    out_iter_dir = canonical_model_dir / "point_cloud" / iter_name
    out_ply = out_iter_dir / "point_cloud.ply"
    out_alias_ply = canonical_model_dir / "point_cloud_canonical.ply"
    out_json = canonical_model_dir / "canonical_transform.json"

    ensure_dir(canonical_model_dir)
    ensure_dir(out_iter_dir)

    copy_model_metadata(gs_model_dir, canonical_model_dir)

    _, data, _ = load_vertex_data(in_ply)
    xyz = structured_to_xyz(data)

    centroid, eigvals, axis = compute_pca_axis(xyz)
    axis = choose_up_vector(axis, flip_axis=args.flip_axis)

    up_target = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }[args.up_axis]

    R_align = rotation_matrix_from_a_to_b(axis, up_target)

    # ── Measure geometry AFTER rotation, BEFORE scaling ──────────────────────
    xyz_rotated = (xyz - centroid) @ R_align.T
    measured_height, measured_radius = measure_height_and_radius(xyz_rotated, percentile=90)

    # ── Decide scale factor ───────────────────────────────────────────────────
    apply_scale = bool(args.apply_scale)
    scale_factor = 1.0
    scale_basis = "none"

    if apply_scale:
        target_height_m = float(args.target_height_m)
        target_radius_m = float(args.target_radius_m)

        scale_from_height = target_height_m / measured_height if measured_height > 1e-9 else 1.0
        scale_from_radius = target_radius_m / measured_radius if measured_radius > 1e-9 else 1.0

        # Height is more reliable for cylindrical objects.
        # If the two agree within 20%, use height; otherwise warn and still use height.
        ratio = scale_from_height / (scale_from_radius + 1e-12)
        if not (0.8 < ratio < 1.25):
            print(f"[Step 2] WARNING: scale_from_height ({scale_from_height:.4f}) and "
                  f"scale_from_radius ({scale_from_radius:.4f}) disagree by more than 20%. "
                  f"Using height-based scale. Check target values or object geometry.")

        scale_factor = scale_from_height
        scale_basis = "height"

        print(f"[Step 2] Scale measurement:")
        print(f"  measured height : {measured_height:.6f} units")
        print(f"  measured radius : {measured_radius:.6f} units")
        print(f"  target  height  : {target_height_m:.6f} m")
        print(f"  target  radius  : {target_radius_m:.6f} m")
        print(f"  scale_from_height = {scale_from_height:.6f}")
        print(f"  scale_from_radius = {scale_from_radius:.6f}")
        print(f"  → using scale_factor = {scale_factor:.6f}  (basis: {scale_basis})")
    else:
        print(f"[Step 2] Scale measurement (apply_scale=0, no scaling applied):")
        print(f"  measured height : {measured_height:.6f} units")
        print(f"  measured radius : {measured_radius:.6f} units")
        print(f"  If you want metric-scale PLY, set apply_scale=1.")

    # ── Write PLY ─────────────────────────────────────────────────────────────
    xyz_before, xyz_rotated_only, xyz_after = write_transformed_ply(
        in_ply, out_ply, centroid, R_align, scale_factor=scale_factor
    )
    shutil.copy2(out_ply, out_alias_ply)

    # ── Preview ───────────────────────────────────────────────────────────────
    preview_path = args.preview_before_after_path
    if preview_path is None:
        preview_path = str(canonical_model_dir / "canonical_preview_before_after.png")
    # build_preview(
    #     xyz_before, xyz_after, preview_path,
    #     max_points=args.max_preview_points,
    #     scale_applied=apply_scale,
    #     scale_factor=scale_factor,
    #     measured_h=measured_height,
    #     measured_r=measured_radius,
    #     target_h=float(args.target_height_m),
    #     target_r=float(args.target_radius_m),
    # )

    # ── Build transform matrix (full similarity: R + t + s) ───────────────────
    # X_canonical = scale_factor * R_align @ (X_original - centroid)
    # As a 4x4 homogeneous matrix (note: this is a similarity transform, not rigid):
    T_mat = np.eye(4, dtype=np.float64)
    T_mat[:3, :3] = scale_factor * R_align
    T_mat[:3, 3]  = -scale_factor * (R_align @ centroid)

    # Suggested gallery radius for step2 (object surface + comfortable distance)
    # Camera distance ≈ 2.5~3× object half-height after scaling
    if apply_scale:
        suggested_gallery_radius = float(args.target_height_m) * 2.8
    else:
        suggested_gallery_radius = measured_height * 2.8

    info = {
        "stage": "step2",
        "type": "gs_aware_canonicalization",
        "input_gs_model_dir": str(gs_model_dir),
        "output_canonical_model_dir": str(canonical_model_dir),
        "input_ply_path": str(in_ply),
        "output_canonical_ply_path": str(out_ply),
        "output_alias_ply_path": str(out_alias_ply),
        "iteration_name": iter_name,
        "axis_method": args.axis_method,
        "up_axis": args.up_axis,
        "flip_axis": args.flip_axis,
        # ── scale ──
        "apply_scale": int(args.apply_scale),
        "scale_factor": float(scale_factor),
        "scale_basis": scale_basis,
        "measured_height_units": float(measured_height),
        "measured_radius_units": float(measured_radius),
        "target_height_m": float(args.target_height_m),
        "target_radius_m": float(args.target_radius_m),
        # ── geometry ──
        "preview_before_after_path": str(preview_path),
        "centroid_original_xyz": centroid.tolist(),
        "pca_eigenvalues_desc": eigvals.tolist(),
        "height_axis_original_frame": axis.tolist(),
        "R_align": R_align.tolist(),
        "T_similarity_4x4": T_mat.tolist(),   # full similarity transform
        # ── hints for downstream steps ──
        "suggested_gallery_radius_m": round(suggested_gallery_radius, 4),
        "notes": [
            "GS-aware canonicalization applied to xyz and Gaussian rotations.",
            "If apply_scale=1: xyz multiplied by scale_factor; scale_0/1/2 (log-space) shifted by log(scale_factor).",
            "If apply_scale=0: only rotation+centering applied, PLY is NOT metric.",
            "T_similarity_4x4: X_canonical = T @ [X_original; 1]  (homogeneous).",
            "suggested_gallery_radius_m: recommended RADIUS value for step2 run_6d.sh.",
            "Quaternion order is (w, x, y, z). Verify with your GS repo loader.",
        ]
    }

    save_json(out_json, info)

    print("=" * 60)
    print("[Step 2] GS-aware canonicalization complete")
    print(f"  input model dir : {gs_model_dir}")
    print(f"  output model dir: {canonical_model_dir}")
    print(f"  input ply       : {in_ply}")
    print(f"  output ply      : {out_ply}")
    print(f"  alias ply       : {out_alias_ply}")
    print(f"  transform json  : {out_json}")
    print(f"  preview         : {preview_path}")
    print(f"  scale_factor    : {scale_factor:.6f}  (apply_scale={int(args.apply_scale)})")
    print(f"  suggested RADIUS for step2: {suggested_gallery_radius:.4f} m")
    print("=" * 60)