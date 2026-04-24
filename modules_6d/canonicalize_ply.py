import os
import json
import numpy as np
import matplotlib.pyplot as plt

from .io_utils import ensure_dir, save_json


def _axis_index(axis_name: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis_name]


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.copy()
    return v / n


def _rotation_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _normalize(a)
    b = _normalize(b)
    v = np.cross(a, b)
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    s = np.linalg.norm(v)
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        # 180-degree rotation: choose arbitrary orthogonal axis
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        v = _normalize(np.cross(a, tmp))
        K = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
        return np.eye(3) + 2 * (K @ K)
    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
    return R


def _choose_height_axis(points_centered: np.ndarray) -> np.ndarray:
    cov = np.cov(points_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs


def _estimate_axis_and_dims(points: np.ndarray, axis: np.ndarray):
    axis = _normalize(axis)
    proj = points @ axis
    height = float(proj.max() - proj.min())
    radial_vecs = points - np.outer(proj, axis)
    radial = np.linalg.norm(radial_vecs, axis=1)
    radius = float(np.percentile(radial, 95))
    return height, radius, proj, radial


def _sample_points(arr: np.ndarray, max_n: int, seed: int = 0) -> np.ndarray:
    n = len(arr)
    if n <= max_n:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return arr[idx]


def _read_ply_with_plyfile(ply_path: str):
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    arr = vertex.data.copy()
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float64)
    return ply, arr, xyz


def _write_ply_with_plyfile(template_ply, vertex_arr, out_path: str):
    from plyfile import PlyData, PlyElement
    el = PlyElement.describe(vertex_arr, 'vertex')
    new_ply = PlyData([el], text=template_ply.text, byte_order=template_ply.byte_order)
    ensure_dir(os.path.dirname(out_path))
    new_ply.write(out_path)


def _maybe_transform_normals(vertex_arr, R: np.ndarray):
    names = set(vertex_arr.dtype.names)
    if {'nx', 'ny', 'nz'}.issubset(names):
        N = np.stack([vertex_arr['nx'], vertex_arr['ny'], vertex_arr['nz']], axis=1).astype(np.float64)
        N2 = (R @ N.T).T
        vertex_arr['nx'] = N2[:, 0].astype(vertex_arr['nx'].dtype)
        vertex_arr['ny'] = N2[:, 1].astype(vertex_arr['ny'].dtype)
        vertex_arr['nz'] = N2[:, 2].astype(vertex_arr['nz'].dtype)


def _maybe_transform_log_scales(vertex_arr, scale_factor: float):
    names = set(vertex_arr.dtype.names)
    scale_names = [n for n in ['scale_0', 'scale_1', 'scale_2'] if n in names]
    if scale_names and scale_factor > 0:
        offset = np.log(scale_factor)
        for n in scale_names:
            vertex_arr[n] = (vertex_arr[n].astype(np.float64) + offset).astype(vertex_arr[n].dtype)


def _plot_before_after(before: np.ndarray, after: np.ndarray, out_path: str, max_points: int = 25000):
    b = _sample_points(before, max_points, seed=0)
    a = _sample_points(after, max_points, seed=1)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.scatter(b[:, 0], b[:, 1], b[:, 2], s=1)
    ax1.set_title('Before canonicalization')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2.scatter(a[:, 0], a[:, 1], a[:, 2], s=1)
    ax2.set_title('After canonicalization')
    ax2.set_xlabel('X_obj')
    ax2.set_ylabel('Y_obj')
    ax2.set_zlabel('Z_obj')

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def run_step25_canonicalize_ply(args):
    try:
        ply_template, vertex_arr, xyz = _read_ply_with_plyfile(args.model_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to read PLY. This step requires 'plyfile'. Install with: pip install plyfile\n"
            f"Original error: {e}"
        )

    centroid = xyz.mean(axis=0)
    xyz_centered = xyz - centroid[None, :]

    eigvals, eigvecs = _choose_height_axis(xyz_centered)
    # PCA major axis candidate for can height
    height_axis = eigvecs[:, 0].copy()

    # Auto flip so positive target axis tends toward original positive world-up if possible
    up_idx = _axis_index(args.up_axis)
    target_axis = np.zeros(3, dtype=np.float64)
    target_axis[up_idx] = 1.0

    if args.flip_axis == 'auto':
        if np.dot(height_axis, np.array([0.0, 0.0, 1.0])) < 0:
            height_axis = -height_axis
    elif args.flip_axis == 'flip':
        height_axis = -height_axis

    R_align = _rotation_from_a_to_b(height_axis, target_axis)
    xyz_aligned = (R_align @ xyz_centered.T).T

    est_height_before, est_radius_before, _, _ = _estimate_axis_and_dims(xyz_centered, height_axis)
    est_height_after, est_radius_after, _, _ = _estimate_axis_and_dims(xyz_aligned, target_axis)

    scale_factor = 1.0
    if int(args.apply_scale) == 1:
        candidates = []
        if args.target_height_m > 0 and est_height_after > 1e-9:
            candidates.append(args.target_height_m / est_height_after)
        if args.target_radius_m > 0 and est_radius_after > 1e-9:
            candidates.append(args.target_radius_m / est_radius_after)
        if len(candidates) == 0:
            raise RuntimeError("apply_scale=1 but no valid target dimension available.")
        # Prefer height because can z-axis is the most meaningful dimension here.
        scale_factor = candidates[0]
        xyz_canonical = xyz_aligned * scale_factor
    else:
        xyz_canonical = xyz_aligned

    # update vertex positions
    vertex_out = vertex_arr.copy()
    vertex_out['x'] = xyz_canonical[:, 0].astype(vertex_out['x'].dtype)
    vertex_out['y'] = xyz_canonical[:, 1].astype(vertex_out['y'].dtype)
    vertex_out['z'] = xyz_canonical[:, 2].astype(vertex_out['z'].dtype)

    # rotate normals, if present
    _maybe_transform_normals(vertex_out, R_align)
    # scale Gaussian log-scales, if present
    _maybe_transform_log_scales(vertex_out, scale_factor)

    _write_ply_with_plyfile(ply_template, vertex_out, args.canonical_ply_path)
    _plot_before_after(xyz, xyz_canonical, args.preview_before_after_path, max_points=args.max_preview_points)

    est_height_final, est_radius_final, _, _ = _estimate_axis_and_dims(xyz_canonical, target_axis)

    T_rs_to_obj = np.eye(4, dtype=np.float64)
    T_rs_to_obj[:3, :3] = scale_factor * R_align
    T_rs_to_obj[:3, 3] = -scale_factor * (R_align @ centroid)

    summary = {
        "stage": "step25",
        "input_model_path": args.model_path,
        "output_canonical_ply_path": args.canonical_ply_path,
        "output_transform_json_path": args.canonical_json_path,
        "preview_before_after_path": args.preview_before_after_path,
        "axis_method": args.axis_method,
        "up_axis": args.up_axis,
        "flip_axis": args.flip_axis,
        "apply_scale": int(args.apply_scale),
        "target_height_m": float(args.target_height_m),
        "target_radius_m": float(args.target_radius_m),
        "centroid_original_xyz": centroid.tolist(),
        "pca_eigenvalues_desc": eigvals.tolist(),
        "estimated_height_before": est_height_before,
        "estimated_radius95_before": est_radius_before,
        "estimated_height_after_before_scaling": est_height_after,
        "estimated_radius95_after_before_scaling": est_radius_after,
        "scale_factor": float(scale_factor),
        "estimated_height_final": est_height_final,
        "estimated_radius95_final": est_radius_final,
        "height_axis_original_frame": height_axis.tolist(),
        "R_align": R_align.tolist(),
        "T_rs_to_obj": T_rs_to_obj.tolist(),
        "notes": [
            "Canonical frame target: object center at origin.",
            f"Chosen up axis: +{args.up_axis}.",
            "PCA major axis is used as the can height axis candidate.",
            "If the result looks wrong, rerun with flip_axis=flip or keep, and inspect the preview image.",
            "If scale is not yet trustworthy, keep apply_scale=0 for now and enable it after visual confirmation."
        ]
    }
    save_json(args.canonical_json_path, summary)

    print("=" * 60)
    print("[Step 2] PLY canonicalization complete")
    print(f"  input       : {args.model_path}")
    print(f"  canonical   : {args.canonical_ply_path}")
    print(f"  transform   : {args.canonical_json_path}")
    print(f"  preview     : {args.preview_before_after_path}")
    print(f"  centroid    : {centroid.tolist()}")
    print(f"  scale       : {scale_factor:.6f}")
    print(f"  height_est  : {est_height_final:.6f}")
    print(f"  radius95_est: {est_radius_final:.6f}")
    print("=" * 60)
