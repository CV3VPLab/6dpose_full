#!/usr/bin/env python3
"""
Generate gallery_poses.json from icosphere camera directions.

This is the compact pose generator used by the current stream pipeline:
icosphere directions, optional level-specific radii, in-plane roll angles, and
JSON/CSV/preview outputs.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def parse_vec3(text: str) -> np.ndarray:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {text}")
    return np.asarray(vals, dtype=np.float64)


def parse_float_list(text: str) -> list[float]:
    vals = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not vals:
        raise ValueError(f"Expected at least one float, got: {text}")
    return vals


def parse_level_radii(text: str) -> dict[int, list[float]]:
    mapping: dict[int, list[float]] = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected LEVEL:RADIUS entry, got: {item}")
        level_text, radii_text = item.split(":", 1)
        level = int(level_text.strip())
        radii = [float(v.strip()) for v in radii_text.split("|") if v.strip()]
        if not radii:
            raise ValueError(f"Expected at least one radius for level {level}")
        mapping[level] = radii
    if not mapping:
        raise ValueError(f"Expected at least one LEVEL:RADIUS entry, got: {text}")
    return mapping


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector encountered during normalization")
    return v / n


def get_icosahedron() -> tuple[np.ndarray, list[list[int]]]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.asarray([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1)[:, None]

    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    return verts, faces


def subdivide(
    verts: np.ndarray,
    faces: list[list[int]],
    levels: list[int],
    midpoint_positions: list[int],
    subdivision_level: int,
) -> tuple[np.ndarray, list[list[int]], list[int], list[int]]:
    new_faces = []
    edge_to_midpoint: dict[tuple[int, int], int] = {}
    verts_list = list(verts)

    def midpoint(v0: int, v1: int, position: int) -> int:
        key = tuple(sorted((v0, v1)))
        if key in edge_to_midpoint:
            return edge_to_midpoint[key]

        mid = normalize((verts_list[v0] + verts_list[v1]) * 0.5)
        idx = len(verts_list)
        verts_list.append(mid)
        levels.append(subdivision_level)
        midpoint_positions.append(position)
        edge_to_midpoint[key] = idx
        return idx

    for v0, v1, v2 in faces:
        a = midpoint(v0, v1, 1)
        b = midpoint(v1, v2, 2)
        c = midpoint(v2, v0, 3)
        new_faces.extend([[v0, a, c], [v1, b, a], [v2, c, b], [a, b, c]])

    return np.asarray(verts_list, dtype=np.float64), new_faces, levels, midpoint_positions


def build_icosphere_with_levels(subdivisions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts, faces = get_icosahedron()
    levels = [0] * len(verts)
    midpoint_positions = [0] * len(verts)

    for subdivision_level in range(1, subdivisions + 1):
        verts, faces, levels, midpoint_positions = subdivide(
            verts, faces, levels, midpoint_positions, subdivision_level
        )

    level_arr = np.asarray(levels, dtype=np.int64)
    position_arr = np.asarray(midpoint_positions, dtype=np.int64)
    azimuth = np.arctan2(verts[:, 1], verts[:, 0])
    order = np.lexsort((azimuth, verts[:, 2], level_arr))
    return verts[order], level_arr[order], position_arr[order]


def radii_by_direction(
    levels: np.ndarray,
    default_radii: list[float],
    level_radii: dict[int, list[float]] | None,
    roll_scheme: str,
) -> list[list[float]]:
    if roll_scheme == "level2_close_rolls":
        radii = []
        for level in levels:
            level_int = int(level)
            if level_int == 0:
                radii.append([0.7])
            elif level_int == 1:
                radii.append([0.6])
            elif level_int == 2:
                radii.append([0.5, 0.4])
            else:
                radii.append([])
        return radii

    if level_radii is None:
        return [default_radii for _ in range(len(levels))]

    radii = []
    for level in levels:
        level_int = int(level)
        if level_int not in level_radii:
            raise ValueError(f"--level_radii does not define radius for icosphere level {level_int}")
        radii.append([float(r) for r in level_radii[level_int]])
    return radii


def rolls_by_direction(
    levels: np.ndarray,
    roll_scheme: str,
    default_rolls: list[float],
) -> list[list[float]]:
    if roll_scheme == "uniform":
        return [default_rolls for _ in range(len(levels))]
    if roll_scheme == "level2_close_rolls":
        return [[] for _ in range(len(levels))]
    raise ValueError(f"Unsupported roll scheme: {roll_scheme}")


def direction_to_az_el(direction: np.ndarray) -> tuple[float, float]:
    x, y, z = direction.tolist()
    az = math.degrees(math.atan2(y, x))
    if az < 0.0:
        az += 360.0
    el = math.degrees(math.asin(float(np.clip(z, -1.0, 1.0))))
    return az, el


def make_opencv_obj_to_cam(camera_pos: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    z_cam_obj = normalize(target - camera_pos)
    x_cam_obj = np.cross(z_cam_obj, up_hint)
    if np.linalg.norm(x_cam_obj) < 1e-8:
        x_cam_obj = np.cross(z_cam_obj, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
    x_cam_obj = normalize(x_cam_obj)
    y_cam_obj = normalize(np.cross(z_cam_obj, x_cam_obj))
    return np.stack([x_cam_obj, y_cam_obj, z_cam_obj], axis=0)


def make_roll_matrix(roll_deg: float) -> np.ndarray:
    r = math.radians(roll_deg)
    cr, sr = math.cos(r), math.sin(r)
    return np.asarray([
        [cr, sr, 0.0],
        [-sr, cr, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def generate_poses(
    directions: np.ndarray,
    levels: np.ndarray,
    midpoint_positions: np.ndarray,
    per_direction_radii: list[list[float]],
    per_direction_rolls: list[list[float]],
    look_at: np.ndarray,
    up_hint: np.ndarray,
    min_elevation_deg: float,
    max_elevation_deg: float,
) -> list[dict]:
    poses = []

    for dir_idx, direction in enumerate(directions):
        az, el = direction_to_az_el(direction)
        if el < min_elevation_deg or el > max_elevation_deg:
            continue

        for radius in per_direction_radii[dir_idx]:
            camera_pos = look_at + float(radius) * direction
            R_base = make_opencv_obj_to_cam(camera_pos, look_at, up_hint)
            rolls = per_direction_rolls[dir_idx]
            if int(levels[dir_idx]) == 2 and float(radius) == 0.4:
                rolls = [float(v) for v in range(0, 360, 30)]
            elif not rolls:
                rolls = [0.0, 72.0, 144.0, 216.0, 288.0]

            for roll in rolls:
                R = R_base if float(roll) == 0.0 else make_roll_matrix(float(roll)) @ R_base
                t = (-R @ camera_pos.reshape(3, 1))[:, 0]
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = t

                poses.append({
                    "index": len(poses),
                    "azimuth_deg": float(az),
                    "elevation_deg": float(el),
                    "radius": float(radius),
                    "roll_deg": float(roll),
                    "look_at": look_at.tolist(),
                    "camera_position_obj_frame": camera_pos.tolist(),
                    "R_obj_to_cam": R.tolist(),
                    "t_obj_to_cam": t.tolist(),
                    "T_obj_to_cam": T.tolist(),
                    "sample_source": "icosphere_vertex",
                    "icosphere_vertex_index": int(dir_idx),
                    "icosphere_level": int(levels[dir_idx]),
                    "icosphere_midpoint_position": int(midpoint_positions[dir_idx]),
                })

    return poses


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_preview(path: Path, poses: list[dict], preview_size: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] Skipping preview because matplotlib import failed: {exc}")
        return

    if not poses:
        print("[warn] Skipping preview because no poses were generated")
        return

    fig = plt.figure(figsize=(8, 8), dpi=max(100, int(preview_size) // 8))
    ax = fig.add_subplot(111, projection="3d")
    positions = np.asarray([p["camera_position_obj_frame"] for p in poses], dtype=np.float64)
    levels = sorted(set(int(p["icosphere_level"]) for p in poses))
    colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))

    for level, color in zip(levels, colors):
        mask = np.asarray([int(p["icosphere_level"]) == level for p in poses])
        ax.scatter(
            positions[mask, 0], positions[mask, 1], positions[mask, 2],
            s=12, color=color, label=f"level {level}",
        )

    ax.scatter([0], [0], [0], s=90, marker="x", color="red")
    lim = max(1e-6, float(np.max(np.abs(positions)))) * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("X_obj")
    ax.set_ylabel("Y_obj")
    ax.set_zlabel("Z_obj")
    ax.set_title(f"Icosphere gallery camera positions ({len(poses)} poses)")
    ax.view_init(elev=28, azim=35)
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate icosphere-sampled gallery_poses.json")
    p.add_argument("--out_dir", default="data", help="Output directory")
    p.add_argument("--subdivisions", type=int, default=4, help="0=12, 1=42, 2=162, 3=642, 4=2562 directions")
    p.add_argument("--radius", default="0.4,0.5,0.6,0.7", help="Fallback radius list when --level_radii is not set")
    p.add_argument("--level_radii", default=None, help="Level radii, e.g. '0:0.4,1:0.4,2:0.4,3:0.5,4:0.6'")
    p.add_argument("--roll_angles_deg", default="-90,-45,0,45,90", help="Comma-separated in-plane roll angles")
    p.add_argument("--roll_scheme", default="uniform", choices=["uniform", "level2_close_rolls"])
    p.add_argument("--look_at", default="0,0,0", help="Object-frame look-at point")
    p.add_argument("--up_hint", default="0,0,1", help="Object-frame up hint")
    p.add_argument("--min_elevation_deg", type=float, default=-90.0)
    p.add_argument("--max_elevation_deg", type=float, default=90.0)
    p.add_argument("--preview_size", type=int, default=900)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.subdivisions < 0:
        raise ValueError("--subdivisions must be >= 0")

    out_dir = Path(args.out_dir)
    default_radii = parse_float_list(args.radius)
    level_radii = parse_level_radii(args.level_radii) if args.level_radii else None
    roll_angles_deg = parse_float_list(args.roll_angles_deg)
    look_at = parse_vec3(args.look_at)
    up_hint = normalize(parse_vec3(args.up_hint))

    directions, levels, midpoint_positions = build_icosphere_with_levels(args.subdivisions)
    per_direction_radii = radii_by_direction(levels, default_radii, level_radii, args.roll_scheme)
    per_direction_rolls = rolls_by_direction(
        levels=levels,
        roll_scheme=args.roll_scheme,
        default_rolls=roll_angles_deg,
    )
    poses = generate_poses(
        directions=directions,
        levels=levels,
        midpoint_positions=midpoint_positions,
        per_direction_radii=per_direction_radii,
        per_direction_rolls=per_direction_rolls,
        look_at=look_at,
        up_hint=up_hint,
        min_elevation_deg=float(args.min_elevation_deg),
        max_elevation_deg=float(args.max_elevation_deg),
    )

    level_counts = {
        str(level): int(np.count_nonzero(levels == level))
        for level in sorted(set(int(v) for v in levels.tolist()))
    }
    midpoint_position_counts = {
        str(position): int(np.count_nonzero(midpoint_positions == position))
        for position in sorted(set(int(v) for v in midpoint_positions.tolist()))
    }

    payload = {
        "stage": "step3",
        "pose_convention": {
            "name": "opencv_like_obj_to_cam",
            "definition": "X_cam = R * X_obj + t, with camera axes x:right, y:down, z:forward",
            "camera_looks_toward": "look_at target in object frame",
        },
        "settings": {
            "sampling": "icosphere",
            "subdivisions": int(args.subdivisions),
            "num_icosphere_vertices": int(len(directions)),
            "icosphere_level_counts": level_counts,
            "icosphere_midpoint_position_counts": midpoint_position_counts,
            "min_elevation_deg": float(args.min_elevation_deg),
            "max_elevation_deg": float(args.max_elevation_deg),
            "radius_mode": "explicit_level_radii" if level_radii is not None else "all_radii_for_all_vertices",
            "radius_settings": {
                "radius_list": default_radii,
                "level_radii": level_radii,
                "base_level": None,
                "base_radius": None,
                "new_vertex_radii": None,
            },
            "roll_scheme": args.roll_scheme,
            "roll_angles_deg": roll_angles_deg,
            "coarse_roll_angles_deg": [0.0, 72.0, 144.0, 216.0, 288.0],
            "level2_close_rolls": {
                "level0": {"radius": 0.7, "roll_angles_deg": [0, 72, 144, 216, 288]},
                "level1": {"radius": 0.6, "roll_angles_deg": [0, 72, 144, 216, 288]},
                "level2_far": {"radius": 0.5, "roll_angles_deg": [0, 72, 144, 216, 288]},
                "level2_close": {"radius": 0.4, "roll_angles_deg": list(range(0, 360, 30))},
            } if args.roll_scheme == "level2_close_rolls" else None,
            "position_roll_level": 3,
            "position_rolls": {
                1: [0.0, 40.0, 80.0],
                2: [120.0, 160.0, 200.0],
                3: [240.0, 280.0, 320.0],
            },
            "look_at": look_at.tolist(),
            "up_hint": up_hint.tolist(),
        },
        "num_poses": len(poses),
        "poses": poses,
    }

    json_path = out_dir / "gallery_poses.json"
    preview_path = out_dir / "gallery_pose_preview.png"
    save_json(json_path, payload)
    save_preview(preview_path, poses, preview_size=int(args.preview_size))

    print("[generate_gallery_poses.py] complete")
    print(f"  subdivisions : {args.subdivisions}")
    print(f"  directions   : {len(directions)}")
    print(f"  level counts : {level_counts}")
    print(f"  poses        : {len(poses)}")
    print(f"  json         : {json_path}")
    print(f"  preview      : {preview_path}")


if __name__ == "__main__":
    main()
