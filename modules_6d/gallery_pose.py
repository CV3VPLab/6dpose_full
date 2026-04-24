import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import ensure_dir, save_json


def _parse_vec3(text: str) -> np.ndarray:
    vals = [float(v.strip()) for v in text.split(",")]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {text}")
    return np.asarray(vals, dtype=np.float64)


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector encountered during normalization")
    return v / n


def spherical_to_cartesian(radius: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = radius * math.cos(el) * math.cos(az)
    y = radius * math.cos(el) * math.sin(az)
    z = radius * math.sin(el)
    return np.asarray([x, y, z], dtype=np.float64)


def make_opencv_obj_to_cam(camera_pos: np.ndarray, target: np.ndarray, up_hint: np.ndarray):
    """
    Build object-to-camera transform using an OpenCV-like camera frame:
      x: right, y: down, z: forward
    The camera looks from camera_pos toward target.
    """
    z_cam_obj = _normalize(target - camera_pos)
    x_cam_obj = np.cross(z_cam_obj, up_hint)
    if np.linalg.norm(x_cam_obj) < 1e-8:
        alt_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        x_cam_obj = np.cross(z_cam_obj, alt_up)
    x_cam_obj = _normalize(x_cam_obj)
    y_cam_obj = _normalize(np.cross(z_cam_obj, x_cam_obj))

    R = np.stack([x_cam_obj, y_cam_obj, z_cam_obj], axis=0)
    t = -R @ camera_pos.reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t[:, 0]
    return R, t[:, 0], T


def make_roll_matrix(roll_deg: float) -> np.ndarray:
    """
    Roll rotation around the camera's z-axis (forward axis).
    Applied after the base R to add in-plane rotation.
    R_roll @ R_base → object appears rotated around the viewing axis.
    """
    r = math.radians(roll_deg)
    cr, sr = math.cos(r), math.sin(r)
    return np.array([
        [ cr, sr, 0.0],
        [-sr, cr, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def generate_gallery_poses(
    elevations_deg,
    azimuth_step_deg,
    radius,               # float 또는 list[float]
    look_at,
    up_hint,
    roll_angles_deg=None, # list[float], e.g. [-15, 0, 15]
):
    """
    Gallery pose 생성.

    radius: float 하나 또는 list → multi-radius 지원
    roll_angles_deg: in-plane roll 목록. None이면 [0.0]으로 처리.

    중복 제거:
      - roll=0일 때만 base pose (roll 없음)와 동일하므로 문제없음
      - 완전 동일한 (az, el, radius, roll) 조합은 생기지 않음
    """
    if isinstance(radius, (int, float)):
        radius_list = [float(radius)]
    else:
        radius_list = [float(r) for r in radius]

    if roll_angles_deg is None:
        roll_angles_deg = [0.0]

    poses = []
    idx = 0
    azimuths = np.arange(0.0, 360.0, azimuth_step_deg, dtype=np.float64)

    for rad in radius_list:
        for el in elevations_deg:
            for az in azimuths:
                camera_pos = spherical_to_cartesian(rad, float(az), float(el)) + look_at
                R_base, t_base, T_base = make_opencv_obj_to_cam(camera_pos, look_at, up_hint)

                for roll in roll_angles_deg:
                    if roll == 0.0:
                        R_final = R_base
                    else:
                        R_roll  = make_roll_matrix(roll)
                        R_final = R_roll @ R_base

                    t_final = -R_final @ camera_pos.reshape(3, 1)
                    t_final = t_final[:, 0]

                    T_final = np.eye(4, dtype=np.float64)
                    T_final[:3, :3] = R_final
                    T_final[:3, 3]  = t_final

                    pose = {
                        "index":                      idx,
                        "azimuth_deg":                float(az),
                        "elevation_deg":              float(el),
                        "radius":                     float(rad),
                        "roll_deg":                   float(roll),
                        "look_at":                    look_at.tolist(),
                        "camera_position_obj_frame":  camera_pos.tolist(),
                        "R_obj_to_cam":               R_final.tolist(),
                        "t_obj_to_cam":               t_final.tolist(),
                        "T_obj_to_cam":               T_final.tolist(),
                    }
                    poses.append(pose)
                    idx += 1

    return poses


def save_pose_csv(csv_path: Path, poses):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "azimuth_deg", "elevation_deg", "radius", "roll_deg",
            "cam_x", "cam_y", "cam_z", "tx", "ty", "tz"
        ])
        for p in poses:
            cx, cy, cz = p["camera_position_obj_frame"]
            tx, ty, tz = p["t_obj_to_cam"]
            writer.writerow([
                p["index"], p["azimuth_deg"], p["elevation_deg"],
                p["radius"], p.get("roll_deg", 0.0),
                cx, cy, cz, tx, ty, tz
            ])


def save_pose_preview(out_path: Path, poses, preview_size: int = 900):
    fig = plt.figure(figsize=(8, 8), dpi=max(100, preview_size // 8))
    ax = fig.add_subplot(111, projection="3d")

    xs = [p["camera_position_obj_frame"][0] for p in poses]
    ys = [p["camera_position_obj_frame"][1] for p in poses]
    zs = [p["camera_position_obj_frame"][2] for p in poses]

    # radius별로 색 구분
    radii  = sorted(set(p["radius"] for p in poses))
    colors = plt.cm.tab10(np.linspace(0, 1, len(radii)))
    rad_to_color = {r: c for r, c in zip(radii, colors)}

    for p in poses:
        col = rad_to_color[p["radius"]]
        ax.scatter(
            p["camera_position_obj_frame"][0],
            p["camera_position_obj_frame"][1],
            p["camera_position_obj_frame"][2],
            s=20, color=col,
        )

    ax.scatter([0], [0], [0], s=90, marker="x", color="red")

    max_abs = max(1e-6, np.max(np.abs(np.asarray([xs, ys, zs], dtype=np.float64))))
    lim = max_abs * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("X_obj")
    ax.set_ylabel("Y_obj")
    ax.set_zlabel("Z_obj")
    ax.set_title(f"Gallery camera positions ({len(radii)} radius, {len(poses)} total)")
    ax.view_init(elev=28, azim=35)

    # legend
    for r, c in zip(radii, colors):
        ax.scatter([], [], [], s=30, color=c, label=f"r={r:.3f}")
    ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_step3_gallery_pose(args):
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    elevations_deg = [float(v.strip()) for v in args.elevations_deg.split(",") if v.strip()]
    look_at  = _parse_vec3(args.look_at)
    up_hint  = _normalize(_parse_vec3(args.up_hint))

    # ── radius: 단일 float 또는 쉼표 구분 리스트 ──────────────────────────────
    radius_raw = str(args.radius).strip()
    if "," in radius_raw:
        radius_list = [float(v.strip()) for v in radius_raw.split(",") if v.strip()]
    else:
        radius_list = [float(radius_raw)]

    # ── roll: 단일 float 또는 쉼표 구분 리스트 ────────────────────────────────
    roll_raw = getattr(args, "roll_angles_deg", "0")
    if roll_raw is None:
        roll_list = [0.0]
    else:
        roll_str = str(roll_raw).strip()
        if "," in roll_str:
            roll_list = [float(v.strip()) for v in roll_str.split(",") if v.strip()]
        else:
            roll_list = [float(roll_str)]

    poses = generate_gallery_poses(
        elevations_deg   = elevations_deg,
        azimuth_step_deg = float(args.azimuth_step_deg),
        radius           = radius_list,
        look_at          = look_at,
        up_hint          = up_hint,
        roll_angles_deg  = roll_list,
    )

    payload = {
        "stage": "step3",
        "pose_convention": {
            "name": "opencv_like_obj_to_cam",
            "definition": "X_cam = R * X_obj + t, with camera axes x:right, y:down, z:forward",
            "camera_looks_toward": "look_at target in object frame",
        },
        "settings": {
            "elevations_deg":    elevations_deg,
            "azimuth_step_deg":  float(args.azimuth_step_deg),
            "radius_list":       radius_list,
            "roll_angles_deg":   roll_list,
            "look_at":           look_at.tolist(),
            "up_hint":           up_hint.tolist(),
        },
        "num_poses": len(poses),
        "poses":     poses,
    }

    json_path    = out_dir / "gallery_poses.json"
    csv_path     = out_dir / "gallery_poses.csv"
    preview_path = out_dir / "gallery_pose_preview.png"

    save_json(str(json_path), payload)
    save_pose_csv(csv_path, poses)
    save_pose_preview(preview_path, poses, preview_size=int(args.preview_size))

    summary_path = out_dir / "step3_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Step 3 complete: ring/elevation gallery pose generation\n")
        f.write(f"num_poses       : {len(poses)}\n")
        f.write(f"elevations_deg  : {elevations_deg}\n")
        f.write(f"azimuth_step_deg: {float(args.azimuth_step_deg)}\n")
        f.write(f"radius_list     : {radius_list}\n")
        f.write(f"roll_angles_deg : {roll_list}\n")
        f.write(f"look_at         : {look_at.tolist()}\n")
        f.write(f"up_hint         : {up_hint.tolist()}\n")
        f.write(f"json            : {json_path}\n")
        f.write(f"csv             : {csv_path}\n")
        f.write(f"preview         : {preview_path}\n")

    print("=" * 60)
    print("[Step 3] Gallery pose generation complete")
    print(f"  num_poses       : {len(poses)}")
    print(f"  radius_list     : {radius_list}")
    print(f"  roll_angles_deg : {roll_list}")
    print(f"  json            : {json_path}")
    print(f"  csv             : {csv_path}")
    print(f"  preview         : {preview_path}")
    print("=" * 60)