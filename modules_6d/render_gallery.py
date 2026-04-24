import json
from math import ceil, sqrt
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import ensure_dir, save_json
from .ply_utils import load_ply_points


def _parse_bg_color(text: str):
    vals = [int(v.strip()) for v in text.split(",")]
    if len(vals) != 3:
        raise ValueError(f"Expected R,G,B for bg_color, got: {text}")
    return tuple(max(0, min(255, v)) for v in vals)


def load_intrinsics(path: str) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            vals.extend([float(p) for p in parts])

    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 9:
        K = arr.reshape(3, 3)
    elif arr.size == 4:
        fx, fy, cx, cy = arr.tolist()
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        raise ValueError(
            f"Intrinsics file must contain 9 values (3x3) or 4 values (fx fy cx cy). Got {arr.size}."
        )
    return K


def subsample_points(xyz, rgb, alpha, max_points: int):
    n = xyz.shape[0]
    if n <= max_points:
        return xyz, rgb, alpha
    idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
    return xyz[idx], rgb[idx], alpha[idx]


def project_points(xyz_obj, R, t, K):
    xyz_cam = (R @ xyz_obj.T).T + t.reshape(1, 3)
    z = xyz_cam[:, 2]
    valid = z > 1e-6
    xyz_cam = xyz_cam[valid]
    z = z[valid]
    if xyz_cam.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), valid
    uvw = (K @ xyz_cam.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv.astype(np.float32), z.astype(np.float32), valid


def render_preview_ply(xyz_obj, rgb, alpha, R, t, K, width, height, point_size=2, bg_color=(0, 0, 0)):
    image = np.full((height, width, 3), bg_color, dtype=np.uint8)
    uv, depth, valid = project_points(xyz_obj, R, t, K)
    rgb_v = rgb[valid]
    alpha_v = alpha[valid]

    inside = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    uv = uv[inside]
    depth = depth[inside]
    rgb_v = rgb_v[inside]
    alpha_v = alpha_v[inside]

    if uv.shape[0] == 0:
        return image

    order = np.argsort(depth)[::-1]  # far -> near, nearer points overwrite later
    uv = uv[order]
    rgb_v = rgb_v[order]
    alpha_v = alpha_v[order]

    radius = max(1, int(point_size))
    for (u, v), c, a in zip(uv, rgb_v, alpha_v):
        color = tuple(int(round(float(x) * 255.0)) for x in c[::-1])  # BGR for OpenCV
        center = (int(round(float(u))), int(round(float(v))))
        if radius <= 1:
            if 0 <= center[0] < width and 0 <= center[1] < height:
                image[center[1], center[0]] = color
        else:
            cv2.circle(image, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    return image


def make_contact_sheet(image_paths, out_path: Path, thumb_w=220, thumb_h=220, text_height=24):
    n = len(image_paths)
    cols = int(ceil(sqrt(n)))
    rows = int(ceil(n / cols))
    sheet = np.zeros((rows * (thumb_h + text_height), cols * thumb_w, 3), dtype=np.uint8)

    for idx, p in enumerate(image_paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_r = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        r = idx // cols
        c = idx % cols
        y0 = r * (thumb_h + text_height)
        x0 = c * thumb_w
        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = img_r
        cv2.putText(sheet, p.stem, (x0 + 6, y0 + thumb_h + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), sheet)


def save_camera_axis_debug(poses_payload, out_path: Path):
    poses = poses_payload["poses"]
    fig = plt.figure(figsize=(8, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([0], [0], [0], s=80, marker="x", color="tab:orange")

    for p in poses:
        c = np.asarray(p["camera_position_obj_frame"], dtype=np.float64)
        R = np.asarray(p["R_obj_to_cam"], dtype=np.float64)
        # camera forward in object frame = R^T [0,0,1]
        fwd_obj = R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        ax.scatter([c[0]], [c[1]], [c[2]], s=25, color="tab:blue")
        ax.quiver(c[0], c[1], c[2], fwd_obj[0], fwd_obj[1], fwd_obj[2],
                  length=0.18, normalize=True, color="tab:red")
        ax.text(c[0], c[1], c[2], str(p["index"]), fontsize=7)

    ax.set_title("Step3 camera positions + forward axes")
    ax.set_xlabel("X_obj")
    ax.set_ylabel("Y_obj")
    ax.set_zlabel("Z_obj")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_step3_render_gallery(args):
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    render_dir = out_dir / "gallery_renders"
    ensure_dir(str(render_dir))

    pose_json = Path(args.gallery_pose_json) if args.gallery_pose_json else out_dir / "gallery_poses.json"
    if not pose_json.exists():
        raise FileNotFoundError(f"gallery pose json not found: {pose_json}")
    if args.model_path is None:
        raise ValueError("--model_path is required for step3")
    if args.intrinsics_path is None:
        raise ValueError("--intrinsics_path is required for step3")

    with open(pose_json, "r", encoding="utf-8") as f:
        pose_payload = json.load(f)

    K = load_intrinsics(args.intrinsics_path)
    xyz, rgb, alpha = load_ply_points(args.model_path)
    xyz, rgb, alpha = subsample_points(xyz, rgb, alpha, int(args.max_points))

    width = int(args.render_width)
    height = int(args.render_height)
    bg_color = _parse_bg_color(args.bg_color)

    image_paths = []
    render_meta = {
        "stage": "step3",
        "backend": args.render_backend,
        "gallery_pose_json": str(pose_json),
        "model_path": str(args.model_path),
        "intrinsics_path": str(args.intrinsics_path),
        "num_input_points": int(xyz.shape[0]),
        "render_width": width,
        "render_height": height,
        "point_size": int(args.point_size),
        "bg_color": list(bg_color),
        "renders": [],
        "note": "preview_ply is a lightweight point-preview renderer for debugging view generation, not final 3DGS quality.",
    }

    for p in pose_payload["poses"]:
        idx = int(p["index"])
        R = np.asarray(p["R_obj_to_cam"], dtype=np.float64)
        t = np.asarray(p["t_obj_to_cam"], dtype=np.float64)

        img = render_preview_ply(
            xyz_obj=xyz,
            rgb=rgb,
            alpha=alpha,
            R=R,
            t=t,
            K=K,
            width=width,
            height=height,
            point_size=int(args.point_size),
            bg_color=bg_color,
        )
        out_path = render_dir / f"{idx:04d}.png"
        cv2.imwrite(str(out_path), img)
        image_paths.append(out_path)
        render_meta["renders"].append({
            "index": idx,
            "image_path": str(out_path),
            "azimuth_deg": p["azimuth_deg"],
            "elevation_deg": p["elevation_deg"],
        })

    contact_sheet_path = out_dir / "gallery_contact_sheet.png"
    make_contact_sheet(image_paths, contact_sheet_path)

    axis_debug_path = out_dir / "gallery_pose_axes_preview.png"
    save_camera_axis_debug(pose_payload, axis_debug_path)

    meta_path = out_dir / "gallery_render_meta.json"
    save_json(str(meta_path), render_meta)

    summary_path = out_dir / "step3_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Step 3 complete: gallery preview rendering\n")
        f.write(f"backend: {args.render_backend}\n")
        f.write(f"gallery_pose_json: {pose_json}\n")
        f.write(f"model_path: {args.model_path}\n")
        f.write(f"intrinsics_path: {args.intrinsics_path}\n")
        f.write(f"num_renders: {len(image_paths)}\n")
        f.write(f"render_dir: {render_dir}\n")
        f.write(f"contact_sheet: {contact_sheet_path}\n")
        f.write(f"axis_debug: {axis_debug_path}\n")
        f.write(f"meta: {meta_path}\n")

    print("=" * 60)
    print("[Step 3] Gallery preview rendering complete")
    print(f"  num_renders   : {len(image_paths)}")
    print(f"  render_dir    : {render_dir}")
    print(f"  contact_sheet : {contact_sheet_path}")
    print(f"  axis_debug    : {axis_debug_path}")
    print(f"  meta          : {meta_path}")
    print("=" * 60)
