#!/usr/bin/env python3
"""Compare estimated poses with GT after camera-coordinate conversion."""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import ConvexHull, distance


FRAME_ID_RE = re.compile(r"^(\d+)(?:_|$)")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_poses(path):
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"Pose file must contain a list: {path}")

    poses = {}
    for record in records:
        query = str(record["query"])
        match = FRAME_ID_RE.match(Path(query).stem)
        if match is None:
            raise ValueError(f"Cannot find six-digit frame id in query: {query}")

        frame_id = match.group(1)
        if frame_id in poses:
            raise ValueError(f"Duplicate frame id {frame_id} in {path}")

        rotation = np.asarray(record["R"], dtype=np.float64)
        translation = np.asarray(record["t"], dtype=np.float64).reshape(3)
        if rotation.shape != (3, 3):
            raise ValueError(f"Invalid R shape in {query}: {rotation.shape}")
        poses[frame_id] = {
            "query": query,
            "R": rotation,
            "t": translation,
        }
    return poses


def calibration_scale_to_meters(calibration, unit=None):
    unit = unit or calibration.get("pattern", {}).get("square_size_unit", "mm")
    scales = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
    if unit not in scales:
        raise ValueError(f"Unsupported calibration unit: {unit}")
    return scales[unit]


def reference_from_camera(calibration, camera, translation_unit=None):
    transform = np.asarray(
        calibration["extrinsics"]["by_stream"][camera]["matrix"],
        dtype=np.float64,
    ).copy()
    transform[:3, 3] *= calibration_scale_to_meters(calibration, translation_unit)
    return transform


def pose_in_reference(pose, transform):
    rotation_ref_cam = transform[:3, :3]
    translation_ref_cam = transform[:3, 3]
    return (
        rotation_ref_cam @ pose["R"],
        rotation_ref_cam @ pose["t"] + translation_ref_cam,
    )


def rotation_error_deg(rotation_est, rotation_gt):
    cosine = (np.trace(rotation_est @ rotation_gt.T) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def load_model_points(path, unit="m"):
    vertex = PlyData.read(str(path))["vertex"]
    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float64
    )
    points *= {"m": 1.0, "cm": 1e-2, "mm": 1e-3}[unit]
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError(f"Invalid model point cloud: {path}")
    return points


def model_diameter(points):
    """Return the exact maximum point distance, reduced to the convex hull."""
    try:
        hull_points = points[ConvexHull(points).vertices]
    except Exception:
        hull_points = points
    return float(np.max(distance.pdist(hull_points)))


def add_error(rotation_est, translation_est, rotation_gt, translation_gt, points):
    transformed_est = points @ rotation_est.T + translation_est
    transformed_gt = points @ rotation_gt.T + translation_gt
    return float(np.mean(np.linalg.norm(transformed_est - transformed_gt, axis=1)))


def evaluate(
    gt_poses,
    estimate_poses,
    gt_transform,
    estimate_transform,
    camera,
    model_points=None,
    add_threshold=None,
):
    frame_ids = sorted(set(gt_poses) & set(estimate_poses))
    if not frame_ids:
        raise ValueError(f"GT and {camera} estimate have no common frame ids")

    rows = []
    for frame_id in frame_ids:
        rotation_gt, translation_gt = pose_in_reference(gt_poses[frame_id], gt_transform)
        rotation_est, translation_est = pose_in_reference(
            estimate_poses[frame_id], estimate_transform
        )
        delta_mm = (translation_est - translation_gt) * 1000.0
        row = {
                "camera": camera,
                "frame_id": frame_id,
                "query": estimate_poses[frame_id]["query"],
                "rotation_error_deg": rotation_error_deg(rotation_est, rotation_gt),
                "translation_error_mm": float(np.linalg.norm(delta_mm)),
                "delta_x_mm": float(delta_mm[0]),
                "delta_y_mm": float(delta_mm[1]),
                "delta_z_mm": float(delta_mm[2]),
            }
        if model_points is not None:
            add_m = add_error(
                rotation_est,
                translation_est,
                rotation_gt,
                translation_gt,
                model_points,
            )
            row["add_mm"] = add_m * 1000.0
            row["add_0.1d_pass"] = add_m <= add_threshold
        rows.append(row)
    return rows


def evaluate_pose_files(
    gt_path,
    estimate_path,
    calibration_path,
    gt_camera,
    estimate_camera,
    output_dir,
    calibration_translation_unit=None,
    model_path=None,
    model_unit="m",
):
    calibration = load_json(calibration_path)
    gt_poses = load_poses(gt_path)
    estimate_poses = load_poses(estimate_path)
    gt_transform = reference_from_camera(
        calibration, gt_camera, calibration_translation_unit
    )
    model_points = load_model_points(model_path, model_unit) if model_path else None
    diameter_m = model_diameter(model_points) if model_points is not None else None
    rows = evaluate(
        gt_poses,
        estimate_poses,
        gt_transform,
        reference_from_camera(
            calibration, estimate_camera, calibration_translation_unit
        ),
        estimate_camera,
        model_points,
        0.1 * diameter_m if diameter_m is not None else None,
    )
    summary = {
        "reference_camera": calibration["extrinsics"]["reference_frame"],
        "gt_camera": gt_camera,
        "metric": {
            "rotation": "SO(3) geodesic error in degrees, without symmetry",
            "translation": "Euclidean object-origin error in millimeters",
        },
        "result": {
            "camera": estimate_camera,
            "pose_file": str(estimate_path),
            "rotation_error_deg": statistics([row["rotation_error_deg"] for row in rows]),
            "translation_error_mm": statistics([row["translation_error_mm"] for row in rows]),
            "missing_estimates": sorted(set(gt_poses) - set(estimate_poses)),
        },
    }
    if model_points is not None:
        add_values = [row["add_mm"] for row in rows]
        passed = sum(row["add_0.1d_pass"] for row in rows)
        summary["metric"]["add"] = (
            "Mean corresponding-point distance, without symmetry handling"
        )
        summary["result"]["add_mm"] = statistics(add_values)
        summary["result"]["add_0.1d"] = {
            "model_path": str(model_path),
            "model_diameter_mm": diameter_m * 1000.0,
            "threshold_mm": 0.1 * diameter_m * 1000.0,
            "passed": int(passed),
            "total": len(rows),
            "accuracy": float(passed / len(rows)),
            "accuracy_percent": float(100.0 * passed / len(rows)),
        }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_frame.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_json(output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--estimate", required=True)
    parser.add_argument("--gt-camera", required=True)
    parser.add_argument("--estimate-camera", required=True)
    parser.add_argument("--calibration", default="data/camera_calibration.json")
    parser.add_argument("--calibration-translation-unit", choices=("m", "cm", "mm"))
    parser.add_argument("--output-dir", default="data/pose_evaluation")
    parser.add_argument("--model-ply", help="Object model point cloud used for ADD")
    parser.add_argument("--model-unit", choices=("m", "cm", "mm"), default="m")
    args = parser.parse_args()
    summary = evaluate_pose_files(
        args.gt,
        args.estimate,
        args.calibration,
        args.gt_camera,
        args.estimate_camera,
        args.output_dir,
        args.calibration_translation_unit,
        args.model_ply,
        args.model_unit,
    )
    result = summary["result"]
    print(
        f"{result['camera']}: "
        f"rotation={result['rotation_error_deg']['mean']:.3f} deg, "
        f"translation={result['translation_error_mm']['mean']:.3f} mm"
    )
    if "add_0.1d" in result:
        add = result["add_0.1d"]
        print(
            f"ADD={result['add_mm']['mean']:.3f} mm, "
            f"ADD@0.1d={add['accuracy_percent']:.2f}% "
            f"({add['passed']}/{add['total']}, threshold={add['threshold_mm']:.3f} mm)"
        )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
