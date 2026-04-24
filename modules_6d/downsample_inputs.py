"""
Downsample a query image and produce a scaled intrinsics file.

Usage:
    python -m modules_6d.downsample_inputs \
        --query_img  data/query/q_00024.png \
        --out_img    data/query_ds/q_00024_ds.png \
        --intrinsics data/can_data/intrinsics.txt \
        --out_intrinsics data/can_data/intrinsics_ds.txt \
        --scale 0.5

The output image will be (original_w * scale) x (original_h * scale).
The output intrinsics will have fx, fy, cx, cy multiplied by scale.
"""

import argparse
import os
import cv2
import numpy as np


def downsample_image(src_path: str, dst_path: str, scale: float) -> tuple[int, int]:
    """Resize image by `scale` and save. Returns (out_w, out_h)."""
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {src_path}")
    h, w = img.shape[:2]
    out_w = int(round(w * scale))
    out_h = int(round(h * scale))
    resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    cv2.imwrite(dst_path, resized)
    print(f"[downsample] {src_path} ({w}x{h}) → {dst_path} ({out_w}x{out_h})")
    return out_w, out_h


def downsample_intrinsics(src_path: str, dst_path: str, scale: float) -> None:
    """Load intrinsics.txt (3x3 matrix), scale fx/fy/cx/cy, save."""
    vals = []
    with open(src_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                vals.extend([float(x) for x in line.split()])

    if len(vals) == 9:
        K = np.array(vals, dtype=np.float64).reshape(3, 3)
    elif len(vals) == 4:
        fx, fy, cx, cy = vals
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported intrinsics format in: {src_path}")

    # Scale focal lengths and principal point
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale  # fx
    K_scaled[1, 1] *= scale  # fy
    K_scaled[0, 2] *= scale  # cx
    K_scaled[1, 2] *= scale  # cy

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    with open(dst_path, "w") as f:
        for row in K_scaled:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

    print(f"[downsample] intrinsics {src_path} → {dst_path}  (scale={scale})")
    print(f"  fx={K_scaled[0,0]:.3f}  fy={K_scaled[1,1]:.3f}  "
          f"cx={K_scaled[0,2]:.3f}  cy={K_scaled[1,2]:.3f}")


def main():
    p = argparse.ArgumentParser(description="Downsample query image + intrinsics")
    p.add_argument("--query_img",       required=True)
    p.add_argument("--out_img",         required=True)
    p.add_argument("--intrinsics",      required=True)
    p.add_argument("--out_intrinsics",  required=True)
    p.add_argument("--scale",           type=float, default=0.5)
    args = p.parse_args()

    downsample_image(args.query_img, args.out_img, args.scale)
    downsample_intrinsics(args.intrinsics, args.out_intrinsics, args.scale)


if __name__ == "__main__":
    main()
