from pathlib import Path
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _extract_rgb_from_record(data) -> np.ndarray:
    names = list(data.dtype.names)

    # Common RGB property names
    rgb_names = [("red", "green", "blue"), ("r", "g", "b")]
    for rn in rgb_names:
        if all(n in names for n in rn):
            rgb = np.stack([data[rn[0]], data[rn[1]], data[rn[2]]], axis=1).astype(np.float32)
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            return np.clip(rgb, 0.0, 1.0)

    # 3DGS-style SH DC coefficients
    if all(n in names for n in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        c0 = 0.28209479177387814
        sh = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1).astype(np.float32)
        rgb = np.clip(0.5 + c0 * sh, 0.0, 1.0)
        return rgb

    # Fallback: grayscale
    n = len(data)
    return np.full((n, 3), 0.7, dtype=np.float32)


def _extract_alpha_from_record(data) -> np.ndarray:
    names = list(data.dtype.names)
    if "opacity" in names:
        alpha = _sigmoid(data["opacity"].astype(np.float32))
        return np.clip(alpha, 0.0, 1.0)
    return np.ones((len(data),), dtype=np.float32)


def load_ply_points(ply_path: str):
    try:
        from plyfile import PlyData
    except Exception as e:
        raise ImportError(
            "plyfile is required for step3 preview rendering. Install with: pip install plyfile"
        ) from e

    ply_path = str(Path(ply_path))
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"].data
    names = list(vertex.dtype.names)
    required = ["x", "y", "z"]
    for n in required:
        if n not in names:
            raise ValueError(f"PLY missing required vertex field: {n}")

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    rgb = _extract_rgb_from_record(vertex)
    alpha = _extract_alpha_from_record(vertex)
    return xyz, rgb, alpha
