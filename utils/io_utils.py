import json
import os
from pathlib import Path

import cv2
import numpy as np


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


def read_txt_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def save_text(path, text):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


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


def K_to_params(K):
    return K[0, 0], K[1, 1], K[0, 2], K[1, 2]


def params_to_K(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def get_obj_path(obj_name, kind):
    # kind : {"output", "object", "gallery", "xyz", "model"}
    if kind == "output":
        return Path("data/output") / obj_name
    
    out_path = Path("data/object") / obj_name
    if kind == "object":
        return out_path 
    
    return out_path / kind
    

def get_K_path(config):
    return Path("data/camera") / config["cam_type"] / config["K_filename"]


def search_for_max_iteration(point_cloud_dir: Path):
    iters = []
    for p in point_cloud_dir.iterdir():
        if p.is_dir() and p.name.startswith("iteration_"):
            try:
                iters.append(int(p.name.split("_")[-1]))
            except Exception:
                pass
    if not iters:
        raise FileNotFoundError(f"No iteration_* dirs in {point_cloud_dir}")
    return max(iters)


def resolve_ply_path(model_dir: Path):
    ply = model_dir / "point_cloud.ply"
    if not ply.exists():
        raise FileNotFoundError(f"point_cloud.ply not found: {ply}")
    return ply


def load_image(path, color=True):
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def get_named_config(configSet):
    configName = configSet["name"]
    configList = configSet["configs"]
    for config in configList:
        if config["name"] == configName:
            return config        
    return None
