import json
import os
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_image(path, color=True):
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def save_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_txt_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def save_text(path, text):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)