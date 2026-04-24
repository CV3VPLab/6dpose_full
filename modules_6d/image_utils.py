import cv2
import numpy as np


def expand_bbox(bbox, margin, width, height):
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - margin)
    y1 = max(0, int(y1) - margin)
    x2 = min(width - 1, int(x2) + margin)
    y2 = min(height - 1, int(y2) + margin)
    return [x1, y1, x2, y2]


def crop_from_bbox(image, bbox):
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2].copy()


def apply_mask(image, mask):
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    out = np.zeros_like(image)
    out[mask > 0] = image[mask > 0]
    return out


def crop_mask_to_bbox(mask, bbox):
    x1, y1, x2, y2 = bbox
    return mask[y1:y2, x1:x2].copy()
