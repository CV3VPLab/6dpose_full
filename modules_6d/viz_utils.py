import cv2
import numpy as np


def draw_bbox(image, bbox, label=None, color=(0, 255, 0), thickness=2):
    out = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label is not None:
        cv2.putText(out, str(label), (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
    return out


def make_mask_overlay(image, mask, alpha=0.45):
    out = image.copy()
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    color = np.zeros_like(out)
    color[:, :, 1] = 255
    m = mask > 0
    out[m] = cv2.addWeighted(out, 1 - alpha, color, alpha, 0)[m]
    return out
