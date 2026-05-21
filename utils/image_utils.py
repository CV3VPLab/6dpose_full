#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import cv2
import numpy as np


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def compute_nonblack_bbox(img_bgr: np.ndarray, thresh: int = 8):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > thresh)
    h, w = gray.shape
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, w, h
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return int(x1), int(y1), int(x2), int(y2)


def compute_bbox(mask: np.ndarray):
    rows_sum = np.sum(mask, axis=1)
    cols_sum = np.sum(mask, axis=0)
    y_indices = np.where(rows_sum > 0)[0]
    x_indices = np.where(cols_sum > 0)[0]
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    
    x1, x2 = x_indices[0], x_indices[-1] + 1
    y1, y2 = y_indices[0], y_indices[-1] + 1
    return x1, y1, x2, y2


def expand_bbox(bbox, margin, w, h):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    return int(x1), int(y1), int(x2), int(y2)


def extract_bbox(img: np.ndarray, margin: int, thresh: int = 8):
    bbox = compute_nonblack_bbox(img, thresh=thresh)
    bbox = expand_bbox(bbox, margin, w=img.shape[1], h=img.shape[0])
    return bbox


def crop_with_bbox(img, bbox):
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2].copy()


def extract_bbox_and_crop(img, margin=12, thresh=8):
    bbox = extract_bbox(img, margin=margin, thresh=thresh)
    crop = crop_with_bbox(img, bbox)
    return bbox, crop


def square_bbox(bbox, bbox_size):
    ax = bbox_size - (bbox[2] - bbox[0])
    ay = bbox_size - (bbox[3] - bbox[1])
    axl = ax // 2        
    ayl = ay // 2
    axr = ax - axl
    ayr = ay - ayl
    bbox_ext = [bbox[0]-axl, bbox[1]-ayl, bbox[2]+axr, bbox[3]+ayr]
    return bbox_ext


def get_max_bbox_size(bboxes):
    bbox_w = np.max( bboxes[:,2] - bboxes[:,0] )
    bbox_h = np.max( bboxes[:,3] - bboxes[:,1] )
    bbox_size = max(bbox_w, bbox_h)
    return bbox_size


def square_pad_resize(img: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0+h, x0:x0+w] = img
    out = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return out


def zeropad_square(img, side):
    h, w = img.shape[:2]
    assert h <= side and w <= side, "Image dimensions should be less than or equal to the specified side length"
    
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    pad_hw = side - img.shape[:2]
    sy, sx = pad_hw // 2
    ey, ex = img.shape[0] + sy, img.shape[1] + sx
    canvas[sy:ey, sx:ex] = img
    return canvas


def unmap_from_square_resize(pts_resized, orig_hw, resize_target):
    h, w = orig_hw
    side = max(h, w)
    x0 = (side - w) // 2
    y0 = (side - h) // 2
    pts_square = pts_resized * (side / resize_target)
    return pts_square - np.array([[x0, y0]], dtype = pts_resized.dtype)


def unmap_to_full_image(pts_crop, bbox, resize_target):
    pts_unscaled = unmap_from_square_resize(pts_crop, (bbox[3]-bbox[1], bbox[2]-bbox[0]), resize_target)
    return pts_unscaled + np.array([[bbox[0], bbox[1]]], dtype=pts_crop.dtype)


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR_RGB)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img
