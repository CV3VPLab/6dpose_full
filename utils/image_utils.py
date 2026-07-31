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

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
from .io_utils import load_json
import matplotlib.pyplot as plt


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

    if isinstance(img, torch.Tensor):
        if img.dim() == 3:
            return img[:, y1:y2, x1:x2].clone()
        elif img.dim() == 2:
            return img[y1:y2, x1:x2].clone()
        else:
            raise ValueError(f"Unsupported tensor shape: {img.shape}")
    else:
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


def get_bbox_size(bbox):
    return bbox[2]-bbox[0], bbox[3]-bbox[1]

 
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
    if not isinstance(side, np.ndarray):
        side = np.array(side)
    pad_hw = side - img.shape[:2]
    sy, sx = pad_hw // 2
    ey, ex = img.shape[0] + sy, img.shape[1] + sx
    canvas[sy:ey, sx:ex] = img
    return canvas, (sx, sy)


def unmap_from_square_resize(pts_resized, orig_hw, resized_size):
    h, w = orig_hw
    side = max(h, w)
    x0 = (side - w) // 2
    y0 = (side - h) // 2
    pts_square = pts_resized * (side / resized_size)
    return pts_square - np.array([[x0, y0]], dtype = pts_resized.dtype)


def unmap_to_full_image(pts_crop, bbox):
    return pts_crop + np.array([[bbox[0], bbox[1]]], dtype=pts_crop.dtype)


def get_specular_mask(img_rgb):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lower_bound = np.array([0, 0, 200])
    upper_bound = np.array([180, 40, 255])
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def apply_mask(image, mask):
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    out = np.zeros_like(image)
    out[mask > 0] = image[mask > 0]
    return out


def get_mask_inlier_indices(pts, mask):
    """
    원본 좌표가 마스크 내부에 있는지 확인하여 True/False 리스트 반환
    """
    h, w = mask.shape
    keep_mask = []
    for pt in pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            keep_mask.append(True)
        else:
            keep_mask.append(False)
            
    return np.array(keep_mask)


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR_RGB)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


def load_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR_BGR)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


def construct_queryInfo(query_img, out_dir):
    q_path = Path(query_img) 
    assert q_path.exists(), print(f"  [ERROR] query image [{query_img}] not found")

    query_full = load_rgb(str(q_path))
    
    mask_path = out_dir / f"q_mask_{q_path.stem}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        assert False, f"Query mask not found at: {mask_path}"

    q_bbox = compute_bbox(mask)

    qh, qw = query_full.shape[:2]
    print(f"  query full image: {qw}x{qh}")
    print(f"  mask for the query image: {mask.shape[1]}x{mask.shape[0]}")
    print(f"  query bbox: {q_bbox}, (width={q_bbox[2]-q_bbox[0]}, height={q_bbox[3]-q_bbox[1]})")

    return {
        "rgb": query_full,      # full query image (RGB)
                                # KSCHOI TODO: currently, masked query image is used as full image, 
                                # but ideally should be original unmasked RGB query
        "mask": mask,           # full query mask (grayscale, 0=background, 255=foreground)
        "bbox": q_bbox          # query bounding box
    }


def construct_galleryInfo(out_dir):
    bboxes_path = out_dir / "g_bboxes.npy"
    assert bboxes_path.exists(), f"Gallery bounding boxes not found at: {bboxes_path}"
    g_bboxes = np.load(str(bboxes_path))
    
    print(f"  DINOv2 cache dir: {out_dir}")

    gallery_feats = np.load( str(out_dir / "g_features.npy") )
    g_feats = torch.from_numpy(gallery_feats) # (N_gallery, D_feat) tensor

    # Cropped gallery images load
    gallery_crops = []
    for idx in tqdm(range(len(g_bboxes)), desc="Loading gallery"):
        gpath = out_dir / "gallery" / f"{idx:04d}.png"
        img_rgb = load_rgb(str(gpath))
        gallery_crops.append(img_rgb)

    bbox_size = get_max_bbox_size(g_bboxes)

    gallery_poses = load_json(out_dir.parent.parent / "gallery_poses.json")["poses"]

    return {
        "crops": gallery_crops,         # cropped gallery images according to g_bboxes
        "poses": gallery_poses,
        "bboxes": g_bboxes,             # all the bounding boxes of gallery renders
        "bbox_size": bbox_size,         # the union bounding box
        "feats": g_feats,               # DINOv2 features of gallery crops, used for cosine similarity retrieval 
        "path": out_dir / "gallery" 
    }


def make_gallery_square(galleryInfo, idx, size):
    g_bbox = galleryInfo["bboxes"][idx]
    gallery_crop = galleryInfo["crops"][idx]
    g_crop_hw = gallery_crop.shape[:2]
    assert g_bbox[2] - g_bbox[0] == g_crop_hw[1] and g_bbox[3] - g_bbox[1] == g_crop_hw[0], f"Gallery crop size mismatch with bbox, check loading and cropping logic for gallery item {item}"
    assert size >= g_crop_hw[0] and size >= g_crop_hw[1]

    g_crop_ext, scoord = zeropad_square(gallery_crop, size)
    g_bbox_ext = g_bbox[:2] - scoord
    g_bbox_ext = np.array( list(g_bbox_ext) + list(g_bbox_ext + size) )
    return g_crop_ext, g_bbox_ext


def render_to_image(render):
    return (render.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)


def np_to_tensor(innp: np.ndarray, device='cuda'):
    resT = torch.from_numpy(innp).to(device).float()
    if innp.ndim == 3:
        assert innp.shape[2] == 1 or innp.shape[2] == 3
        resT = resT.permute(2,0,1)
    return resT

        
def tensor_to_np(tensor: torch.Tensor):
    tensor = tensor.detach().cpu()
    if tensor.dim() == 4:
        assert tensor.shape[0] == 1, "we assume that a tensor of bchw format has a single batch size"
        tensor = tensor.squeeze(0)    
    if tensor.dim() == 3:
        tensor = tensor.permute(1,2,0)
    return tensor.numpy()


def npfloat_to_image(innp: np.ndarray):
    assert innp.dtype.kind == 'f'
    if np.min(innp) < 0:    # float
        innp = innp + 0.5    
    return (innp * 255.0).astype(np.uint8)


def tensor_to_image(tensor: torch.Tensor):
    return npfloat_to_image(tensor_to_np(tensor))


def imshow_tensor(tensor: torch.Tensor):
    plt.imshow( tensor_to_np(tensor) )


def get_boundary(binimg: np.ndarray):
    if not hasattr(get_boundary, "kernel"):
        get_boundary.kernel = np.ones((3, 3), np.uint8)

    eimg = cv2.erode(binimg, get_boundary.kernel, iterations=1)
    return binimg - eimg


def bin_to_color(binimg: np.ndarray, color):
    assert len(color) == 3
    resimg = np.zeros((binimg.shape[0], binimg.shape[1], 3), dtype=binimg.dtype)
    resimg[:,:,0] = binimg * color[0]
    resimg[:,:,1] = binimg * color[1]
    resimg[:,:,2] = binimg * color[2]
    return resimg


def scale_image_draw_maskcontour(img: np.ndarray, scale, mask: np.ndarray, color):
    scaledimg = (img.astype(np.float32) * scale).astype(np.uint8)
    contours, _ = cv2.findContours( mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    resimg = cv2.drawContours(scaledimg, contours, -1, tuple(color), 1)
    return resimg


def draw_crop_matches(gcrop, qcrop, gbbox, qbbox, pts0, pts1, conf, out_path):
    gimg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    qimg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    gimg[gbbox[1]:gbbox[3], gbbox[0]:gbbox[2]] = gcrop
    qimg[qbbox[1]:qbbox[3], qbbox[0]:qbbox[2]] = qcrop
    match_img = draw_matches(qimg, gimg, pts0, pts1, conf, out_path)
    return match_img


def draw_matches(img0, img1, mkpts0, mkpts1, conf, out_path, max_draw=400):
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    H = max(h0, h1)
    canvas = np.zeros((H, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:] = img1
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    idx = np.arange(len(mkpts0))
    if len(idx) > max_draw:
        idx = np.argsort(-conf)[:max_draw]

    for i in idx:
        p0 = tuple(np.round(mkpts0[i]).astype(int))
        p1 = tuple(np.round(mkpts1[i]).astype(int) + np.array([w0, 0]))
        color = (0, 255, 0)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, (0, 255, 255), -1)
        cv2.circle(canvas, p1, 2, (0, 255, 255), -1)

    if out_path is not None:
        cv2.imwrite(str(out_path), canvas)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def draw_contour(img: np.ndarray, mask: np.ndarray, color=(0,255,0), thickness=1):
    if mask.dtype is not np.uint8:
        mask = mask.astype(np.uint8) * 255        
    contour, _ = cv2.findContours( mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cv2.drawContours(img, contour, -1, color, thickness)


def get_gradient_filters(device='gpu'):
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [0., 0., 0.],
                            [1., 2., 1.]], device=device).view(1, 1, 3, 3)
    sobel_x = sobel_x.repeat(1, 1, 1, 1)
    sobel_y = sobel_y.repeat(1, 1, 1, 1)
    return sobel_x, sobel_y


def compute_gradients(image, filters_x, filters_y):
    if image.shape[1] > 1:
        image = image.mean(dim=1, keepdim=True)
    grad_x = F.conv2d(image, filters_x, padding=1)
    grad_y = F.conv2d(image, filters_y, padding=1)

    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    # grad_dir = torch.cat([grad_x, grad_y], dim=1) / (grad_mag + 1e-6)

    return grad_mag #, grad_dir


def erode_binary_tensor(img_tensor, kernel_size=3):
    """
    img_tensor: (B, C, H, W) 형태의 0과 1로 이루어진 이진 텐서
    """
    pad = kernel_size // 2
    
    inv_tensor = 1.0 - img_tensor
    dilated_inv = F.max_pool2d(inv_tensor, kernel_size, stride=1, padding=pad)
    eroded_tensor = 1.0 - dilated_inv
    
    return eroded_tensor


def dice_loss(pred, target, smooth=1e-6):
    """
    pred: 모델의 출력값 (Logits). 형태는 보통 (N, C, H, W)
    target: 정답 레이블 (0 또는 1). 형태는 pred와 동일
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice_score = (2. * intersection + smooth) / (union + smooth)
    return 1. - dice_score


def iou_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    total = pred.sum() + target.sum()
    union = total - intersection # A U B = A + B - (A ∩ B)
    
    iou_score = (intersection + smooth) / (union + smooth)
    return 1. - iou_score


def gradient_matching_loss(render_img, query_img, mask=None):
    if not hasattr(gradient_matching_loss, "filters_x"):
        device = render_img.device
        filters_x, filters_y = get_gradient_filters(device)
        
    emask = erode_binary_tensor(mask, 3)
    render_grad_mag = compute_gradients(render_img, filters_x, filters_y)
    query_grad_mag = compute_gradients(query_img, filters_x, filters_y)
    
    mag_loss = F.l1_loss(render_grad_mag * emask, query_grad_mag * emask)    

    return mag_loss


