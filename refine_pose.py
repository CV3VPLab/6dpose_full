"""
refine_pose.py  —  GS-Pose style pose refinement (v5)
======================================================
변경 사항:
  - Identity camera (canonical 좌표계, full resolution) 유지
  - t_can 기반 projected_bbox_from_pose → differentiable crop
  - query도 동일 bbox로 crop (numpy)
  - loss: D-SSIM + D-MS-SSIM (GS-Pose 방식, L1 제거)
  - query_mask: query crop의 non-black만 loss에 반영
  - optimizer: AdamW + CosineAnnealing + warmup + early stopping


greate result in unmasked query
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from gaussian_renderer import GaussianModel
from gsplat import rasterization as _gsplat_rasterize_3dgs
from gsplat import rasterization_2dgs as _gsplat_rasterize_2dgs

from utils.image_utils import construct_queryInfo, get_bbox_size, square_bbox, crop_with_bbox, apply_mask
from modules_6d.io_utils import (
    ensure_dir, save_json, load_json, 
    load_intrinsics, K_to_params, 
    resolve_ply_path
)

def _rasterize(means, quats, scales, opacities, colors, viewmats, Ks,
               width, height, sh_degree, near_plane, far_plane, backgrounds, packed):
    """Unified rasterizer: auto-detects 2DGS (scales dim=2) vs 3DGS (scales dim=3)."""
    if scales.shape[-1] == 2:
        pad = torch.full((*scales.shape[:-1], 1), 1e-10,
                         dtype=scales.dtype, device=scales.device)
        scales_3 = torch.cat([scales, pad], dim=-1)
        out = _gsplat_rasterize_2dgs(
            means=means, quats=quats, scales=scales_3, opacities=opacities,
            colors=colors, viewmats=viewmats, Ks=Ks,
            width=width, height=height, sh_degree=sh_degree,
            near_plane=near_plane, far_plane=far_plane,
            backgrounds=backgrounds, packed=packed,
        )
        return out[0], out[1], out[-1]   # renders, alphas, meta
    else:
        return _gsplat_rasterize_3dgs(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=viewmats, Ks=Ks,
            width=width, height=height, sh_degree=sh_degree,
            near_plane=near_plane, far_plane=far_plane,
            backgrounds=backgrounds, packed=packed,
        )


# ──────────────────────────────────────────────────────────
# Arg parsing & IO utilities
# ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Refine object pose – GS-Pose style crop")
    p.add_argument("--gs_model_dir",        required=True)
    p.add_argument("--out_dir",             required=True)
    p.add_argument("--query_img",           required=True)
    p.add_argument("--intrinsics_path",     required=True)
    p.add_argument("--width",               required=True, type=int)
    p.add_argument("--height",              required=True, type=int)
    p.add_argument("--background",          default="0,0,0")    
    p.add_argument("--sh_degree",           default=3,   type=int)
    # optimizer
    p.add_argument("--iters",               default=100, type=int)
    p.add_argument("--lr_rot",              default=1e-2, type=float)
    p.add_argument("--lr_trans",            default=5e-3, type=float)
    p.add_argument("--warmup_steps",        default=10,  type=int)
    p.add_argument("--early_stop_steps",    default=20,  type=int,
                   help="Early stop when loss grad norm (last N steps) < threshold")
    p.add_argument("--early_stop_thresh",   default=1e-5, type=float)
    # # crop
    p.add_argument("--crop_size",           default=320, type=int,
                   help="Square crop target size for both query and render")
    p.add_argument("--crop_margin_scale",   default=1.3, type=float,
                   help="Margin factor around mask bbox (1.0 = tight)")
    p.add_argument("--rt_mode", action="store_true",
                   help="Skip all debug image saves; output only refined_pose.json")
    return p.parse_args()






# ──────────────────────────────────────────────────────────
# Mask bbox & crop utilities
# ──────────────────────────────────────────────────────────
def get_mask_bbox(mask_gray: np.ndarray):
    """binary mask(H,W uint8)에서 tight bbox (x1,y1,x2,y2) 반환."""
    ys, xs = np.where(mask_gray > 0)
    if len(xs) == 0:
        h, w = mask_gray.shape
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1


def bbox_to_square_with_margin(bbox, img_w, img_h, margin_scale=1.3):
    """
    tight bbox를 margin을 포함한 정사각형 bbox로 확장.
    반환: (cx, cy, side)  — 중심과 한 변의 길이
    """
    assert type(bbox) == np.ndarray
    cx, cy = ( bbox[:2] + bbox[2:] ) / 2.0
    bw, bh = bbox[2:] - bbox[:2]
    side = max(bw, bh) * float(margin_scale)
    # 이미지 경계 클램프
    half = side / 2.0
    cx = float(np.clip(cx, half, img_w - half))
    cy = float(np.clip(cy, half, img_h - half))
    side = float(side)
    return cx, cy, side


def crop_center_size(img: np.ndarray, cx, cy, side):
    """
    img_bgr (H,W,3) uint8을 (cx,cy) 중심의 side×side 영역으로 crop 후
    target_size×target_size로 resize.
    경계 밖은 0으로 padding.
    """
    h, w = img.shape[:2]
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + int(round(side))
    y2 = y1 + int(round(side))

    # padding을 이용한 안전 crop
    pad_left  = max(0, -x1)
    pad_top   = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bot   = max(0, y2 - h)

    x1c, y1c = x1 + pad_left, y1 + pad_top
    x2c, y2c = x2 + pad_left, y2 + pad_top

    canvas_w = w + pad_left + pad_right
    canvas_h = h + pad_top  + pad_bot
    if img.ndim == 3:
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=img.dtype)
    elif img.ndim == 2:
        canvas = np.zeros((canvas_h, canvas_w), dtype=img.dtype)
    else:
        assert False
    canvas[pad_top:pad_top+h, pad_left:pad_left+w] = img

    cropped = canvas[y1c:y2c, x1c:x2c]
    return cropped


def crop_and_resize(img_bgr: np.ndarray, cx, cy, side, target_size: int):
    """
    img_bgr (H,W,3) uint8을 (cx,cy) 중심의 side×side 영역으로 crop 후
    target_size×target_size로 resize.
    경계 밖은 0으로 padding.
    """
    h, w = img_bgr.shape[:2]
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + int(round(side))
    y2 = y1 + int(round(side))

    # padding을 이용한 안전 crop
    pad_left  = max(0, -x1)
    pad_top   = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bot   = max(0, y2 - h)

    x1c, y1c = x1 + pad_left, y1 + pad_top
    x2c, y2c = x2 + pad_left, y2 + pad_top

    canvas_w = w + pad_left + pad_right
    canvas_h = h + pad_top  + pad_bot
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=img_bgr.dtype)
    canvas[pad_top:pad_top+h, pad_left:pad_left+w] = img_bgr

    cropped = canvas[y1c:y2c, x1c:x2c]
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return resized


def crop_and_resize_gray(mask_gray: np.ndarray, cx, cy, side, target_size: int):
    """단채널 mask용 crop_and_resize."""
    h, w = mask_gray.shape
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + int(round(side))
    y2 = y1 + int(round(side))

    pad_left  = max(0, -x1)
    pad_top   = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bot   = max(0, y2 - h)

    x1c, y1c = x1 + pad_left, y1 + pad_top
    x2c, y2c = x2 + pad_left, y2 + pad_top

    canvas = np.zeros((h + pad_top + pad_bot, w + pad_left + pad_right), dtype=mask_gray.dtype)
    canvas[pad_top:pad_top+h, pad_left:pad_left+w] = mask_gray

    cropped = canvas[y1c:y2c, x1c:x2c]
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return resized


# ──────────────────────────────────────────────────────────
# Identity camera (canonical 좌표계, full resolution)
# Gaussians을 RigidPoseGaussianProxy로 변환하고 여기서 렌더
# ──────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────
# Differentiable crop (t_cur 기반 bbox → grid_sample)
# ──────────────────────────────────────────────────────────
def projected_bbox_from_pose(t_obj_to_cam, fx, fy, cx, cy,
                             obj_height, obj_diameter,
                             img_w, img_h, margin_scale=1.3):
    tx, ty, tz = t_obj_to_cam[0], t_obj_to_cam[1], t_obj_to_cam[2]
    tz = torch.clamp(tz, min=1e-4)
    u = fx * (tx / tz) + cx
    v = fy * (ty / tz) + cy
    h_px    = fy * (obj_height / tz)
    w_px    = fx * (obj_diameter / tz)
    half    = 0.5 * torch.maximum(h_px, w_px) * margin_scale
    x1, x2 = u - half, u + half
    y1, y2 = v - half, v + half
    x1 = torch.clamp(x1, min=0.0, max=float(img_w - 2))
    y1 = torch.clamp(y1, min=0.0, max=float(img_h - 2))
    x2 = torch.clamp(torch.maximum(x2, x1 + 1.0), max=float(img_w - 1))
    y2 = torch.clamp(torch.maximum(y2, y1 + 1.0), max=float(img_h - 1))
    return x1, y1, x2, y2


def crop_chw_with_bbox(render_chw, bbox):
    x1, y1, x2, y2 = bbox
    crop = render_chw[:, y1:y2, x1:x2]
    return crop


def crop_resize_chw_by_bbox(render_chw, bbox, out_size=320):
    C, H, W = render_chw.shape
    x1, y1, x2, y2 = bbox
    side    = max(x2 - x1, y2 - y1)
    cx_box  = (x1 + x2) * 0.5
    cy_box  = (y1 + y2) * 0.5
    sq_x1   = cx_box - side * 0.5
    sq_y1   = cy_box - side * 0.5
    sq_x2   = cx_box + side * 0.5
    sq_y2   = cy_box + side * 0.5
    xs = torch.linspace(float(sq_x1), float(sq_x2), out_size, device=render_chw.device)
    ys = torch.linspace(float(sq_y1), float(sq_y2), out_size, device=render_chw.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    gx   = (xx / (W - 1)) * 2 - 1
    gy   = (yy / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    crop = F.grid_sample(render_chw.unsqueeze(0), grid,
                         mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(0)
    return crop


# ──────────────────────────────────────────────────────────
# Math utilities
# ──────────────────────────────────────────────────────────
def so3_exp_map(w):
    theta = torch.norm(w) + 1e-12
    wx, wy, wz = w[0], w[1], w[2]
    K = torch.stack([
        torch.stack([torch.tensor(0.0, device=w.device), -wz,  wy]),
        torch.stack([wz,  torch.tensor(0.0, device=w.device), -wx]),
        torch.stack([-wy,  wx, torch.tensor(0.0, device=w.device)])
    ])
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    A = torch.sin(theta) / theta
    B = (1.0 - torch.cos(theta)) / (theta * theta)
    return I + A * K + B * (K @ K)


def rotation_matrix_to_quaternion_wxyz_torch(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    S0 = torch.sqrt((tr + 1.0).clamp(min=1e-10)) * 2.0
    qw0 = 0.25 * S0; qx0 = (R[2,1]-R[1,2])/S0; qy0 = (R[0,2]-R[2,0])/S0; qz0 = (R[1,0]-R[0,1])/S0
    S1 = torch.sqrt((1.0+R[0,0]-R[1,1]-R[2,2]).clamp(min=1e-10)) * 2.0
    qw1 = (R[2,1]-R[1,2])/S1; qx1 = 0.25*S1; qy1 = (R[0,1]+R[1,0])/S1; qz1 = (R[0,2]+R[2,0])/S1
    S2 = torch.sqrt((1.0+R[1,1]-R[0,0]-R[2,2]).clamp(min=1e-10)) * 2.0
    qw2 = (R[0,2]-R[2,0])/S2; qx2 = (R[0,1]+R[1,0])/S2; qy2 = 0.25*S2; qz2 = (R[1,2]+R[2,1])/S2
    S3 = torch.sqrt((1.0+R[2,2]-R[0,0]-R[1,1]).clamp(min=1e-10)) * 2.0
    qw3 = (R[1,0]-R[0,1])/S3; qx3 = (R[0,2]+R[2,0])/S3; qy3 = (R[1,2]+R[2,1])/S3; qz3 = 0.25*S3
    cond0 = tr > 0
    cond1 = (R[0,0] > R[1,1]) & (R[0,0] > R[2,2]) & ~cond0
    cond2 = (R[1,1] > R[2,2]) & ~cond0 & ~cond1
    qw = torch.where(cond0, qw0, torch.where(cond1, qw1, torch.where(cond2, qw2, qw3)))
    qx = torch.where(cond0, qx0, torch.where(cond1, qx1, torch.where(cond2, qx2, qx3)))
    qy = torch.where(cond0, qy0, torch.where(cond1, qy1, torch.where(cond2, qy2, qy3)))
    qz = torch.where(cond0, qz0, torch.where(cond1, qz1, torch.where(cond2, qz2, qz3)))
    q = torch.stack([qw, qx, qy, qz])
    return q / (torch.norm(q) + 1e-12)


def quaternion_multiply_wxyz(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1)
    w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


import math as _math

def rotation_matrix_to_quaternion_np(R):
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = _math.sqrt(tr+1.0)*2
        qw,qx,qy,qz = 0.25*S,(R[2,1]-R[1,2])/S,(R[0,2]-R[2,0])/S,(R[1,0]-R[0,1])/S
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        S = _math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
        qw,qx,qy,qz = (R[2,1]-R[1,2])/S,0.25*S,(R[0,1]+R[1,0])/S,(R[0,2]+R[2,0])/S
    elif R[1,1]>R[2,2]:
        S = _math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
        qw,qx,qy,qz = (R[0,2]-R[2,0])/S,(R[0,1]+R[1,0])/S,0.25*S,(R[1,2]+R[2,1])/S
    else:
        S = _math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
        qw,qx,qy,qz = (R[1,0]-R[0,1])/S,(R[0,2]+R[2,0])/S,(R[1,2]+R[2,1])/S,0.25*S
    q = np.array([qw,qx,qy,qz], dtype=np.float64)
    return q / (np.linalg.norm(q)+1e-12)


def rotation_matrix_to_euler_xyz_deg(R):
    sy = _math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        x = _math.atan2(R[2,1], R[2,2])
        y = _math.atan2(-R[2,0], sy)
        z = _math.atan2(R[1,0], R[0,0])
    else:
        x = _math.atan2(-R[1,2], R[1,1])
        y = _math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x,y,z])


# ──────────────────────────────────────────────────────────
# RigidPoseGaussianProxy  (변환된 Gaussian을 differentiable하게 wrap)
# ──────────────────────────────────────────────────────────
class RigidPoseGaussianProxy:
    """
    GaussianModel을 rigid transform (R, t)으로 감싸는 proxy.
    Gaussians 자체는 고정, pose 파라미터(delta_r, delta_t)를 통해 gradient 흐름.
    GS-Pose와 달리 delta를 내부에 넣지 않고 외부에서 주입하는 방식.
    """
    def __init__(self, base, R_obj2cam, t_obj2cam):
        self.base = base
        self.R = R_obj2cam   # [3,3] torch
        self.t = t_obj2cam   # [3] torch
        self.active_sh_degree = base.active_sh_degree
        self.max_sh_degree    = base.max_sh_degree

    @property
    def get_xyz(self):
        return self.base.get_xyz @ self.R.transpose(0,1) + self.t.unsqueeze(0)

    @property
    def get_opacity(self):
        return self.base.get_opacity

    @property
    def get_scaling(self):
        return self.base.get_scaling

    @property
    def get_features(self):
        return self.base.get_features

    @property
    def get_rotation(self):
        q_base = self.base.get_rotation   # [N,4] wxyz
        q_pose = rotation_matrix_to_quaternion_wxyz_torch(self.R)  # [4]
        q_pose = q_pose.unsqueeze(0).expand(q_base.shape[0], 4)
        q_new  = quaternion_multiply_wxyz(q_pose, q_base)
        return q_new / (torch.norm(q_new, dim=1, keepdim=True) + 1e-12)

    def get_covariance(self, scaling_modifier=1.0):
        return self.base.get_covariance(scaling_modifier)

    def get_exposure_from_name(self, image_name):
        return self.base.get_exposure_from_name(image_name)
    
    # KSCHOI added
    def set_T(self, R, t):
        self.R = R
        self.t = t


# ──────────────────────────────────────────────────────────
# Loss: D-SSIM + D-MS-SSIM  (GS-Pose 방식)
# ──────────────────────────────────────────────────────────
def rgb_to_gray(x):
    return 0.299*x[:,0] + 0.587*x[:,1] + 0.114*x[:,2]

def simple_ssim(x, y):
    xg, yg = rgb_to_gray(x), rgb_to_gray(y)
    C1, C2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(xg, 3, 1, 1)
    mu_y = F.avg_pool2d(yg, 3, 1, 1)
    sigma_x  = F.avg_pool2d(xg*xg, 3, 1, 1) - mu_x*mu_x
    sigma_y  = F.avg_pool2d(yg*yg, 3, 1, 1) - mu_y*mu_y
    sigma_xy = F.avg_pool2d(xg*yg, 3, 1, 1) - mu_x*mu_y
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2) + 1e-12
    return (num/den).mean()

def simple_ms_ssim(x, y, levels=3):
    """간단한 Multi-Scale SSIM (levels개 scale)."""
    weights = [0.0448, 0.2856, 0.3001][:levels]
    weights = [w / sum(weights) for w in weights]
    val = 0.0
    for i, w in enumerate(weights):
        if i == len(weights) - 1:
            val = val + w * simple_ssim(x, y)
        else:
            val = val + w * simple_ssim(x, y)
            x = F.avg_pool2d(x, 2, 2)
            y = F.avg_pool2d(y, 2, 2)
    return val

def dssim_loss(render, target):
    """D-SSIM = 1 - SSIM"""
    return 1.0 - simple_ssim(render.unsqueeze(0), target.unsqueeze(0))

def dms_ssim_loss(render, target):
    """D-MS-SSIM = 1 - MS-SSIM"""
    return 1.0 - simple_ms_ssim(render.unsqueeze(0), target.unsqueeze(0))


# ──────────────────────────────────────────────────────────
# Learning rate scheduler with warmup (cosine annealing)
# ──────────────────────────────────────────────────────────
class CosineWarmupScheduler:
    def __init__(self, optimizer, total_steps, warmup_steps, max_lr, min_lr):
        self.optimizer    = optimizer
        self.total_steps  = total_steps
        self.warmup_steps = warmup_steps
        self.max_lr       = max_lr
        self.min_lr       = min_lr
        self._step        = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            lr = self.max_lr * self._step / max(1, self.warmup_steps)
        else:
            progress = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + _math.cos(_math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ──────────────────────────────────────────────────────────
# render → bgr uint8
# ──────────────────────────────────────────────────────────
def chw_to_bgr(chw):
    x = chw.detach().cpu().permute(1,2,0).numpy() * 255
    x = x.clip(0,255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def run_refine_pose(args, gaussians=None, rt_mode=False):
    """
    Run pose refinement.
    Pass pre-loaded gaussians to skip model loading (for in-process preloading).
    Set rt_mode=True to skip all debug image saves and intermediate renders.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA required.")

    model_dir  = Path(args.gs_model_dir)
    output_dir = Path(args.out_dir)
    ensure_dir(output_dir)

    K = load_intrinsics(args.intrinsics_path)
    fx, fy, cx_orig, cy_orig = K_to_params(K)

    T0_path = output_dir / "T0.json"
    T0 = load_json(str(T0_path))

    r0 = np.array(T0["R_obj_to_cam"], dtype=np.float32)
    t0 = np.array(T0["t_obj_to_cam"], dtype=np.float32)
    print( f"initial pose by PnP \n- R: {r0} \n- t: {t0}")
    # ── GS model 로드 ──
    ply_path = resolve_ply_path(model_dir)
    if gaussians is None:
        gaussians = GaussianModel(args.sh_degree)
        gaussians.load_ply(str(ply_path), use_train_test_exp=False)
    else:
        print("[refine_pose.py] Using pre-loaded GaussianModel (skipping PLY load)")

    gaussians.freeze_except_pose()
    
    # Pre-compute gsplat render constants (identity viewmat + K)
    _viewmat_id = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)  # (1,4,4)
    _K_mat = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)  # (1,3,3)
    bg_val = [float(x) / 255.0 for x in args.background.split(",")]
    _bg = torch.tensor(bg_val, dtype=torch.float32, device=device).unsqueeze(0)  # (1,3)

    # ──────────────────────────────────────────────────────
    # 1. crop 파라미터 계산
    # ──────────────────────────────────────────────────────
    query_info = construct_queryInfo(args.query_img, output_dir)

    q_bbox = np.array(query_info["q_bbox"])
    q_side = get_bbox_size(q_bbox)
    q_bbox_ext = square_bbox(q_bbox, int(q_side * 1.2) )

    print(f"[Crop] query mask bbox : {q_bbox} -> {q_bbox_ext}")
    
    # ──────────────────────────────────────────────────────
    # 2. query 이미지 crop → tensor (numpy, 시각화용)
    # ──────────────────────────────────────────────────────
    qm = crop_with_bbox(query_info["query_mask"], q_bbox_ext)
    query_mask = torch.from_numpy(qm).float() / 255.0
    query_mask = query_mask.to(device)

    qc = crop_with_bbox(query_info["query_full"], q_bbox_ext)
    qc = apply_mask(qc, qm)
    query_crop = torch.from_numpy(qc).float().permute(2,0,1) / 255.0
    query_crop = query_crop.to(device)

    # ──────────────────────────────────────────────────────
    # 3. Identity camera (canonical 좌표계, full resolution)
    #    render → crop_resize_chw_by_bbox → crop_size×crop_size
    #    이 방식은 v4에서 검증된 방식이고, t_can으로 bbox를 잡으면
    #    항상 render 위의 캔 위치와 일치함
    # ──────────────────────────────────────────────────────
    init_R_t = torch.tensor(r0, dtype=torch.float32, device=device)
    init_t_t = torch.tensor(t0, dtype=torch.float32, device=device)

    # bbox를 torch tensor로 고정 (query mask 기반, 매 iter 재계산 안 함)
    # render에서도 캔이 동일한 위치에 나타나므로 같은 bbox로 crop
    bbox_fixed = q_bbox_ext

    # ── sanity render (init pose) — skipped in rt_mode ──
    best_render_np = None    
    with torch.no_grad():
        proxy_init = RigidPoseGaussianProxy(gaussians, init_R_t, init_t_t)
        _r, _, _ = _rasterize(
            means=proxy_init.get_xyz, quats=proxy_init.get_rotation,
            scales=proxy_init.get_scaling, opacities=proxy_init.get_opacity.squeeze(-1),
            colors=proxy_init.get_features, viewmats=_viewmat_id, Ks=_K_mat,
            width=int(args.width), height=int(args.height),
            sh_degree=int(proxy_init.active_sh_degree),
            near_plane=0.01, far_plane=100.0, backgrounds=_bg, packed=False,
        )
        render_full = _r[0].permute(2, 0, 1).clamp(0, 1)
        render_crop = crop_chw_with_bbox(render_full, bbox_fixed)

        if not rt_mode:
            render_np   = chw_to_bgr(render_crop)
            cv2.imwrite(str(sanity_dir / "init_render_crop.png"), render_np)
            best_render_np = render_np.copy()
        

    # ──────────────────────────────────────────────────────
    # 4. Optimization
    # ──────────────────────────────────────────────────────
    delta_r = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)
    delta_t = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.AdamW([
        {"params": [delta_r], "lr": args.lr_rot},
        {"params": [delta_t], "lr": args.lr_trans},
    ])

    scheduler = CosineWarmupScheduler(
        optimizer,
        total_steps  = args.iters,
        warmup_steps = args.warmup_steps,
        max_lr       = args.lr_rot,
        min_lr       = args.lr_rot * 0.01,
    )

    print("=" * 60)
    print("[refine_pose v5] Identity cam + differentiable crop")
    print(f"  model_dir  : {model_dir}")
    print(f"  ply_path   : {ply_path}")
    print(f"  iters      : {args.iters}")
    print(f"  crop_size  : {args.crop_size}")
    print(f"  lr_rot     : {args.lr_rot}  lr_trans: {args.lr_trans:.5f}")
    print(f"  loss       : D-SSIM + D-MS-SSIM (GS-Pose style)")
    t_arr = init_t_t.detach().cpu().numpy()
    print(f"  t0     : [{t_arr[0]:.9f}, {t_arr[1]:.9f}, {t_arr[2]:.9f}]")
    R_arr = init_R_t.detach().cpu().numpy()
    print(f"  r0[0]  : [{R_arr[0,0]:.9f}, {R_arr[0,1]:.9f}, {R_arr[0,2]:.9f}]")
    print("=" * 60)

    losses = []
    best_loss  = 1e9
    best_state = {
        "R": init_R_t.detach().cpu().numpy().copy(),
        "t": init_t_t.detach().cpu().numpy().copy(),
        "iter": 0,
    }

    st = time.perf_counter()
    _iter = range(args.iters) if rt_mode else tqdm(range(args.iters), desc="Refining pose")
    for it in _iter:
        # st0 = time.perf_counter()

        optimizer.zero_grad()

        dR    = so3_exp_map(delta_r)
        R_cur = dR @ init_R_t
        # tz clamp: init_t의 ±40% 범위로 제한 (frustum 이탈 방지)
        tz_init = float(init_t_t[2].item())
        t_raw = init_t_t + delta_t
        t_cur = torch.stack([
            t_raw[0],
            t_raw[1],
            torch.clamp(t_raw[2], min=tz_init * 0.6, max=tz_init * 1.4),
        ])

        # full resolution render with identity cam (gsplat)
        proxy = RigidPoseGaussianProxy(gaussians, R_cur, t_cur)

        _r, _, _ = _rasterize(
            means=proxy.get_xyz, quats=proxy.get_rotation,
            scales=proxy.get_scaling, opacities=proxy.get_opacity.squeeze(-1),
            colors=proxy.get_features, viewmats=_viewmat_id, Ks=_K_mat,
            width=int(args.width), height=int(args.height),
            sh_degree=int(proxy.active_sh_degree),
            near_plane=0.01, far_plane=100.0, backgrounds=_bg, packed=False,
        )

        render_full = _r[0].permute(2, 0, 1).clamp(0, 1)
        render_crop = crop_chw_with_bbox(render_full, bbox_fixed)

# 현재 학습 진행도 (0.0 ~ 1.0)
        progress = it / max(1, args.iters)

        # 1. Silhouette (Mask) Loss
        render_alpha = (render_crop.sum(dim=0, keepdim=True) > 0.05).float()
        loss_mask = F.l1_loss(render_alpha, query_mask)

        # 2. RGB Loss (SSIM, L1)
        loss_ssim = dssim_loss(render_crop, query_crop) + dms_ssim_loss(render_crop, query_crop)
        loss_l1_rgb = F.l1_loss(render_crop, query_crop)

        # 3. Blur Loss
        blur_target = F.avg_pool2d(query_crop.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)
        blur_render = F.avg_pool2d(render_crop.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)
        loss_blur = F.l1_loss(blur_render, blur_target)

        # 4. [핵심] Dynamic Weighting (Coarse-to-Fine)
        # 초반: Blur 중심 (크게 돌리기) / 후반: SSIM 중심 (칼같이 맞추기)
        weight_blur = 1.0 * (1.0 - progress)  # 1.0에서 0.0으로 서서히 감소
        weight_ssim = 0.1 + 0.9 * progress    # 0.1에서 1.0으로 서서히 증가
        weight_l1   = 0.5                     # 기본 위치 유지를 위해 고정
        weight_mask = 1.0                     # 크기 유지를 위해 고정

        loss = (weight_ssim * loss_ssim) + (weight_l1 * loss_l1_rgb) + (weight_blur * loss_blur) + (weight_mask * loss_mask)

        loss.backward()

        torch.nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        losses.append({"iter": it, "loss": loss_val})

        # Log R (euler angles) and t per iteration for trajectory visualization
        if not rt_mode:
            _euler = rotation_matrix_to_euler_xyz_deg(R_cur.detach().cpu().numpy())
            _t_np  = t_cur.detach().cpu().numpy()
            traj_record = {
                "iter": it,
                "rx": float(_euler[0]), "ry": float(_euler[1]), "rz": float(_euler[2]),
                "tx": float(_t_np[0]),  "ty": float(_t_np[1]),  "tz": float(_t_np[2]),
            }
            losses[-1].update(traj_record)  # inline with loss entry

        tracking_loss = float(loss_ssim.item() + loss_l1_rgb.item() + loss_blur.item() + loss_mask.item())

        # 이제 loss_val이 아닌 tracking_loss 기준으로 최고를 갱신합니다.
        if tracking_loss < best_loss:
            best_loss  = tracking_loss
            best_state = {
                "R": R_cur.detach().cpu().numpy().copy(),
                "t": t_cur.detach().cpu().numpy().copy(),
                "iter": it,
            }
            print(f"  [NewBest] iter={it}  tracking_loss={tracking_loss:.6f}  translation=[{t_cur[0].item()*100:.2f}, {t_cur[1].item()*100:.2f}, {t_cur[2].item()*100:.2f}]"
                  )
            if True: #not rt_mode:
                best_render_np = chw_to_bgr(render_crop.detach().clamp(0, 1))

        # Early stopping
        if it >= args.early_stop_steps:
            loss_vals  = torch.tensor([l["loss"] for l in losses])
            loss_grads = (loss_vals[1:] - loss_vals[:-1]).abs()
            recent_grad = loss_grads[-args.early_stop_steps:].mean().item()
            if recent_grad < args.early_stop_thresh:
                print(f"  [EarlyStop] iter={it}  grad_norm={recent_grad:.2e}")
                break

    et = time.perf_counter()
    print(f"refinement time: {et-st} seconds")
    # ──────────────────────────────────────────────────────
    # 5. 저장
    # ──────────────────────────────────────────────────────
    best_R = best_state["R"]
    best_t = best_state["t"]
    
    if True: #not rt_mode:
        # side-by-side
        q_np = cv2.cvtColor(
            (query_crop.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8),
            cv2.COLOR_RGB2BGR)
        comp = np.zeros((q_np.shape[0], q_np.shape[1]*2, 3), dtype=np.uint8)
        comp[:, :q_np.shape[1]]  = q_np
        comp[:, q_np.shape[1]:]  = best_render_np
        cv2.putText(comp, "query (masked)", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(comp, "refined render", (q_np.shape[1]+12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imwrite(str(output_dir / "refined_query_vs_render.png"), comp)

        # full-res render (best pose)
        with torch.no_grad():
            R_b = torch.tensor(best_R, dtype=torch.float32, device=device)
            t_b = torch.tensor(best_t, dtype=torch.float32, device=device)
            proxy_best = RigidPoseGaussianProxy(gaussians, R_b, t_b)
            _r, _, _ = _rasterize(
                means=proxy_best.get_xyz, quats=proxy_best.get_rotation,
                scales=proxy_best.get_scaling, opacities=proxy_best.get_opacity.squeeze(-1),
                colors=proxy_best.get_features, viewmats=_viewmat_id, Ks=_K_mat,
                width=int(args.width), height=int(args.height),
                sh_degree=int(proxy_best.active_sh_degree),
                near_plane=0.01, far_plane=100.0, backgrounds=_bg, packed=False,
            )
            refined_render_full_np = chw_to_bgr(_r[0].permute(2, 0, 1).clamp(0, 1))
            cv2.imwrite(str(output_dir / "refined_render_full.png"), refined_render_full_np)

        alpha = 0.6
        overlay_crop = cv2.addWeighted(q_np, alpha, best_render_np, 1.0 - alpha, 0)        

        comp_overlay = np.zeros((q_np.shape[0], q_np.shape[1] * 3, 3), dtype=np.uint8)
        comp_overlay[:, :q_np.shape[1]] = q_np
        comp_overlay[:, q_np.shape[1]:q_np.shape[1]*2] = best_render_np
        comp_overlay[:, q_np.shape[1]*2:] = overlay_crop
        cv2.putText(comp_overlay, "Query",   (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(comp_overlay, "Render",  (q_np.shape[1] + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(comp_overlay, "Overlay", (q_np.shape[1]*2 + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imwrite(str(output_dir / "refined_overlay_crop_comp.png"), comp_overlay)

        save_json(output_dir / "refinement_curve.json", {"iters": args.iters, "losses": losses})

        # ── R / t trajectory plot ──────────────────────────────────────
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            iters_arr = [r["iter"] for r in losses if "rx" in r]
            rx_arr = [r["rx"] for r in losses if "rx" in r]
            ry_arr = [r["ry"] for r in losses if "ry" in r]
            rz_arr = [r["rz"] for r in losses if "rz" in r]
            tx_arr = [r["tx"] for r in losses if "tx" in r]
            ty_arr = [r["ty"] for r in losses if "ty" in r]
            tz_arr = [r["tz"] for r in losses if "tz" in r]
            best_it = int(best_state["iter"])

            fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
            fig.suptitle("Step7 GS refinement  —  R / t trajectory", fontsize=13)

            def _plot(ax, y, label, color):
                ax.plot(iters_arr, y, color=color, linewidth=1.0)
                ax.axvline(best_it, color="red", linestyle="--", linewidth=1.2, label=f"best iter={best_it}")
                ax.set_ylabel(label, fontsize=9)
                ax.legend(fontsize=7, loc="upper right")
                ax.grid(True, linewidth=0.4)

            _plot(axes[0, 0], rx_arr, "Rx  (deg)", "#1f77b4")
            _plot(axes[1, 0], ry_arr, "Ry  (deg)", "#ff7f0e")
            _plot(axes[2, 0], rz_arr, "Rz  (deg)", "#2ca02c")
            _plot(axes[0, 1], tx_arr, "tx  (m)",   "#d62728")
            _plot(axes[1, 1], ty_arr, "ty  (m)",   "#9467bd")
            _plot(axes[2, 1], tz_arr, "tz  (m)",   "#8c564b")

            for ax in axes[2]:
                ax.set_xlabel("iteration", fontsize=9)

            plt.tight_layout()
            traj_png = output_dir / "refinement_trajectory.png"
            plt.savefig(str(traj_png), dpi=120)
            plt.close(fig)
            print(f"  [step7] trajectory plot saved: {traj_png}")
        except Exception as _e:
            print(f"  [step7] trajectory plot failed: {_e}")

    pose_record = {
        "stage": "step6",
        "model_dir":                  str(model_dir),
        "ply_path":                   str(ply_path),
        "best_iter":                  int(best_state["iter"]),
        "final_loss":                 float(best_loss),
        "crop": {
            "bbox_cx": str((q_bbox_ext[0]+q_bbox_ext[2])/2.0), "bbox_cy": str((q_bbox_ext[1]+q_bbox_ext[3])/2.0), "bbox_side": str(q_side),
            "margin_scale": args.crop_margin_scale,
        },
        "R_obj_to_cam_refined":       best_R.tolist(),
        "t_obj_to_cam_refined":       best_t.tolist()
    }
    if not rt_mode:
        pose_record["outputs"] = {            
            "refined_render_full":       str(output_dir / "refined_render_full.png"),
            "refined_query_vs_render":   str(output_dir / "refined_query_vs_render.png"),
            "refinement_curve_json":     str(output_dir / "refinement_curve.json"),
            "refined_overlay_crop_comp": str(output_dir / "refined_overlay_crop_comp.png"),
        }
    save_json(output_dir / "refined_pose.json", pose_record)

    print("=" * 60)
    print("[refine_pose v5] Done")
    print(f"  best_iter  : {best_state['iter']}")
    print(f"  final_loss : {best_loss:.6f}")
    print(f"  t (m)      : {best_t.tolist()}")
    print("=" * 60)


def main():
    args = parse_args()
    run_refine_pose(args, rt_mode=args.rt_mode)


if __name__ == "__main__":
    main()