"""
refine_pose.py  —  GS-Pose style pose refinement (v5)
======================================================
변경 사항:
  - Identity camera (canonical 좌표계, full resolution) 유지
  - t_can 기반 projected_bbox_from_pose → differentiable crop
  - query도 동일 bbox로 crop (numpy)
  - loss: D-SSIM + D-MS-SSIM (GS-Pose 방식, L1 제거)
  - trunc_mask: query crop의 non-black만 loss에 반영
  - optimizer: AdamW + CosineAnnealing + warmup + early stopping


greate result in unmasked query
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from gaussian_renderer import GaussianModel
from gsplat import rasterization as _gsplat_rasterize_3dgs
from gsplat import rasterization_2dgs as _gsplat_rasterize_2dgs


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
    p.add_argument("--model_dir",           required=True)
    p.add_argument("--initial_pose_json",   required=True)
    p.add_argument("--query_masked_path",   required=True,
                   help="Masked query image (background black), full resolution")
    p.add_argument("--query_mask_path",     required=True,
                   help="Binary mask image (uint8, foreground > 0), full resolution")
    p.add_argument("--intrinsics_path",     required=True)
    p.add_argument("--output_dir",          required=True)
    p.add_argument("--width",               required=True, type=int)
    p.add_argument("--height",              required=True, type=int)
    p.add_argument("--background",          default="0,0,0")
    p.add_argument("--iteration",           default=-1, type=int)
    p.add_argument("--sh_degree",           default=3,   type=int)
    # optimizer
    p.add_argument("--iters",               default=100, type=int)
    p.add_argument("--lr_rot",              default=1e-2, type=float)
    p.add_argument("--lr_trans",            default=5e-3, type=float)
    p.add_argument("--warmup_steps",        default=10,  type=int)
    p.add_argument("--early_stop_steps",    default=20,  type=int,
                   help="Early stop when loss grad norm (last N steps) < threshold")
    p.add_argument("--early_stop_thresh",   default=1e-5, type=float)
    # crop
    p.add_argument("--crop_size",           default=320, type=int,
                   help="Square crop target size for both query and render")
    p.add_argument("--crop_margin_scale",   default=1.3, type=float,
                   help="Margin factor around mask bbox (1.0 = tight)")
    p.add_argument("--rt_mode", action="store_true",
                   help="Skip all debug image saves; output only refined_pose.json")
    return p.parse_args()


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

def resolve_ply_path(model_dir: Path, iteration: int):
    pc_root = model_dir / "point_cloud"
    if not pc_root.exists():
        raise FileNotFoundError(f"point_cloud dir not found: {pc_root}")
    if iteration == -1:
        iteration = search_for_max_iteration(pc_root)
    ply = pc_root / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply.exists():
        raise FileNotFoundError(f"point_cloud.ply not found: {ply}")
    return ply, iteration

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
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    elif len(vals) == 4:
        fx, fy, cx, cy = vals
    else:
        raise ValueError(f"Unsupported intrinsics format: {path}")
    return fx, fy, cx, cy


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


def bbox_to_square_with_margin(x1, y1, x2, y2, img_w, img_h, margin_scale=1.3):
    """
    tight bbox를 margin을 포함한 정사각형 bbox로 확장.
    반환: (cx, cy, side)  — 중심과 한 변의 길이
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = x2 - x1
    bh = y2 - y1
    side = max(bw, bh) * margin_scale
    # 이미지 경계 클램프
    half = side / 2.0
    cx = float(np.clip(cx, half, img_w - half))
    cy = float(np.clip(cy, half, img_h - half))
    side = float(side)
    return cx, cy, side


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


def crop_resize_chw_by_bbox(render_chw, bbox, out_size=320):
    C, H, W = render_chw.shape
    x1, y1, x2, y2 = bbox
    side    = torch.maximum(x2 - x1, y2 - y1)
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


# ──────────────────────────────────────────────────────────
# Loss: D-SSIM + D-MS-SSIM  (GS-Pose 방식)
# ──────────────────────────────────────────────────────────
def rgb_to_gray(x):
    return 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]

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
def render_chw_to_bgr_uint8(render_chw):
    x = render_chw.detach().cpu().permute(1,2,0).numpy()
    x = (x*255).clip(0,255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
def load_gaussians(model_dir, iteration=-1, sh_degree=3):
    """Load GaussianModel from PLY. Call once and keep in GPU memory."""
    ply_path, resolved_iter = resolve_ply_path(Path(model_dir), iteration)
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(ply_path), use_train_test_exp=False)
    print(f"[GS] Loaded gaussians: {ply_path}  (iter={resolved_iter})")
    return gaussians, ply_path, resolved_iter


def run_refine_pose(args, gaussians=None, rt_mode=False):
    """
    Run pose refinement.
    Pass pre-loaded gaussians to skip model loading (for in-process preloading).
    Set rt_mode=True to skip all debug image saves and intermediate renders.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA required.")

    model_dir  = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    fx, fy, cx_orig, cy_orig = load_intrinsics(args.intrinsics_path)
    init = load_json(args.initial_pose_json)

    init_R = np.array(init["R_obj_to_cam"], dtype=np.float32)
    init_t = np.array(init.get("t_obj_to_cam", init.get("tvec")), dtype=np.float32)

    # ── GS model 로드 ──
    ply_path, resolved_iter = resolve_ply_path(model_dir, args.iteration)
    if gaussians is None:
        gaussians = GaussianModel(args.sh_degree)
        gaussians.load_ply(str(ply_path), use_train_test_exp=False)
    else:
        print("[refine_pose.py] Using pre-loaded GaussianModel (skipping PLY load)")

    # Freeze all Gaussian parameters: only delta_r and delta_t should be
    # in the autograd graph.  Without this, loss.backward() computes (and
    # accumulates) gradients for every nn.Parameter in GaussianModel on
    # every iteration.  optimizer.zero_grad() only zeros delta_r / delta_t,
    # so Gaussian grads pile up, growing the retained graph and introducing
    # numerical drift that differs between in-process (RT) and subprocess
    # (non-RT) execution contexts.
    # GaussianModel is NOT an nn.Module, so iterate its Parameters manually.
    for _attr in ("_xyz", "_features_dc", "_features_rest",
                  "_scaling", "_rotation", "_opacity", "_exposure"):
        _p = getattr(gaussians, _attr, None)
        if isinstance(_p, torch.nn.Parameter):
            _p.requires_grad_(False)

    # ── Scale correction (canonical ↔ real-world) ──
    xyz_np = gaussians.get_xyz.detach().cpu().numpy()
    gs_extent = xyz_np.max(axis=0) - xyz_np.min(axis=0)
    real_height = float(init.get("object_height_m", 0.125))

    # canonical PLY는 apply_scale=1로 target_height_m 기준으로 normalize됨
    # → canonical 1 unit ≈ 1 meter → w2c_scale = 1.0
    canonical_transform_json = model_dir / "canonical_transform.json"
    if canonical_transform_json.exists():
        w2c_scale = 1.0
        print(f"[Scale] canonical_transform.json found → w2c_scale=1.0 (metric PLY)")
    else:
        canonical_height = float(gs_extent[2])
        w2c_scale = canonical_height / real_height
        print(f"[Scale] canonical_transform.json not found → w2c_scale={w2c_scale:.4f}")

    init_t_can = init_t * w2c_scale  # w2c_scale=1.0이면 init_t 그대로

    print(f"[Scale] PLY extent   : {gs_extent}")
    print(f"[Scale] real height  : {real_height:.4f} m")
    print(f"[Scale] w2c_scale    : {w2c_scale:.4f}")
    print(f"[Scale] init_t (m)   : {init_t}")
    print(f"[Scale] init_t (can) : {init_t_can}")

    # Pre-compute gsplat render constants (identity viewmat + K)
    _viewmat_id = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)  # (1,4,4)
    _K_mat = torch.tensor(
        [[fx, 0.0, cx_orig], [0.0, fy, cy_orig], [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)  # (1,3,3)
    bg_val = [float(x) / 255.0 for x in args.background.split(",")]
    _bg = torch.tensor(bg_val, dtype=torch.float32, device=device).unsqueeze(0)  # (1,3)

    # ──────────────────────────────────────────────────────
    # 1. crop 파라미터 계산
    #    init_t_can (canonical scale) → identity cam 좌표계에서 projection
    #    identity cam은 canonical 좌표계에서 동작하므로 canonical t로 계산
    # ──────────────────────────────────────────────────────
    mask_gray = cv2.imread(str(args.query_mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise FileNotFoundError(f"Failed to load mask: {args.query_mask_path}")
    img_h, img_w = mask_gray.shape

    # query mask에서 직접 bbox 계산 (canonical t projection 대신)
    # render에서 캔도 동일한 위치에 나타나므로 같은 bbox로 crop하면 정확히 맞음
    x1_m, y1_m, x2_m, y2_m = get_mask_bbox(mask_gray)
    bbox_cx, bbox_cy, bbox_side = bbox_to_square_with_margin(
        x1_m, y1_m, x2_m, y2_m,
        img_w, img_h,
        margin_scale=args.crop_margin_scale,
    )

    print(f"[Crop] query mask bbox : ({x1_m},{y1_m}) → ({x2_m},{y2_m})")
    print(f"[Crop] square center   : ({bbox_cx:.1f},{bbox_cy:.1f})")
    print(f"[Crop] bbox_side (px)  : {bbox_side:.1f}")
    print(f"[Crop] target_size     : {args.crop_size}")

    # ──────────────────────────────────────────────────────
    # 2. query 이미지 crop → tensor (numpy, 시각화용)
    # ──────────────────────────────────────────────────────
    query_bgr = cv2.imread(str(args.query_masked_path), cv2.IMREAD_COLOR)
    if query_bgr is None:
        raise FileNotFoundError(f"Failed to load query: {args.query_masked_path}")

    query_crop_bgr = crop_and_resize(query_bgr, bbox_cx, bbox_cy, bbox_side, args.crop_size)
    mask_crop_gray = crop_and_resize_gray(mask_gray, bbox_cx, bbox_cy, bbox_side, args.crop_size)

    # RGB tensor [C,H,W] float32 [0,1]
    query_rgb  = cv2.cvtColor(query_crop_bgr, cv2.COLOR_BGR2RGB)
    query_crop_t = torch.from_numpy(query_rgb).float().permute(2,0,1) / 255.0  # [3,S,S]
    mask_crop_t  = torch.from_numpy((mask_crop_gray > 0).astype(np.float32)).unsqueeze(0)  # [1,S,S]

    # target = query_crop * mask_crop (segmented object only)
    target_img = (query_crop_t * mask_crop_t).to(device)   # [3,S,S]

    # trunc_mask: query crop의 non-black 픽셀
    trunc_mask = (query_crop_t.sum(dim=0, keepdim=True) > 0).float().to(device)

    if not rt_mode:
        sanity_dir = output_dir / "sanity_check"
        ensure_dir(sanity_dir)
        cv2.imwrite(str(sanity_dir / "query_crop.png"), query_crop_bgr)
        cv2.imwrite(str(sanity_dir / "mask_crop.png"),  mask_crop_gray)
        cv2.imwrite(str(sanity_dir / "target_masked.png"),
                    cv2.cvtColor(
                        (target_img.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR))

    # ──────────────────────────────────────────────────────
    # 3. Identity camera (canonical 좌표계, full resolution)
    #    render → crop_resize_chw_by_bbox → crop_size×crop_size
    #    이 방식은 v4에서 검증된 방식이고, t_can으로 bbox를 잡으면
    #    항상 render 위의 캔 위치와 일치함
    # ──────────────────────────────────────────────────────
    init_R_t = torch.tensor(init_R, dtype=torch.float32, device=device)
    init_t_t = torch.tensor(init_t_can, dtype=torch.float32, device=device)

    # bbox를 torch tensor로 고정 (query mask 기반, 매 iter 재계산 안 함)
    # render에서도 캔이 동일한 위치에 나타나므로 같은 bbox로 crop
    bbox_fixed = (
        torch.tensor(bbox_cx - bbox_side / 2.0, dtype=torch.float32, device=device),
        torch.tensor(bbox_cy - bbox_side / 2.0, dtype=torch.float32, device=device),
        torch.tensor(bbox_cx + bbox_side / 2.0, dtype=torch.float32, device=device),
        torch.tensor(bbox_cy + bbox_side / 2.0, dtype=torch.float32, device=device),
    )

    # ── sanity render (init pose) — skipped in rt_mode ──
    best_render_np = None
    if not rt_mode:
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

            render_init_crop = crop_resize_chw_by_bbox(render_full, bbox_fixed, out_size=args.crop_size)
            render_init_np   = render_chw_to_bgr_uint8(render_init_crop)
            cv2.imwrite(str(sanity_dir / "init_render_crop.png"), render_init_np)

            nz    = int((render_init_crop > 0.01).any(dim=0).sum().item())
            total = args.crop_size * args.crop_size
            print(f"[Sanity] init render non-zero={nz}/{total}")
            print(f"[Sanity] bbox (fixed): x1={float(bbox_fixed[0]):.0f} y1={float(bbox_fixed[1]):.0f} "
                  f"x2={float(bbox_fixed[2]):.0f} y2={float(bbox_fixed[3]):.0f}")
            best_render_np = render_init_np.copy()

    # ──────────────────────────────────────────────────────
    # 4. Optimization
    # ──────────────────────────────────────────────────────
    delta_r = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)
    delta_t = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)

    lr_trans_scaled = args.lr_trans * w2c_scale

    optimizer = torch.optim.AdamW([
        {"params": [delta_r], "lr": args.lr_rot},
        {"params": [delta_t], "lr": lr_trans_scaled},
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
    print(f"  lr_rot     : {args.lr_rot}  lr_trans(scaled): {lr_trans_scaled:.5f}")
    print(f"  loss       : D-SSIM + D-MS-SSIM (GS-Pose style)")
    t_arr = init_t_t.detach().cpu().numpy()
    print(f"  init_t     : [{t_arr[0]:.9f}, {t_arr[1]:.9f}, {t_arr[2]:.9f}]")
    R_arr = init_R_t.detach().cpu().numpy()
    print(f"  init_R[0]  : [{R_arr[0,0]:.9f}, {R_arr[0,1]:.9f}, {R_arr[0,2]:.9f}]")
    print("=" * 60)

    losses = []
    best_loss  = 1e9
    best_state = {
        "R": init_R_t.detach().cpu().numpy().copy(),
        "t": init_t_t.detach().cpu().numpy().copy(),
        "iter": 0,
    }

    _iter = range(args.iters) if rt_mode else tqdm(range(args.iters), desc="Refining pose")
    for it in _iter:
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

        # 고정 bbox (query mask 기반)로 crop — t_cur가 변해도 bbox는 고정
        render_crop = crop_resize_chw_by_bbox(render_full, bbox_fixed, out_size=args.crop_size)

        # trunc_mask 적용 (render의 non-black만 loss에 반영)
        # render_masked = render_crop * trunc_mask

        # # GS-Pose loss: D-SSIM + D-MS-SSIM
        # loss = dssim_loss(render_masked, target_img) + dms_ssim_loss(render_masked, target_img)

# # 1. Silhouette (Mask) Loss - 크기와 t_z 유지
#         render_alpha = (render_crop.sum(dim=0, keepdim=True) > 0.05).float()
#         loss_mask = F.l1_loss(render_alpha, trunc_mask)

#         # 2. RGB Loss (SSIM은 Local 디테일, L1은 Global 위치)
#         loss_ssim = dssim_loss(render_crop, target_img) + dms_ssim_loss(render_crop, target_img)
#         loss_l1_rgb = F.l1_loss(render_crop, target_img)

#         # 3. [핵심] Blur L1 Loss 추가 (텍스처 수렴 유도)
#         # 커널 사이즈(9x9)를 주어 이미지를 강하게 뭉갠 뒤 비교합니다.
#         # 이렇게 하면 로고가 살짝 어긋나 있어도 흐릿한 색상 정보(빨강/파랑)를 따라 
#         # 올바른 방향으로 Rotation Gradient가 발생합니다.
#         blur_target = F.avg_pool2d(target_img.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)
#         blur_render = F.avg_pool2d(render_crop.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)
#         loss_blur = F.l1_loss(blur_render, blur_target)

#         # 4. 최종 Loss 조합 (가중치 리밸런싱)
#         # SSIM의 비중을 대폭 줄이고, 멀리서 당겨오는 Blur와 L1의 비중을 높입니다.
#         loss = (0.2 * loss_ssim) + (0.5 * loss_l1_rgb) + (1.0 * loss_blur) + (1.0 * loss_mask)

# 현재 학습 진행도 (0.0 ~ 1.0)
        progress = it / max(1, args.iters)

        # 1. Silhouette (Mask) Loss
        render_alpha = (render_crop.sum(dim=0, keepdim=True) > 0.05).float()
        loss_mask = F.l1_loss(render_alpha, trunc_mask)

        # 2. RGB Loss (SSIM, L1)
        loss_ssim = dssim_loss(render_crop, target_img) + dms_ssim_loss(render_crop, target_img)
        loss_l1_rgb = F.l1_loss(render_crop, target_img)

        # 3. Blur Loss
        blur_target = F.avg_pool2d(target_img.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)
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
            _t_np  = t_cur.detach().cpu().numpy() / w2c_scale
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
            if not rt_mode:
                best_render_np = render_chw_to_bgr_uint8(render_crop.detach().clamp(0, 1))
        # Early stopping
        if it >= args.early_stop_steps:
            loss_vals  = torch.tensor([l["loss"] for l in losses])
            loss_grads = (loss_vals[1:] - loss_vals[:-1]).abs()
            recent_grad = loss_grads[-args.early_stop_steps:].mean().item()
            if recent_grad < args.early_stop_thresh:
                print(f"  [EarlyStop] iter={it}  grad_norm={recent_grad:.2e}")
                break

    # ──────────────────────────────────────────────────────
    # 5. 저장
    # ──────────────────────────────────────────────────────
    best_R     = best_state["R"]
    best_t_can = best_state["t"]
    best_t_m   = best_t_can / w2c_scale

    q     = rotation_matrix_to_quaternion_np(best_R)
    euler = rotation_matrix_to_euler_xyz_deg(best_R)

    if not rt_mode:
        # best render crop
        cv2.imwrite(str(output_dir / "refined_render_crop.png"), best_render_np)

        # side-by-side
        q_np = cv2.cvtColor(
            (target_img.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8),
            cv2.COLOR_RGB2BGR)
        comp = np.zeros((args.crop_size, args.crop_size*2, 3), dtype=np.uint8)
        comp[:, :args.crop_size]  = q_np
        comp[:, args.crop_size:]  = best_render_np
        cv2.putText(comp, "query (masked)", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(comp, "refined render", (args.crop_size+12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imwrite(str(output_dir / "refined_query_vs_render.png"), comp)

        # full-res render (best pose)
        with torch.no_grad():
            R_b = torch.tensor(best_R,     dtype=torch.float32, device=device)
            t_b = torch.tensor(best_t_can, dtype=torch.float32, device=device)
            proxy_best = RigidPoseGaussianProxy(gaussians, R_b, t_b)
            _r, _, _ = _rasterize(
                means=proxy_best.get_xyz, quats=proxy_best.get_rotation,
                scales=proxy_best.get_scaling, opacities=proxy_best.get_opacity.squeeze(-1),
                colors=proxy_best.get_features, viewmats=_viewmat_id, Ks=_K_mat,
                width=int(args.width), height=int(args.height),
                sh_degree=int(proxy_best.active_sh_degree),
                near_plane=0.01, far_plane=100.0, backgrounds=_bg, packed=False,
            )
            refined_render_full_np = render_chw_to_bgr_uint8(_r[0].permute(2, 0, 1).clamp(0, 1))
            cv2.imwrite(str(output_dir / "refined_render_full.png"), refined_render_full_np)

        alpha = 0.6
        overlay_crop = cv2.addWeighted(query_crop_bgr, alpha, best_render_np, 1.0 - alpha, 0)
        cv2.imwrite(str(output_dir / "overlay_crop.png"), overlay_crop)

        if query_bgr.shape == refined_render_full_np.shape:
            overlay_full = cv2.addWeighted(query_bgr, alpha, refined_render_full_np, 1.0 - alpha, 0)
            cv2.imwrite(str(output_dir / "overlay_full.png"), overlay_full)

        comp_overlay = np.zeros((args.crop_size, args.crop_size * 3, 3), dtype=np.uint8)
        comp_overlay[:, :args.crop_size] = query_crop_bgr
        comp_overlay[:, args.crop_size:args.crop_size*2] = best_render_np
        comp_overlay[:, args.crop_size*2:] = overlay_crop
        cv2.putText(comp_overlay, "Query",   (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(comp_overlay, "Render",  (args.crop_size + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(comp_overlay, "Overlay", (args.crop_size*2 + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
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
        "initial_pose_json":          str(args.initial_pose_json),
        "model_dir":                  str(model_dir),
        "ply_path":                   str(ply_path),
        "iteration":                  resolved_iter,
        "best_iter":                  int(best_state["iter"]),
        "final_loss":                 float(best_loss),
        "world_to_canonical_scale":   float(w2c_scale),
        "crop": {
            "bbox_cx": bbox_cx, "bbox_cy": bbox_cy, "bbox_side": bbox_side,
            "source": "query_mask",
            "crop_size": args.crop_size, "margin_scale": args.crop_margin_scale,
        },
        "R_obj_to_cam_refined":       best_R.tolist(),
        "t_obj_to_cam_refined":       best_t_m.tolist(),
        "t_obj_to_cam_canonical":     best_t_can.tolist(),
        "quat_wxyz_refined":          q.tolist(),
        "euler_xyz_deg_refined":      euler.tolist(),
    }
    if not rt_mode:
        pose_record["outputs"] = {
            "refined_render_crop":       str(output_dir / "refined_render_crop.png"),
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
    print(f"  t (m)      : {best_t_m.tolist()}")
    print(f"  t (can)    : {best_t_can.tolist()}")
    print(f"  scale      : {w2c_scale:.4f}")
    print("=" * 60)


def main():
    args = parse_args()
    run_refine_pose(args, rt_mode=args.rt_mode)


if __name__ == "__main__":
    main()