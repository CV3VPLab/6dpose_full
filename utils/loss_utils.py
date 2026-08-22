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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from .image_utils import rgb_to_gray, get_gradient_filters, compute_gradients, erode_binary_tensor
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None


class ECCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(ECCLoss, self).__init__()
        self.eps = eps # 0으로 나누어지는 것을 방지하기 위한 작은 값

    def forward(self, img1, img2):
        """ img1, img2: (B, C, H, W) 형태의 텐서 """
        # 1. 각 이미지의 공간 차원(H, W)에 대한 평균을 구하고 빼줍니다 (Zero-mean)
        img1_mean = img1 - torch.mean(img1, dim=[-2, -1], keepdim=True)
        img2_mean = img2 - torch.mean(img2, dim=[-2, -1], keepdim=True)

        # 2. 분자: 두 이미지의 공분산 (Covariance)
        cov = torch.mean(img1_mean * img2_mean, dim=[-2, -1], keepdim=True)

        # 3. 분모: 각 이미지의 표준편차 (Standard Deviation)
        std1 = torch.std(img1_mean, dim=[-2, -1], keepdim=True)
        std2 = torch.std(img2_mean, dim=[-2, -1], keepdim=True)

        # 4. 정규화된 교차 상관계수 (ECC)
        ecc = cov / (std1 * std2 + self.eps)

        # 5. Loss 구성 (1 - ECC) 및 배치 전체 평균 반환
        loss = 1.0 - ecc
        return loss.mean()
    

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


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
    return 1.0 - simple_ssim(render, target)

def dms_ssim_loss(render, target):
    """D-MS-SSIM = 1 - MS-SSIM"""
    return 1.0 - simple_ms_ssim(render, target)


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
