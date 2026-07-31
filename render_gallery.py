import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from gaussian_renderer import GaussianModel
from refine_pose import GaussianRenderer

from modules_6d.retrieval_dino import DinoV2Extractor

from utils.io_utils import (
    ensure_dir, 
    resolve_ply_path,
    load_json,
    load_intrinsics, K_to_params
)
from utils.image_utils import (
    compute_bbox,
    expand_bbox,
    crop_with_bbox,
    get_max_bbox_size,
    square_pad_resize,
    zeropad_square,
    load_rgb    
)
from utils.image_utils import tensor_to_np as t2np

from utils.geom_utils import depth_to_xyz_map


def parse_args():
    p = argparse.ArgumentParser(description="Render gallery images from custom poses for GS model")
    p.add_argument("--obj_dir", required=True, type=str)    
    p.add_argument("--gallery_pose_json", required=True, type=str,
                   help="Path to gallery_poses.json")
    p.add_argument("--intrinsics_path", required=True, type=str,
                   help="Path to intrinsics txt file")
    p.add_argument("--width", required=True, type=int)
    p.add_argument("--height", required=True, type=int)
    p.add_argument("--background", default="0,0,0", type=str,
                   help="R,G,B in 0-255")
    p.add_argument("--sh_degree", default=3, type=int)
    p.add_argument("--convert_SHs_python", action="store_true")
    p.add_argument("--compute_cov3D_python", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--antialiasing", action="store_true")
    p.add_argument("--gs_mode", default="3dgs", choices=["3dgs", "2dgs"])
    p.add_argument("--save_depth", action="store_true")
    p.add_argument("--save_xyz", action="store_true")
    p.add_argument("--dino_model", type=str, default="dinov2_vits14")
    p.add_argument("--dino_input_size", type=int, default=224)
    return p.parse_args()


def load_gallery_poses(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def project_obj_to_image(X_obj, fx, fy, cx, cy, R_obj_to_cam, t_obj_to_cam):
    X_obj = np.asarray(X_obj, dtype=np.float32).reshape(3, 1)
    R = np.asarray(R_obj_to_cam, dtype=np.float32)
    t = np.asarray(t_obj_to_cam, dtype=np.float32).reshape(3, 1)

    X_cam = R @ X_obj + t
    xc, yc, zc = X_cam.reshape(3)

    if zc <= 1e-8:
        return None

    u = fx * xc / zc + cx
    v = fy * yc / zc + cy
    return np.array([u, v], dtype=np.float32)


def save_xyz_reprojection_check(render_bgr, xyz_obj, fx, fy, cx, cy, R, t, out_path, stride=200):
    vis = render_bgr.copy()
    H, W = xyz_obj.shape[:2]

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            X_obj = xyz_obj[y, x]
            if not np.isfinite(X_obj).all():
                continue
            if np.linalg.norm(X_obj) < 1e-8:
                continue

            uv = project_obj_to_image(X_obj, fx, fy, cx, cy, R, t)
            if uv is None:
                continue

            u2, v2 = int(round(uv[0])), int(round(uv[1]))

            cv2.circle(vis, (x, y), 2, (0, 255, 255), -1, cv2.LINE_AA)

            if 0 <= u2 < W and 0 <= v2 < H:
                cv2.circle(vis, (u2, v2), 2, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.line(vis, (x, y), (u2, v2), (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), vis)


def render_save(pose_info, gaussianRenderer: GaussianRenderer, K, args):
    obj_dir = Path(args.obj_dir)    # 3DGS path
    render_dir = obj_dir / "gallery" # rendered gallery path            
    if args.save_depth:
        depth_dir = obj_dir / "depth"            
    if args.save_xyz:
        xyz_dir = obj_dir / "xyz"

    idx = pose_info["index"]        

    R = np.array(pose_info["R_obj_to_cam"], dtype=np.float32)
    t = np.array(pose_info["t_obj_to_cam"], dtype=np.float32)

    gaussianRenderer.set_T(R, t)
    _r, _a, _ = gaussianRenderer.render_no_grad(render_mode="RGB+ED")

    mask = t2np((_a[0].squeeze(2) > 0.5).float())
    bbox = compute_bbox(mask)
    g_crop_bbox = expand_bbox(bbox, margin=12, w=mask.shape[1], h=mask.shape[0])        
    mask_crop = crop_with_bbox(mask, g_crop_bbox)

    rgb_np = _r[0][..., 0:3].clip(0.0, 1.0).detach().cpu().numpy()
    rgb_crop = crop_with_bbox(rgb_np, g_crop_bbox)
    rgb_crop = ((rgb_crop * mask_crop[..., np.newaxis]) * 255.0).astype(np.uint8)
    cv2.imwrite(str(render_dir / f"{idx:04d}.png"), cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))            

    # 32-bit float depth map in camera coordinates (same unit as canonical Gaussian means, typically meters). Invalid pixels have value 0.
    depth_np = None
    if args.save_depth:
        depth_np = t2np(_r[0][..., 3]) * mask
        assert depth_np.dtype == np.float32, f"Expected depth tensor to be float32, got {depth_np.dtype}"

        # save cropped depth data
        depth_crop = crop_with_bbox(depth_np, g_crop_bbox)
        fn_crop = depth_dir / f"{idx:04d}.npy"
        np.save(str(fn_crop), depth_crop)    

    if args.save_xyz:
        if depth_np is None:
            depth_np = t2np(_r[0][..., 3]) * mask

        xyz_obj = depth_to_xyz_map(
            depth_np=depth_np,
            fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2],
            R_obj_to_cam=pose_info["R_obj_to_cam"],
            t_obj_to_cam=pose_info["t_obj_to_cam"]
        )
        xyz_obj_crop = crop_with_bbox(xyz_obj, g_crop_bbox)            
        np.save(str(xyz_dir / f"{idx:04d}.npy"), xyz_obj_crop)
        
    return g_crop_bbox

def run_render_gallery(args, gaussians=None):
    """
    Run gallery rendering.
    Pass pre-loaded gaussians to skip model loading (for in-process preloading).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This renderer is expected to run on CUDA.")

    obj_dir = Path(args.obj_dir)    # 3DGS path
    model_dir = obj_dir / "model"
    render_dir = obj_dir / "gallery" # rendered gallery path
    ensure_dir(obj_dir)
    ensure_dir(model_dir)
    ensure_dir(render_dir)
        
    if args.save_depth:
        depth_dir = obj_dir / "depth"
        ensure_dir(depth_dir)
    
    if args.save_xyz:
        xyz_dir = obj_dir / "xyz"
        ensure_dir(xyz_dir)

    galleryPoses = load_gallery_poses(args.gallery_pose_json)["poses"]
    # fx, fy, cx, cy = K_to_params(load_intrinsics(args.intrinsics_path))
    calib_info = load_json(args.intrinsics_path)
    fx = calib_info["left_rect"]["fx"]
    fy = calib_info["left_rect"]["fy"]
    cx = calib_info["left_rect"]["cx"]
    cy = calib_info["left_rect"]["cy"]

    ply_path = resolve_ply_path(model_dir)

    print("=" * 60)
    print("[render_gallery.py] GS gallery render")
    print(f"  model_dir   : {model_dir}")
    print(f"  ply_path    : {ply_path}")
    print(f"  width/height: {args.width} x {args.height}")
    print(f"  fx, fy      : {fx:.4f}, {fy:.4f}")
    print(f"  cx, cy      : {cx:.4f}, {cy:.4f}")
    print("=" * 60)

    if gaussians is None:
        gaussians = GaussianModel(args.sh_degree)
        gaussians.load_ply(str(ply_path), scale=1.0, use_train_test_exp=False)
        gaussians.freeze_except_pose()

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        gaussianR = GaussianRenderer(
            gaussians, K,
            np.array([0,0,0], dtype=np.float32) / 255.0,
            args.width, args.height, False
        )
    else:
        print("[render_gallery.py] Using pre-loaded GaussianModel (skipping PLY load)")

    gallery_bboxes = []

    # Sanity check: ensure pose indices are 0...N-1 without gaps
    nPoses = 0
    for pose in galleryPoses:
        idx = pose["index"]
        assert idx == nPoses, f"Expected pose index {nPoses}, got {idx}"
        nPoses += 1

    # Render each gallery pose with GSplat and save results
    print("GS gallery rendering - Cropped rendered image, depth map, XYZ map")
    gallery_bboxes = np.zeros((nPoses, 4), dtype=np.int32)
    for i, pose in enumerate(tqdm(galleryPoses, desc="GS gallery rendering")):
        g_crop_bbox = render_save(pose, gaussianR, K, args)
        gallery_bboxes[i] = g_crop_bbox        

    # save bounding boxes of gallery for later reference
    np.save(obj_dir / "g_bboxes.npy", gallery_bboxes)

    return gallery_bboxes


def extract_dino_features(args, gallery_bboxes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This renderer is expected to run on CUDA.")
    
    bbox_size = get_max_bbox_size(gallery_bboxes)
    
    extractor = DinoV2Extractor(args.dino_model, device=device) 
    dino_in_size = args.dino_input_size

    cache_dir = Path(args.obj_dir)
    print(f"  DINOv2 cache dir: {cache_dir}")

    n_horz_patches = 1 if bbox_size < dino_in_size else 2
    dino_out_size = 384 * n_horz_patches * n_horz_patches

    features = torch.zeros( size=[len(gallery_bboxes), dino_out_size], dtype=torch.float32 )  

    render_dir = Path(args.obj_dir) / "gallery"    
    for idx in tqdm(range(len(gallery_bboxes)), desc=f"DINO feature extraction (feat. dim: {dino_out_size})"):
        img_rgb_crop = load_rgb(render_dir / f"{idx:04d}.png")
        assert gallery_bboxes[idx][2] - gallery_bboxes[idx][0] == img_rgb_crop.shape[1] and gallery_bboxes[idx][3] - gallery_bboxes[idx][1] == img_rgb_crop.shape[0], f"Crop size mismatch: expected ({gallery_bboxes[idx][2] - gallery_bboxes[idx][0]}, {gallery_bboxes[idx][3] - gallery_bboxes[idx][1]}), got {img_rgb_crop.shape[1]}, {img_rgb_crop.shape[0]}"
        gallery_crop, _ = zeropad_square(img_rgb_crop, bbox_size)
        
        # 4 tiles of DINO input, each tile gets a quarter of the original bbox crop (with some shared margin)
        gallery_crop_dino = square_pad_resize(gallery_crop, dino_in_size * n_horz_patches)  
        if n_horz_patches == 1:
            feat = extractor.encode_rgb(gallery_crop_dino)
        else:
            feat = extractor.encode_4rgb(gallery_crop_dino)
        assert feat.shape == (dino_out_size,), f"Unexpected DINO feature shape: {feat.shape}"

        features[idx] = feat
        # feat = feat.numpy()
        # cache_feat_path = cache_dir / f"{idx:04d}_{args.dino_model.replace('/', '_')}.npy"
        # np.save(str(cache_feat_path), feat)

    features_np = features.numpy()
    np.save( str(cache_dir / f"g_features.npy"), features_np)

    print(f"Saved DINO features {features_np.shape}")
    
    return features # (3024, 384 * n_horz_patches ^ 2) CPU tensor


def main():
    args = parse_args()
    gallery_bboxes = run_render_gallery(args)
    # gallery_bboxes = np.load(Path(args.obj_dir) / "g_bboxes.npy")

    extract_dino_features(args, gallery_bboxes)

    print("=" * 60)
    print("[render_gallery.py] Done")
    print(f"  object_dir : {args.obj_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()