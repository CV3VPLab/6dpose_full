import argparse
import warnings
import json
import joblib  
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.rich import tqdm
from tqdm import TqdmExperimentalWarning


import numpy as np
import torch
import cv2
from sklearn.decomposition import PCA

from gaussian_renderer    import GaussianModel
from refine_pose          import GaussianRenderer
from estimate_object_pose import init_gaussians

from modules_6d.retrieval_dino import load_extractor, preprocess_for_dinov2

from utils.general_utils import sync_time

from utils.io_utils import (
    ensure_dir, 
    resolve_ply_path,
    load_json,
    load_intrinsics, K_to_params, params_to_K,
    get_obj_path, get_K_path,
    get_named_config
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

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def render_save(pose_info, gaussianRenderer: GaussianRenderer, K, obj_name, bsave_depth=False):
    obj_dir    = get_obj_path(obj_name, "object")        
    render_dir = get_obj_path(obj_name, "gallery") 
    xyz_dir    = get_obj_path(obj_name, "xyz")
    
    if bsave_depth:
        depth_dir = get_obj_path(obj_name, "depth")

    idx = pose_info["index"]        
    assert idx < 10000 # max file number : 9999

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

    # XYZ map, optionally depth map
    depth_np = t2np(_r[0][..., 3]) * mask

    xyz_obj = depth_to_xyz_map(
        depth_np=depth_np,
        fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2],
        R_obj_to_cam=pose_info["R_obj_to_cam"],
        t_obj_to_cam=pose_info["t_obj_to_cam"]
    )
    xyz_obj_crop = crop_with_bbox(xyz_obj, g_crop_bbox)            
    np.save(str(xyz_dir / f"{idx:04d}.npy"), xyz_obj_crop)

    if bsave_depth:
        # save cropped depth data
        depth_crop = crop_with_bbox(depth_np, g_crop_bbox)
        fn_crop = depth_dir / f"{idx:04d}.npy"
        np.save(str(fn_crop), depth_crop)
        
    return g_crop_bbox, mask_crop


def run_render_gallery(config):
    """
    Run gallery rendering.
    Pass pre-loaded gaussians to skip model loading (for in-process preloading).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This renderer is expected to run on CUDA.")

    obj_config = get_named_config(config["objects"])
    obj_name   = obj_config["name"]
    obj_params = obj_config["params"]
    
    # folder check
    obj_dir    = get_obj_path(obj_name, "object")    
    model_dir  = get_obj_path(obj_name, "model")
    render_dir = get_obj_path(obj_name, "gallery") 
    xyz_dir    = get_obj_path(obj_name, "xyz")

    ensure_dir(obj_dir)
    ensure_dir(model_dir)
    ensure_dir(render_dir)
    ensure_dir(xyz_dir)
        
    bsave_depth = config["gallery"]["save_depth"]
    if bsave_depth == True:        
        depth_dir = get_obj_path(obj_name, "depth")
        ensure_dir(depth_dir)

    # gallery pose loading
    gallery_pose_path = obj_dir.parent.parent / "gallery_poses.json"
    galleryPoses = load_json(gallery_pose_path)["poses"]

    # fx, fy, cx, cy = K_to_params(load_intrinsics(args.intrinsics_path))
    calib_info = load_json(get_K_path(config["inputs"][config["input"]]))
    fx = calib_info["left_rect"]["fx"]
    fy = calib_info["left_rect"]["fy"]
    cx = calib_info["left_rect"]["cx"]
    cy = calib_info["left_rect"]["cy"]

    ply_path = resolve_ply_path(model_dir)

    img_width = config["renderer"]["options"]["width"]
    img_height = config["renderer"]["options"]["height"]

    print("=" * 60)
    print("[render_gallery.py] GS gallery render")
    print(f"  model_dir   : {model_dir}")
    print(f"  ply_path    : {ply_path}")
    print(f"  width/height: {img_width} x {img_height}")
    print(f"  fx, fy      : {fx:.4f}, {fy:.4f}")
    print(f"  cx, cy      : {cx:.4f}, {cy:.4f}")
    print("=" * 60)

    K = params_to_K(fx, fy, cx, cy)
    gaussianR = GaussianRenderer(
        init_gaussians(config["renderer"], ply_path, scale=1.0), 
        K, np.array([0,0,0], dtype=np.float32) / 255.0,
        img_width, img_height, False
    )    

    gallery_bboxes = []

    # Sanity check: ensure pose indices are 0...N-1 without gaps
    nPoses = 0
    for pose in galleryPoses:
        idx = pose["index"]
        assert idx == nPoses, f"Expected pose index {nPoses}, got {idx}"
        nPoses += 1

    # Render each gallery pose with GSplat and save results
    print(f"GS gallery rendering - Cropped rendered image, XYZ map{', depth map' if bsave_depth else ''}")
    gallery_bboxes = np.zeros((nPoses, 4), dtype=np.int32)
    masks = []
    for i, pose in enumerate(tqdm(galleryPoses, desc="GS gallery rendering")):
        g_crop_bbox, mask_crop = render_save(pose, gaussianR, K, obj_name, bsave_depth)
        gallery_bboxes[i] = g_crop_bbox 
        masks.append(mask_crop)       

    # save bounding boxes of gallery for later reference
    np.save(obj_dir / "g_bboxes.npy", gallery_bboxes)

    return gallery_bboxes, masks


def extract_dino_features(gallery_bboxes, extractor_config, obj_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("This renderer is expected to run on CUDA.")
    device = "cuda"

    bbox_size = get_max_bbox_size(gallery_bboxes)
    
    extractor, ext_options = load_extractor(extractor_config)

    dino_in_size = ext_options["input_size"]

    cache_dir = obj_dir
    print(f"  DINOv2 cache dir: {cache_dir}")

    n_horz_patches = 1 if bbox_size < dino_in_size else 2
    dino_out_size = 384 * n_horz_patches * n_horz_patches

    features = torch.zeros( size=[len(gallery_bboxes), dino_out_size], dtype=torch.float32 )  

    render_dir = Path(obj_dir) / "gallery"    
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

        features[idx] = feat.cpu()        

    feature_data = [extractor_config["name"], features.numpy()]
    np.savez_compressed( str(cache_dir / f"g_features.npz"), *feature_data)

    print(f"Saved DINO features {feature_data[1].shape}")
    
    return feature_data # (3024, 384 * n_horz_patches ^ 2) CPU tensor


def extract_dino_masking_features(gallery_bboxes, masks, extractor_config, obj_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("This renderer is expected to run on CUDA.")
    device = "cuda"
    
    bbox_size = get_max_bbox_size(gallery_bboxes)
    
    extractor, ext_options = load_extractor(extractor_config)
    dino_in_size = ext_options["input_size"]

    cache_dir = obj_dir
    print(f"  DINOv2 cache dir: {cache_dir}")

    n_horz_patches = 1 if bbox_size < dino_in_size else 2    

    features = []

    for idx in tqdm(range(len(gallery_bboxes)), desc="DINO masked feature extraction"):
        img_rgb_crop = load_rgb( obj_dir / "gallery" / f"{idx:04d}.png")
        assert gallery_bboxes[idx][2] - gallery_bboxes[idx][0] == img_rgb_crop.shape[1] and gallery_bboxes[idx][3] - gallery_bboxes[idx][1] == img_rgb_crop.shape[0], f"Crop size mismatch: expected ({gallery_bboxes[idx][2] - gallery_bboxes[idx][0]}, {gallery_bboxes[idx][3] - gallery_bboxes[idx][1]}), got {img_rgb_crop.shape[1]}, {img_rgb_crop.shape[0]}"
        assert img_rgb_crop.shape[:2] == masks[idx].shape, f"Mask shape mismatch: expected {img_rgb_crop.shape[:2]}, got {masks[idx].shape}"
        
        gallery_crop, _ = zeropad_square(img_rgb_crop, bbox_size)        
        gmask_crop,   _ = zeropad_square(masks[idx],   bbox_size)
        
        gallery_crop_dino = square_pad_resize(gallery_crop, dino_in_size * n_horz_patches)  
        gmask_crop_dino   = square_pad_resize(gmask_crop,   dino_in_size * n_horz_patches)  
        gal_t, gmsk_t = preprocess_for_dinov2(gallery_crop_dino, gmask_crop_dino)  # Preprocess the image and mask for DINOv2
        feat = extractor.extract_masked_patch_tokens(gal_t.to(device), gmsk_t.to(device))[0]  
        features.append( t2np(feat) )

    # dimension reduction using PCA
    sample_ratio = 0.1  # 전체 토큰의 10%만 사용하여 PCA 피팅
    sampled_tokens = []
    for tokens in features:
        num_samples = max(1, int(len(tokens) * sample_ratio))
        indices = np.random.choice(len(tokens), num_samples, replace=False)
        sampled_tokens.append(tokens[indices])

    train_tokens = np.vstack(sampled_tokens) # [N_sampled_total, 384]

    # normalization for cosine similarity
    train_tokens = train_tokens / np.linalg.norm(train_tokens, axis=1, keepdims=True)

    print(f"PCA 피팅에 사용될 토큰 개수: {train_tokens.shape[0]}")
    pca = PCA(n_components=64)
    pca.fit(train_tokens)

    # PCA 모델 저장
    joblib.dump(pca, str(cache_dir / "dinov2_pca_64.pkl"))

    reduced_gallery_list = [extractor_config["name"]]
    for tokens in features:
        tokens_norm = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
        tokens_reduced = pca.transform(tokens_norm)
        tokens_reduced_norm = tokens_reduced / np.linalg.norm(tokens_reduced, axis=1, keepdims=True)
        
        reduced_gallery_list.append(tokens_reduced_norm.astype(np.float16))
        
    np.savez_compressed( str(cache_dir / "g_masked_features.npz"), *reduced_gallery_list)
    
    print("Saved DINO features")
    
    return reduced_gallery_list 


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="ope_config.json")
    args = parser.parse_args()
    config_fn = args.config if Path(args.config).is_file() else "ope_config.json"

    config = load_json(config_fn)
    
    obj_config = get_named_config(config["objects"])
    obj_dir    = get_obj_path(obj_config["name"], "object")    

    gallery_bboxes, masks = run_render_gallery(config)
    # gallery_bboxes = np.load(Path(obj_dir) / "g_bboxes.npy")

    ext_config = get_named_config(config["feat_extractors"])
    ext_name   = ext_config["name"]

    print("=" * 60)
    print(f"  Feature Extractor : {ext_name}") 
    print(f"  object(cache)_dir : {obj_dir}")  
    print("=" * 60)

    if ext_name == "DINOv2_MASK":
        extract_dino_masking_features(gallery_bboxes, masks, ext_config, obj_dir)
    else:
        extract_dino_features(gallery_bboxes, ext_config, obj_dir)

    print("=" * 60)
    print("[render_gallery.py] Done")
    print("=" * 60)


if __name__ == "__main__":
    main()