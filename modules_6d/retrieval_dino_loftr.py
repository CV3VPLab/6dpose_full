"""
retrieval_dino_loftr.py
=======================
step5: DINOv2 LoFTR 
"""

import time
import json
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import kornia.feature as KF

from .retrieval_dino import (
    DinoV2Extractor
)

from modules_6d.io_utils import (
    ensure_dir
)
from utils.image_utils import (
    crop_with_bbox,
    get_max_bbox_size,
    square_bbox,
    square_pad_resize,
    load_rgb,
    zeropad_square,
    unmap_to_full_image,
    construct_queryInfo,
    construct_galleryInfo
)

TIME_CHECK = False        

# ─────────────────────────────────────────────────────────────────────────────
# LoFTR helpers
# ─────────────────────────────────────────────────────────────────────────────

def image_to_loftr_tensor(img_bgr, device="cuda"):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
    x = torch.from_numpy(gray).float()[None, None] / 255.0
    return x.to(device)


def compute_loftr_matches(matcher, img0_bgr, img1_bgr, device="cuda"):
    inp = {
        "image0": image_to_loftr_tensor(img0_bgr, device=device),
        "image1": image_to_loftr_tensor(img1_bgr, device=device),
    }
    with torch.no_grad():
        out = matcher(inp)
    mkpts0 = out["keypoints0"].cpu().numpy()
    mkpts1 = out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()
    return mkpts0, mkpts1, conf


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


def draw_loftr_matches(img0, img1, mkpts0, mkpts1, conf, out_path, max_draw=400):
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

    cv2.imwrite(str(out_path), canvas)


def save_best_match_data(
        npz_path,
        pts0,
        pts1,
        conf,
        best_idx
    ):
    ensure_dir(npz_path.parent)    

    assert pts0.dtype == np.float32 and pts1.dtype == np.float32 and conf.dtype == np.float32, "Expected float32 arrays"

    np.savez(str(npz_path),
             mkpts0 = pts0,
             mkpts1 = pts1,
             conf = conf,
             best_idx = best_idx)



# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def draw_loftr_matches_full(img0_full, img1_full,
                            mkpts0_crop, mkpts1_crop,
                            q_bbox, g_bbox,
                            out_path,
                            loftr_size=840, max_draw=400):

    m0_full = unmap_to_full_image(mkpts0_crop, q_bbox, loftr_size)
    m1_full = unmap_to_full_image(mkpts1_crop, g_bbox, loftr_size)

    h0, w0 = img0_full.shape[:2]
    h1, w1 = img1_full.shape[:2]

    if h0 > h1:
        img0_vis = cv2.resize(img0_full, (int(w0 * h1 / h0), h1))
        img1_vis = img1_full.copy()
        sc0, sc1 = h1 / h0, 1.0
    else:
        img0_vis = img0_full.copy()
        img1_vis = cv2.resize(img1_full, (int(w1 * h0 / h1), h0))
        sc0, sc1 = 1.0, h0 / h1

    H_vis = img0_vis.shape[0]
    W0_vis, W1_vis = img0_vis.shape[1], img1_vis.shape[1]
    canvas = np.zeros((H_vis, W0_vis + W1_vis, 3), dtype=np.uint8)
    canvas[:, :W0_vis] = img0_vis
    canvas[:, W0_vis:] = img1_vis
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    n = len(mkpts0_crop)
    if n > max_draw:
        shuffle_idx = np.random.permutation(n)
        idx = shuffle_idx[:max_draw]
    else:
        idx = np.arange(n)

    for i in idx:
        p0 = (int(round(m0_full[i, 0] * sc0)), int(round(m0_full[i, 1] * sc0)))
        p1 = (int(round(m1_full[i, 0] * sc1 + W0_vis)), int(round(m1_full[i, 1] * sc1)))
        color = (0, 255, 0)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, (0, 255, 255), -1)
        cv2.circle(canvas, p1, 3, (0, 255, 255), -1)

    cv2.putText(canvas, f"Full image  inliers={n}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 230), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def retrieve_topk(query, query_mask, q_bbox, g_bbox_size, gfeats, extractor, dino_size, topk):
    # KSCHOI TODO: occlusion 상황에서 topk가 의미 있는지 확인 (현재는 DINO feature similarity top-1만 사용함)

    # query bounding box can be larger than gallery bbox, so we use the max of both for square cropping
    q_bbox_size = max( g_bbox_size, q_bbox[2] - q_bbox[0], q_bbox[3] - q_bbox[1] )
    q_bbox_ext = square_bbox(q_bbox, q_bbox_size)
    query_crop = crop_with_bbox(query, q_bbox_ext)
    
    # A cropped query image for extracting DINOv2 feature
    query_dino_in = square_pad_resize(query_crop, dino_size * 2)
    qfeat = extractor.encode_4rgb(query_dino_in)
    # qfeat = extractor.encode_rgb(query_dino_in)    
    
    scores = (gfeats @ qfeat).numpy()
    topk_items = np.argsort(scores)[-topk:][::-1]

    return topk_items, scores[topk_items], query_crop, q_bbox_ext


def retrieve_match(query_info, gallery_info, feat_extractor_info, matcher_info, out_dir=None):
    # KSCHOI TODO: occlusion 상황에서 topk가 의미 있는지 확인 (현재는 DINO feature similarity top-1만 사용함)
    query = query_info["query_full"]
    query_mask = query_info["query_mask"]
    q_bbox = query_info["q_bbox"]
    
    gallery_crops = gallery_info["gallery_crops"]
    g_bboxes = gallery_info["g_bboxes"]
    gfeats = gallery_info["gfeats"]

    extractor = feat_extractor_info["extractor"]
    dino_size = feat_extractor_info["input_size"]
    topk = feat_extractor_info["topk"]
    
    matcher = matcher_info["matcher"]
    conf_thresh = matcher_info["conf_thresh"]
    device = matcher_info["device"]

    bbox_size = get_max_bbox_size(g_bboxes)

    if TIME_CHECK:
        start_time = time.perf_counter()

    topk_items, scores, query_crop, q_bbox_ext = retrieve_topk(query, query_mask, q_bbox, bbox_size, gfeats, extractor, dino_size, topk)

    if TIME_CHECK:
        end_time = time.perf_counter()
        print(f"  Feature-based Retrieval time : {end_time - start_time:.6f} seconds")

    item = topk_items[0]

    g_bbox = g_bboxes[item]
    g_bbox_ext = square_bbox(g_bbox, bbox_size)

    g_crop_hw = gallery_crops[item].shape[:2]
    assert g_bbox[2] - g_bbox[0] == g_crop_hw[1] and g_bbox[3] - g_bbox[1] == g_crop_hw[0], f"Gallery crop size mismatch with bbox, check loading and cropping logic for gallery item {item}"

    gallery_crop = zeropad_square(gallery_crops[item], bbox_size)

    # ── LoFTR rerank ───────────────────────────────────────────────────────
    LOFTR_SIZE = 840
    query_l = square_pad_resize(query_crop, LOFTR_SIZE)
    gallery_l = square_pad_resize(gallery_crop, LOFTR_SIZE)
    
    # mkpts0: query crop 좌표계, mkpts1: gallery crop 좌표계, conf: 매칭 confidence
    if TIME_CHECK:  
        start_time = time.perf_counter()
    
    mkpts0, mkpts1, conf = compute_loftr_matches(matcher, query_l, gallery_l, device=device)

    if TIME_CHECK:
        end_time = time.perf_counter()
        print(f"  LoFTR matching time : {end_time - start_time:.6f} seconds")

    valid = conf >= conf_thresh
    mkpts0, mkpts1, conf = mkpts0[valid], mkpts1[valid], conf[valid]

    # 마스크 필터링: query crop coords → full image coords → mask 체크
    pts0_full = unmap_to_full_image(mkpts0, q_bbox_ext, LOFTR_SIZE)
    mask_keep = get_mask_inlier_indices(pts0_full, query_mask)

    mkpts0 = mkpts0[mask_keep]
    mkpts1 = mkpts1[mask_keep]
    conf   = conf[mask_keep]

    pts0_full = pts0_full[mask_keep]
    pts1_full = unmap_to_full_image(mkpts1, g_bbox_ext, LOFTR_SIZE)

    # crop 공간 시각화
    if out_dir is not None:
        vis_path = out_dir / f"loftr_vis_{item:04d}.png"
        draw_loftr_matches(query_l, gallery_l, mkpts0, mkpts1, conf, vis_path)

    return pts0_full, pts1_full, conf, item


def  run_step5_dino_loftr_rerank(args):
    """
    DINOv2 retrieval + LoFTR rerank 통합.
    query_masked_path: query_masked_full.png (full 해상도, 배경 검정)
    query_image: RGB image 가정 (sensor로부터 얻어질 때 RGB라 가정)
    """
    if args.dino_scores_json is None:
        pass

    device = args.device if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    data_dir = Path(args.out_dir).parent.parent
    ensure_dir(data_dir)
    
    # DINOv2 extractor 초기화
    extractor = DinoV2Extractor(args.dino_model, device=device)
    # LoFTR 초기화
    matcher = KF.LoFTR(pretrained=args.loftr_pretrained).to(device).eval()
    
    #######################################################
    # Inputs for LoFTR reranking
    # Data ::
    #######################################################
    query_info = construct_queryInfo(out_dir)
    #     "query_full": query_full,       # full query image (RGB)
    #     "query_mask": mask,             # full query mask (grayscale, 0=background, 255=foreground)
    #     "q_bbox": q_bbox                # query bounding box
    gallery_info = construct_galleryInfo(data_dir / "can_data")
    #     "gallery_crops": gallery_crops, # cropped gallery images according to g_bboxes
    #     "g_bboxes": g_bboxes,           # all the bounding boxes of gallery renders
    #     "gfeats": gfeats                # DINOv2 features of gallery crops, used for cosine similarity retrieval  
    
    #######################################################
    # Inputs for LoFTR reranking
    # Model ::
    #######################################################
    feat_extractor_info = {
        "extractor": extractor,                 # DINOv2 feature extractor instance, used for encoding query crop
        "input_size": args.dino_input_size,
        "topk": args.topk   
    }
    matcher_info = {
        "matcher": matcher,                     # LoFTR matcher instance, used for computing matches between query crop and gallery crop   
        "conf_thresh": args.loftr_conf_thresh,
        "device": device
    }

    if TIME_CHECK:
        start_time = time.perf_counter()

    pts0_full, pts1_full, conf, best_idx = retrieve_match(query_info, gallery_info, feat_extractor_info, matcher_info, out_dir=None)

    if TIME_CHECK:
        end_time = time.perf_counter()
        print(f"  Feature-based Retrieval & LoFTR matching time : {end_time - start_time:.6f} seconds")

    save_best_match_data(
        npz_path = out_dir / "loftr_best_match_data.npz",
        pts0 = pts0_full,
        pts1 = pts1_full,
        conf = conf,
        best_idx = best_idx
    )

    print("=" * 60)
    print("[Step 5] DINOv2 + LoFTR rerank complete")
    print(f"  best_render     : {best_idx}")
    print(f"  num_matches     : {len(pts0_full)}")
    print("=" * 60)

