import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import kornia.feature as KF

from modules_6d.retrieval_dino import (
    ensure_dir,
    load_rgb,
    tight_crop_nonblack,
    square_pad_resize,
    save_json,
    DinoV2Extractor,
    compute_nonblack_bbox,
    expand_bbox,
    crop_with_bbox,
)


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 feature cache
# ─────────────────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_or_compute_gallery_features(
    gallery_files: list,
    extractor: DinoV2Extractor,
    cache_dir: Path,
    model_name: str,
    nonblack_thresh: int,
    crop_margin: int,
) -> dict:
    ensure_dir(cache_dir)
    cache_index_path = cache_dir / f"gallery_feat_index_{model_name.replace('/', '_')}.json"

    if cache_index_path.exists():
        with open(cache_index_path, "r") as f:
            cache_index = json.load(f)
    else:
        cache_index = {}

    results = {}
    updated = False

    for gp in gallery_files:
        key = gp.name
        file_hash = _file_hash(gp)
        cache_feat_path = cache_dir / f"{gp.stem}_{model_name.replace('/', '_')}.npy"

        if (key in cache_index
                and cache_index[key].get("hash") == file_hash
                and cache_feat_path.exists()):
            feat = np.load(str(cache_feat_path))
            print(f"  [Cache HIT ] {key}")
        else:
            gimg = load_rgb(gp)
            gh, gw = gimg.shape[:2]
            gbox = compute_nonblack_bbox(gimg, thresh=nonblack_thresh)
            gbox = expand_bbox(gbox, crop_margin, gw, gh)
            gcrop = crop_with_bbox(gimg, gbox)
            gin = square_pad_resize(gcrop, 224)
            feat_tensor = extractor.encode_bgr(gin)
            feat = feat_tensor.numpy()

            np.save(str(cache_feat_path), feat)
            cache_index[key] = {"hash": file_hash}
            updated = True
            print(f"  [Cache MISS] {key} → computed & saved")

        results[key] = {"feat": feat, "path": str(gp)}

    if updated:
        with open(cache_index_path, "w") as f:
            json.dump(cache_index, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# LoFTR helpers
# ─────────────────────────────────────────────────────────────────────────────

def image_to_loftr_tensor(img_bgr, device="cuda"):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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


def estimate_inliers(mkpts0, mkpts1, ransac_thresh=3.0):
    if len(mkpts0) < 4:
        return None, np.zeros(len(mkpts0), dtype=bool)
    H, mask = cv2.findHomography(
        mkpts0, mkpts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )
    if mask is None:
        return H, np.zeros(len(mkpts0), dtype=bool)
    return H, mask.ravel().astype(bool)


def unmap_from_square_resize(pts_resized, orig_hw, resize_target=840):
    h, w = orig_hw
    side = max(h, w)
    x0 = (side - w) // 2
    y0 = (side - h) // 2
    pts_square = np.asarray(pts_resized, dtype=np.float64) * (side / resize_target)
    return pts_square - np.array([[x0, y0]], dtype=np.float64)


def save_best_match_data(
    out_dir, mkpts0_all, mkpts1_all, conf_all, inlier_mask,
    query_hw, query_nonblack_bbox_xyxy,
    gallery_crop_hw, gallery_nonblack_bbox_xyxy, gallery_img_hw,
    loftr_resize_target=840,
):
    ensure_dir(out_dir)
    out_dir = Path(out_dir)

    inlier_pts0 = mkpts0_all[inlier_mask].astype(np.float32)
    inlier_pts1 = mkpts1_all[inlier_mask].astype(np.float32)
    inlier_conf = conf_all[inlier_mask].astype(np.float32)

    npz_path = out_dir / "loftr_best_match_data.npz"
    np.savez(str(npz_path),
             mkpts0_inlier_840=inlier_pts0,
             mkpts1_inlier_840=inlier_pts1,
             conf_inlier=inlier_conf)

    meta = {
        "loftr_resize_target": loftr_resize_target,
        "query_crop_hw": list(query_hw),
        "query_nonblack_bbox_xyxy": list(query_nonblack_bbox_xyxy),
        "gallery_crop_hw": list(gallery_crop_hw),
        "gallery_nonblack_bbox_xyxy": [int(v) for v in gallery_nonblack_bbox_xyxy],
        "gallery_img_hw": list(gallery_img_hw),
    }
    save_json(out_dir / "loftr_best_match_meta.json", meta)
    print(f"  [LoFTR] Saved {int(inlier_mask.sum())} inlier match points → {npz_path.name}")
    return npz_path


def get_mask_inlier_indices(mkpts0_full, mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.ones(len(mkpts0_full), dtype=bool)

    h, w = mask.shape
    keep_mask = []
    for pt in mkpts0_full:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            keep_mask.append(True)
        else:
            keep_mask.append(False)
            
    return np.array(keep_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def run_step4_dino_loftr_rerank_rt(args, model_cache=None):

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    device = args.device if torch.cuda.is_available() else "cpu"

    # ── 1. Query load (full resolution, no crop) ──────────────────────────────
    query_masked_full_path = Path(args.out_dir) / "query_masked_full.png"
    if not query_masked_full_path.exists():
        print(f"  [WARN] query_masked_full.png not found, falling back to query_masked_path")
        query_masked_full_path = Path(args.query_masked_path)

    query_full = load_rgb(str(query_masked_full_path))
    qh, qw = query_full.shape[:2]
    print(f"  Query full image: {qw}x{qh}")

    LOFTR_SIZE = 840
    step1_json_path = Path(args.out_dir) / "step1_result.json"
    q_loftr_bbox = None
    if step1_json_path.exists():
        with open(step1_json_path) as f:
            step1_data = json.load(f)
        raw_bbox = step1_data.get("mask_bbox_xyxy") or step1_data.get("bbox_xyxy")
        if raw_bbox is not None:
            q_loftr_bbox = list(expand_bbox(raw_bbox, args.crop_margin, qw, qh))
            print(f"  [LoFTR] Query crop from step1 bbox: {q_loftr_bbox}")

    if q_loftr_bbox is None:
        q_loftr_bbox = [0, 0, qw, qh]
        print(f"  [LoFTR] step1 bbox not found, using full image")

    query_loftr_crop = crop_with_bbox(query_full, q_loftr_bbox)
    q_loftr_h, q_loftr_w = query_loftr_crop.shape[:2]
    query_l = square_pad_resize(query_loftr_crop, LOFTR_SIZE)
    print(f"  [LoFTR] Query crop size: {q_loftr_w}x{q_loftr_h} → {LOFTR_SIZE}px")

    # ── 2. Gallery file list ──────────────────────────────────────────────────
    gallery_dir = Path(args.gallery_dir)
    # Use cached file list if available (avoids per-frame OS iterdir scan)
    if model_cache is not None and model_cache.gallery_files is not None:
        gallery_files = model_cache.gallery_files
    else:
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        gallery_files = sorted([p for p in gallery_dir.iterdir() if p.suffix.lower() in exts])
    if not gallery_files:
        raise FileNotFoundError(f"No gallery images in: {gallery_dir}")
    print(f"  Gallery: {len(gallery_files)} images in {gallery_dir}")

    # ── 3. DINOv2 extractor + gallery features ────────────────────────────────
    # Use pre-loaded extractor if available, otherwise load fresh
    if model_cache is not None and model_cache.dino is not None:
        extractor = model_cache.dino
    else:
        extractor = DinoV2Extractor(args.dino_model, device=device)

    qbox = compute_nonblack_bbox(query_full, thresh=args.nonblack_thresh)
    qbox = expand_bbox(qbox, args.crop_margin, qw, qh)
    query_crop_for_dino = crop_with_bbox(query_full, qbox)
    query_dino_in = square_pad_resize(query_crop_for_dino, args.dino_input_size)
    qfeat = extractor.encode_bgr(query_dino_in)

    # Use in-memory gallery features if pre-loaded, otherwise compute from disk cache
    if model_cache is not None and model_cache.gallery_feats is not None:
        gallery_feats = model_cache.gallery_feats
        print(f"  DINOv2 gallery features: using in-memory cache ({len(gallery_feats)} images)")
    else:
        if getattr(args, "dino_cache_dir", None):
            cache_dir = Path(args.dino_cache_dir)
        elif (Path(args.out_dir).parent.parent / "can_data").exists():
            cache_dir = Path(args.out_dir).parent.parent / "can_data" / "dino_cache_3dgs_1920"
        else:
            cache_dir = Path(args.out_dir) / "dino_cache"
        print(f"  DINOv2 cache dir: {cache_dir}")
        gallery_feats = load_or_compute_gallery_features(
            gallery_files=gallery_files,
            extractor=extractor,
            cache_dir=cache_dir,
            model_name=args.dino_model,
            nonblack_thresh=args.nonblack_thresh,
            crop_margin=args.crop_margin,
        )

    # ── 4. DINOv2 cosine similarity → top-K ──────────────────────────────────
    if model_cache is not None and model_cache.gallery_feat_matrix is not None:
        # Vectorized: (N, D) @ (D,) = (N,) — single matmul replaces Python loop
        qfeat_np = qfeat.numpy() if hasattr(qfeat, "numpy") else np.asarray(qfeat)
        scores_arr = model_cache.gallery_feat_matrix @ qfeat_np
        scores = [
            {"file": gp.name, "path": str(gp), "score_cosine": float(scores_arr[i])}
            for i, gp in enumerate(gallery_files)
        ]
    else:
        scores = []
        for gp in gallery_files:
            key = gp.name
            gfeat = torch.from_numpy(gallery_feats[key]["feat"])
            score = float(torch.dot(qfeat, gfeat).item())
            scores.append({"file": key, "path": str(gp), "score_cosine": score})

    scores_sorted = sorted(scores, key=lambda x: x["score_cosine"], reverse=True)
    topk_items = scores_sorted[:args.topk]
    print(f"  DINOv2 top-{args.topk}: {[x['file'] for x in topk_items]}")


    # ── 5. LoFTR rerank ───────────────────────────────────────────────────────
    # Use pre-loaded LoFTR if available, otherwise load fresh
    if model_cache is not None and model_cache.loftr is not None:
        matcher = model_cache.loftr
    else:
        matcher = KF.LoFTR(pretrained=args.loftr_pretrained).to(device).eval()

    results = []
    match_cache = {}

    query_mask_path = Path(args.out_dir) / "query_mask.png"

    for item in topk_items:
        gpath = Path(item["path"])
        gimg = load_rgb(str(gpath))
        gh, gw = gimg.shape[:2]

        # gallery: nonblack bbox crop → LoFTR 입력
        g_loftr_bbox = list(compute_nonblack_bbox(gimg, thresh=args.nonblack_thresh))
        g_loftr_bbox = list(expand_bbox(g_loftr_bbox, args.crop_margin, gw, gh))
        gallery_loftr_crop = crop_with_bbox(gimg, g_loftr_bbox)
        g_loftr_h, g_loftr_w = gallery_loftr_crop.shape[:2]
        gallery_l = square_pad_resize(gallery_loftr_crop, LOFTR_SIZE)
        mkpts0, mkpts1, conf = compute_loftr_matches(matcher, query_l, gallery_l, device=device)

        valid = conf >= args.loftr_conf_thresh
        mkpts0_v, mkpts1_v, conf_v = mkpts0[valid], mkpts1[valid], conf[valid]

        # Filter by query mask
        m0_crop_temp = unmap_from_square_resize(mkpts0_v, (q_loftr_h, q_loftr_w), resize_target=LOFTR_SIZE)
        m0_full_temp = m0_crop_temp + np.array([[q_loftr_bbox[0], q_loftr_bbox[1]]])
        mask_keep = get_mask_inlier_indices(m0_full_temp, query_mask_path)
        mkpts0_v = mkpts0_v[mask_keep]
        mkpts1_v = mkpts1_v[mask_keep]
        conf_v = conf_v[mask_keep]

        H_mat, inlier_mask = estimate_inliers(mkpts0_v, mkpts1_v, ransac_thresh=args.loftr_ransac_thresh)

        total_matches = len(conf_v)
        inliers = int(inlier_mask.sum()) if total_matches > 0 else 0
        mean_conf = float(conf_v.mean()) if total_matches > 0 else 0.0
        inlier_ratio = float(inliers / total_matches) if total_matches > 0 else 0.0

        results.append({
            "file": item["file"],
            "path": item["path"],
            "dino_score": item["score_cosine"],
            "total_matches": total_matches,
            "inliers": inliers,
            "mean_conf": mean_conf,
            "inlier_ratio": inlier_ratio,
        })

        match_cache[item["file"]] = {
            "mkpts0_all":  mkpts0_v,
            "mkpts1_all":  mkpts1_v,
            "conf_all":    conf_v,
            "inlier_mask": inlier_mask,
            "q_loftr_bbox":  q_loftr_bbox,
            "q_loftr_hw":    (q_loftr_h, q_loftr_w),
            "g_loftr_bbox":  g_loftr_bbox,
            "g_loftr_hw":    (g_loftr_h, g_loftr_w),
            "gimg_hw":       (gh, gw),
            "query_l":       query_l,
            "gallery_l":     gallery_l,
            "gimg":          gimg,
        }

    # ── 6. LoFTR score sort → best ────────────────────────────────────────────
    max_inliers = max([r["inliers"] for r in results] + [1])
    for r in results:
        norm_inliers = r["inliers"] / max_inliers
        r["loftr_score"] = 0.5 * norm_inliers + 0.3 * r["mean_conf"] + 0.2 * r["inlier_ratio"]

    results = sorted(results, key=lambda x: x["loftr_score"], reverse=True)
    best = results[0]

    # ── 7. Save best match data ───────────────────────────────────────────────
    best_cache = match_cache[best["file"]]
    save_best_match_data(
        out_dir=out_dir,
        mkpts0_all=best_cache["mkpts0_all"],
        mkpts1_all=best_cache["mkpts1_all"],
        conf_all=best_cache["conf_all"],
        inlier_mask=best_cache["inlier_mask"],
        query_hw=best_cache["q_loftr_hw"],
        query_nonblack_bbox_xyxy=best_cache["q_loftr_bbox"],
        gallery_crop_hw=best_cache["g_loftr_hw"],
        gallery_nonblack_bbox_xyxy=best_cache["g_loftr_bbox"],
        gallery_img_hw=best_cache["gimg_hw"],
        loftr_resize_target=LOFTR_SIZE,
    )

    summary = {
        "stage": "step5_dino_loftr",
        "sim_method": "dino_loftr",
        "query_masked_full": str(query_masked_full_path),
        "gallery_dir": str(gallery_dir),
        "dino_model": args.dino_model,
        "loftr_pretrained": args.loftr_pretrained,
        "loftr_conf_thresh": args.loftr_conf_thresh,
        "loftr_ransac_thresh": args.loftr_ransac_thresh,
        "best_render": best["file"],
        "best_loftr_score": best["loftr_score"],
        "results_sorted": results,
        "loftr_best_match_npz": str(out_dir / "loftr_best_match_data.npz"),
        "loftr_best_match_meta": str(out_dir / "loftr_best_match_meta.json"),
    }
    save_json(out_dir / "loftr_scores.json", summary)

    print("=" * 60)
    print("[Step 5 RT] DINOv2 + LoFTR rerank complete")
    print(f"  best_render     : {best['file']}")
    print(f"  best_loftr_score: {best['loftr_score']:.4f}")
    print("=" * 60)
