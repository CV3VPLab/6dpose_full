"""
retrieval_dino_loftr.py
=======================
step5: DINOv2 retrieval + LoFTR rerank 통합 버전

변경사항:
  - sim_method=dino_loftr 하나로 dino 계산 → loftr rerank 한번에 처리
  - gallery DINOv2 feature를 data/can_data/dino_cache/ 에 캐싱 (재실행 시 재사용)
  - query 쪽 tight_crop_nonblack 제거 → query_masked_full.png (full 해상도) 그대로 사용
    (좌표계 일관성: 모든 query 좌표가 full image 기준)
  - gallery 쪽은 기존과 동일하게 tight_crop_nonblack 유지
    (gallery render는 배경이 검정이라 crop이 유효)
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import kornia.feature as KF

from .retrieval_dino import (
    ensure_dir,
    load_rgb,
    tight_crop_nonblack,
    square_pad_resize,
    save_json,
    make_query_vs_best_image,
    DinoV2Extractor,
    compute_nonblack_bbox,
    expand_bbox,
    crop_with_bbox,
)


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 feature cache
# ─────────────────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    """파일 내용 기반 MD5 해시 (캐시 유효성 확인용)"""
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
    """
    gallery 이미지들의 DINOv2 feature를 캐시에서 로드하거나 계산해서 저장.
    DINO 입력용으로는 nonblack crop 사용 (similarity 품질을 위해).
    LoFTR에는 full 이미지를 사용하므로 crop 좌표는 저장하지 않음.

    cache_dir: data/can_data/dino_cache/
    반환: {filename: {"feat": np.ndarray(384,), "path": str}}
    """
    ensure_dir(cache_dir)
    cache_index_path = cache_dir / f"gallery_feat_index_{model_name.replace('/', '_')}.json"

    # 기존 캐시 인덱스 로드
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

        # 캐시 hit: 파일 해시가 같고 feature 파일이 존재
        if (key in cache_index
                and cache_index[key].get("hash") == file_hash
                and cache_feat_path.exists()):
            feat = np.load(str(cache_feat_path))
            print(f"  [Cache HIT ] {key}")
        else:
            # 캐시 miss: full 이미지 → nonblack crop → DINO (crop은 DINO 입력용으로만)
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
        print(f"  [Cache] index updated → {cache_index_path.name}")

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


def draw_loftr_matches(img0, img1, mkpts0, mkpts1, conf, inlier_mask, out_path, max_draw=400):
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    H = max(h0, h1)
    canvas = np.zeros((H, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:] = img1

    idx = np.arange(len(mkpts0))
    if len(idx) > max_draw:
        idx = np.argsort(-conf)[:max_draw]

    for i in idx:
        p0 = tuple(np.round(mkpts0[i]).astype(int))
        p1 = tuple(np.round(mkpts1[i]).astype(int) + np.array([w0, 0]))
        color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, (0, 255, 255), -1)
        cv2.circle(canvas, p1, 2, (0, 255, 255), -1)

    cv2.imwrite(str(out_path), canvas)


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
    loftr_resize_target=640,
):
    """
    query 쪽: full image 기준 (crop 없음)
      - query_hw: full image (H, W) e.g. (2160, 3840)
      - query_nonblack_bbox_xyxy: [0, 0, W, H] (전체 이미지)

    gallery 쪽: crop 기준 유지
      - gallery_crop_hw: tight crop 크기
      - gallery_nonblack_bbox_xyxy: full gallery render 이미지 기준 bbox
    """
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
        # Query: full image 기준 (crop 없음)
        "query_crop_hw": list(query_hw),                        # full image HW
        "query_nonblack_bbox_xyxy": list(query_nonblack_bbox_xyxy),  # [0,0,W,H]
        # Gallery: crop 기준
        "gallery_crop_hw": list(gallery_crop_hw),
        "gallery_nonblack_bbox_xyxy": [int(v) for v in gallery_nonblack_bbox_xyxy],
        "gallery_img_hw": list(gallery_img_hw),
        "note": (
            "Query: mkpts0 in 840px space of full query image (no crop). "
            "unmap → full query image coords directly. "
            "Gallery: mkpts1 in 840px space of gallery crop. "
            "unmap → add gallery_nonblack_bbox[:2] → full gallery render coords."
        ),
    }
    save_json(out_dir / "loftr_best_match_meta.json", meta)
    print(f"  [LoFTR] Saved {int(inlier_mask.sum())} inlier match points → {npz_path.name}")
    return npz_path

def get_mask_inlier_indices(mkpts0_full, mask_path):
    """
    원본 좌표가 마스크 내부에 있는지 확인하여 True/False 리스트 반환
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.ones(len(mkpts0_full), dtype=bool)

    h, w = mask.shape
    keep_mask = []
    for pt in mkpts0_full:
        x, y = int(round(pt[0])), int(round(pt[1]))
        # 마스크 범위 내에 있고, 물체 영역(>0)인 경우만 True
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            keep_mask.append(True)
        else:
            keep_mask.append(False)
            
    return np.array(keep_mask)

# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def draw_loftr_matches_full(img0_full, img1_full,
                            mkpts0_crop, mkpts1_crop,
                            q_bbox, g_bbox,
                            q_crop_hw, g_crop_hw,
                            inlier_mask, out_path,
                            loftr_size=840, max_draw=400):
    """
    crop 좌표계 mkpts를 full image 좌표로 역변환한 뒤 full image에 시각화.
    """
    m0_crop = unmap_from_square_resize(mkpts0_crop, q_crop_hw, loftr_size)
    m1_crop = unmap_from_square_resize(mkpts1_crop, g_crop_hw, loftr_size)

    # crop → full image
    m0_full = m0_crop + np.array([[q_bbox[0], q_bbox[1]]], dtype=np.float64)
    m1_full = m1_crop + np.array([[g_bbox[0], g_bbox[1]]], dtype=np.float64)

    h0, w0 = img0_full.shape[:2]
    h1, w1 = img1_full.shape[:2]

    # 두 이미지를 같은 높이로 리사이즈해서 나란히
    scale = min(h0, h1) / max(h0, h1)
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

    n = len(mkpts0_crop)
    idx = np.arange(n)
    if n > max_draw:
        idx = idx[:max_draw]

    for i in idx:
        p0 = (int(round(m0_full[i, 0] * sc0)), int(round(m0_full[i, 1] * sc0)))
        p1 = (int(round(m1_full[i, 0] * sc1 + W0_vis)), int(round(m1_full[i, 1] * sc1)))
        color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, (0, 255, 255), -1)
        cv2.circle(canvas, p1, 3, (0, 255, 255), -1)

    cv2.putText(canvas, f"Full image  inliers={int(inlier_mask.sum())}/{n}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 230), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def run_step5_dino_loftr_rerank(args):
    """
    DINOv2 retrieval + LoFTR rerank 통합.
    query_masked_path: query_masked_full.png (full 해상도, 배경 검정)
    """
    if args.dino_scores_json is None:
        pass

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    device = args.device if torch.cuda.is_available() else "cpu"

    # ── 1. Query 로드 ─────────────────────────────────────────────────────────
    query_masked_full_path = Path(args.out_dir) / "query_masked_full.png"
    if not query_masked_full_path.exists():
        print(f"  [WARN] query_masked_full.png not found, falling back to query_masked_path")
        query_masked_full_path = Path(args.query_masked_path)

    query_full = load_rgb(str(query_masked_full_path))
    qh, qw = query_full.shape[:2]
    print(f"  Query full image: {qw}x{qh}")

    LOFTR_SIZE = 840

    # ── step1 bbox로 query crop → LoFTR 입력 ─────────────────────────────────
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

    # ── 2. Gallery 파일 목록 ──────────────────────────────────────────────────
    gallery_dir = Path(args.gallery_dir)
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    gallery_files = sorted([p for p in gallery_dir.iterdir() if p.suffix.lower() in exts])
    if not gallery_files:
        raise FileNotFoundError(f"No gallery images in: {gallery_dir}")
    print(f"  Gallery: {len(gallery_files)} images in {gallery_dir}")

    # ── 3. DINOv2 feature (캐시 활용) ────────────────────────────────────────
    extractor = DinoV2Extractor(args.dino_model, device=device)

    # query feature (full image → nonblack crop → square_pad → DINO)
    # query 자체도 crop해서 DINO에 넣어야 similarity가 의미있음
    # (full 3840x2160 이미지를 224px로 줄이면 캔이 너무 작아짐)
    qbox = compute_nonblack_bbox(query_full, thresh=args.nonblack_thresh)
    qbox = expand_bbox(qbox, args.crop_margin, qw, qh)
    query_crop_for_dino = crop_with_bbox(query_full, qbox)
    query_dino_in = square_pad_resize(query_crop_for_dino, args.dino_input_size)
    qfeat = extractor.encode_bgr(query_dino_in)

    # gallery feature 캐시 경로: data/can_data/dino_cache/
    cache_dir = Path(args.out_dir).parent.parent / "can_data" / "dino_cache_3dgs_1920"
    # fallback: out_dir 기준
    if not (Path(args.out_dir).parent.parent / "can_data").exists():
        cache_dir = Path(args.out_dir) / "dino_cache_3dgs"
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
    scores = []
    for gp in gallery_files:
        key = gp.name
        gfeat = torch.from_numpy(gallery_feats[key]["feat"])
        score = float(torch.dot(qfeat, gfeat).item())
        scores.append({
            "file": key,
            "path": str(gp),
            "score_cosine": score,
        })

    scores_sorted = sorted(scores, key=lambda x: x["score_cosine"], reverse=True)
    topk_items = scores_sorted[:args.topk]
    print(f"  DINOv2 top-{args.topk}: {[x['file'] for x in topk_items]}")

    # dino scores json 저장 (step4 단독 실행 결과와 호환)
    dino_scores_path = out_dir / "retrieval_scores.json"
    save_json(dino_scores_path, {
        "stage": "step5",
        "sim_method": "dino",
        "query_masked_path": str(query_masked_full_path),
        "gallery_dir": str(gallery_dir),
        "dino_model": args.dino_model,
        "num_gallery": len(gallery_files),
        "topk": topk_items,
        "all_scores_sorted": scores_sorted,
    })

    # ── 5. LoFTR rerank ───────────────────────────────────────────────────────
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

        # 마스크 필터링: query crop coords → full image coords → mask 체크
        m0_crop_temp = unmap_from_square_resize(mkpts0_v, (q_loftr_h, q_loftr_w), resize_target=LOFTR_SIZE)
        m0_full_temp = m0_crop_temp + np.array([[q_loftr_bbox[0], q_loftr_bbox[1]]])
        mask_keep = get_mask_inlier_indices(m0_full_temp, query_mask_path)

        mkpts0_v = mkpts0_v[mask_keep]
        mkpts1_v = mkpts1_v[mask_keep]
        conf_v   = conf_v[mask_keep]

        H_mat, inlier_mask = estimate_inliers(mkpts0_v, mkpts1_v, ransac_thresh=args.loftr_ransac_thresh)

        total_matches = len(conf_v)
        inliers       = int(inlier_mask.sum()) if total_matches > 0 else 0
        mean_conf     = float(conf_v.mean())   if total_matches > 0 else 0.0
        inlier_ratio  = float(inliers / total_matches) if total_matches > 0 else 0.0

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

        # crop 공간 시각화
        vis_path = out_dir / f"loftr_vis_{gpath.stem}.png"
        draw_loftr_matches(query_l, gallery_l, mkpts0_v, mkpts1_v, conf_v, inlier_mask, vis_path)

        # full image 공간 시각화
        vis_full_path = out_dir / f"loftr_vis_full_{gpath.stem}.png"
        draw_loftr_matches_full(
            query_full, gimg,
            mkpts0_v, mkpts1_v,
            q_loftr_bbox, g_loftr_bbox,
            (q_loftr_h, q_loftr_w), (g_loftr_h, g_loftr_w),
            inlier_mask, vis_full_path,
            loftr_size=LOFTR_SIZE,
        )

    # ── 6. LoFTR score로 최종 정렬 ───────────────────────────────────────────
    max_inliers = max([r["inliers"] for r in results] + [1])
    for r in results:
        norm_inliers = r["inliers"] / max_inliers
        r["loftr_score"] = 0.5 * norm_inliers + 0.3 * r["mean_conf"] + 0.2 * r["inlier_ratio"]

    results = sorted(results, key=lambda x: x["loftr_score"], reverse=True)
    best = results[0]

    # ── 7. Best match 데이터 저장 ─────────────────────────────────────────────
    best_cache = match_cache[best["file"]]
    save_best_match_data(
        out_dir=out_dir,
        mkpts0_all=best_cache["mkpts0_all"],
        mkpts1_all=best_cache["mkpts1_all"],
        conf_all=best_cache["conf_all"],
        inlier_mask=best_cache["inlier_mask"],
        # Query: step1 crop 기준 (full image bbox 저장 → step6에서 offset 복원)
        query_hw=best_cache["q_loftr_hw"],
        query_nonblack_bbox_xyxy=best_cache["q_loftr_bbox"],
        # Gallery: nonblack crop 기준 (full image bbox 저장)
        gallery_crop_hw=best_cache["g_loftr_hw"],
        gallery_nonblack_bbox_xyxy=best_cache["g_loftr_bbox"],
        gallery_img_hw=best_cache["gimg_hw"],
        loftr_resize_target=LOFTR_SIZE,
    )

    # ── 8. 시각화 ─────────────────────────────────────────────────────────────
    best_render_img = best_cache["gimg"]
    best_gcrop_for_vis, _ = tight_crop_nonblack(
        best_render_img, thresh=args.nonblack_thresh, margin=args.crop_margin
    )
    query_best_vis = make_query_vs_best_image(query_crop_for_dino, best_gcrop_for_vis)
    cv2.imwrite(str(out_dir / "query_vs_best_reranked.png"), query_best_vis)

    # crop 공간 best match 시각화 복사
    best_match_vis_src = out_dir / f"loftr_vis_{Path(best['file']).stem}.png"
    best_match_vis_dst = out_dir / "loftr_matches_best.png"
    if best_match_vis_src.exists():
        best_match_vis_dst.write_bytes(best_match_vis_src.read_bytes())

    # full image 공간 best match 시각화 복사
    best_match_vis_full_src = out_dir / f"loftr_vis_full_{Path(best['file']).stem}.png"
    best_match_vis_full_dst = out_dir / "loftr_matches_best_full.png"
    if best_match_vis_full_src.exists():
        best_match_vis_full_dst.write_bytes(best_match_vis_full_src.read_bytes())

    # ── 9. 결과 저장 ──────────────────────────────────────────────────────────
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
    save_json(out_dir / "step5_rerank_summary.json", {
        "best_render": best["file"],
        "best_loftr_score": best["loftr_score"],
    })

    print("=" * 60)
    print("[Step 5] DINOv2 + LoFTR rerank complete")
    print(f"  best_render     : {best['file']}")
    print(f"  best_loftr_score: {best['loftr_score']:.4f}")
    print(f"  dino_cache      : {cache_dir}")
    print(f"  loftr_scores    : {out_dir / 'loftr_scores.json'}")
    print(f"  match_vis       : {out_dir / 'loftr_matches_best.png'}")
    print("=" * 60)