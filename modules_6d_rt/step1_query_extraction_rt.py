"""
step1_query_extraction_rt.py
============================
Real-time version of step1: YOLO + SAM2 query extraction.
Saves only the two outputs needed by downstream steps:
  - query_mask.png
  - query_masked_full.png
  - step1_result.json
No debug visualization images.
"""

import os
import cv2
import numpy as np

from modules_6d.io_utils import ensure_dir, load_image, save_json
from modules_6d.image_utils import expand_bbox, crop_from_bbox, apply_mask, crop_mask_to_bbox
from modules_6d.yolo_sam import load_yolo_model, detect_with_yolo, load_sam2_predictor, segment_from_bbox
from modules_6d.manual_bbox import select_bbox_opencv


def run_step1_query_extraction_rt(args, model_cache=None):
    ensure_dir(args.out_dir)

    image = load_image(args.query_img)
    h, w = image.shape[:2]

    # Use pre-loaded YOLO if available, otherwise load fresh
    if model_cache is not None and model_cache.yolo is not None:
        yolo_model = model_cache.yolo
    else:
        yolo_model = load_yolo_model(args.yolo_weights)

    det = detect_with_yolo(yolo_model, image, conf_thres=args.yolo_conf)

    yolo_success = det is not None
    manual_fallback_used = False

    if yolo_success:
        bbox = det["bbox_xyxy"]
        bbox_conf = det["conf"]
        detector_name = "yolo"
    else:
        if not args.use_manual_fallback:
            raise RuntimeError("YOLO failed and manual fallback is disabled.")
        bbox = select_bbox_opencv(image)
        bbox_conf = None
        manual_fallback_used = True
        detector_name = "manual_bbox"

    bbox = expand_bbox(bbox, args.bbox_margin, w, h)

    # Use pre-loaded SAM2 if available, otherwise load fresh
    if model_cache is not None and model_cache.sam2 is not None:
        predictor = model_cache.sam2
    else:
        predictor = load_sam2_predictor(
            sam2_repo=args.sam2_repo,
            checkpoint_path=args.sam2_checkpoint,
            config_path=args.sam2_config,
            device=args.device,
        )
    mask, sam_score = segment_from_bbox(predictor, image, bbox)

    # Full-resolution masked image (black background) — used by all downstream steps
    masked_full = apply_mask(image, mask)

    cv2.imwrite(os.path.join(args.out_dir, "query_mask.png"), mask)
    cv2.imwrite(os.path.join(args.out_dir, "query_masked_full.png"), masked_full)

    # Compute mask stats for step1_result.json (used by step6 coordinate compat check)
    crop_bbox_img = crop_from_bbox(image, bbox)
    crop_mask = crop_mask_to_bbox(mask, bbox)

    ys, xs = np.where(mask > 0)
    mask_area = int((mask > 0).sum())
    mask_bbox = None
    if len(xs) > 0 and len(ys) > 0:
        mask_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    result = {
        "query_img": args.query_img,
        "detector": detector_name,
        "yolo_success": yolo_success,
        "manual_fallback_used": manual_fallback_used,
        "bbox_xyxy": [int(v) for v in bbox],
        "bbox_conf": bbox_conf,
        "sam_score": sam_score,
        "mask_area": mask_area,
        "mask_bbox_xyxy": mask_bbox,
        "crop_size_hw": [int(crop_bbox_img.shape[0]), int(crop_bbox_img.shape[1])],
        "crop_mask_size_hw": [int(crop_mask.shape[0]), int(crop_mask.shape[1])],
        "saved_files": {
            "query_mask": os.path.join(args.out_dir, "query_mask.png"),
            "query_masked_full": os.path.join(args.out_dir, "query_masked_full.png"),
        },
    }
    save_json(os.path.join(args.out_dir, "step1_result.json"), result)
    print("[OK] Step 1 RT finished:", args.out_dir)
