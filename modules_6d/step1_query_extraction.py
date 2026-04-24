import os
import cv2
import numpy as np

from .io_utils import ensure_dir, load_image, save_json
from .image_utils import expand_bbox, crop_from_bbox, apply_mask, crop_mask_to_bbox
from .manual_bbox import select_bbox_opencv
from .viz_utils import draw_bbox, make_mask_overlay
from .yolo_sam import load_yolo_model, detect_with_yolo, load_sam2_predictor, segment_from_bbox


def run_step1_query_extraction(args):
    ensure_dir(args.out_dir)

    image = load_image(args.query_img)
    h, w = image.shape[:2]

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

    predictor = load_sam2_predictor(
        sam2_repo=args.sam2_repo,
        checkpoint_path=args.sam2_checkpoint,
        config_path=args.sam2_config,
        device=args.device,
    )
    mask, sam_score = segment_from_bbox(predictor, image, bbox)

    bbox_vis = draw_bbox(image, bbox, label=detector_name)
    mask_overlay = make_mask_overlay(image, mask)

    crop_bbox = crop_from_bbox(image, bbox)
    masked_image = apply_mask(image, mask)
    crop_mask = crop_mask_to_bbox(mask, bbox)
    crop_masked = crop_from_bbox(masked_image, bbox)

    # full size masked image (배경 검정, 3840x2160 그대로)
    # 이후 모든 단계에서 crop 없이 full image 좌표계로 일관되게 사용
    masked_full = apply_mask(image, mask)  # 배경 검정, full 해상도

    # cv2.imwrite(os.path.join(args.out_dir, "query_with_bbox.png"), bbox_vis)
    cv2.imwrite(os.path.join(args.out_dir, "query_mask.png"), mask)
    # cv2.imwrite(os.path.join(args.out_dir, "query_mask_overlay.png"), mask_overlay)
    # cv2.imwrite(os.path.join(args.out_dir, "query_crop_bbox.png"), crop_bbox)
    # cv2.imwrite(os.path.join(args.out_dir, "query_crop_masked.png"), crop_masked)
    cv2.imwrite(os.path.join(args.out_dir, "query_masked_full.png"), masked_full)

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
        "crop_size_hw": [int(crop_bbox.shape[0]), int(crop_bbox.shape[1])],
        "crop_mask_size_hw": [int(crop_mask.shape[0]), int(crop_mask.shape[1])],
        "saved_files": {
            "query_with_bbox": os.path.join(args.out_dir, "query_with_bbox.png"),
            "query_mask": os.path.join(args.out_dir, "query_mask.png"),
            "query_mask_overlay": os.path.join(args.out_dir, "query_mask_overlay.png"),
            "query_crop_bbox": os.path.join(args.out_dir, "query_crop_bbox.png"),
            "query_crop_masked": os.path.join(args.out_dir, "query_crop_masked.png"),
            "query_masked_full": os.path.join(args.out_dir, "query_masked_full.png"),
        }
    }
    save_json(os.path.join(args.out_dir, "step1_result.json"), result)
    print("[OK] Step 1 finished:", args.out_dir)