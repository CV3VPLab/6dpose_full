import os
import sys
import argparse
import cv2
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.io_utils import ensure_dir, load_image, save_json
from utils.image_utils import expand_bbox, crop_with_bbox, compute_bbox, apply_mask
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

    from pathlib import Path
    outPath = Path(args.out_dir)
    queryPath = Path(args.query_img)
    maskPath = outPath / f"q_mask_{queryPath.stem}.png"
    
    cv2.imwrite(str(maskPath), mask)
    
    bbox_vis = draw_bbox(image, bbox, label=detector_name)
    mask_overlay = make_mask_overlay(image, mask)
    query_crop = crop_with_bbox(image, bbox)
    masked_query = apply_mask(image, mask)
    
    mask_bbox = compute_bbox(mask)    

    bboxPath = outPath / f"q_bbox_{queryPath.stem}.npy"
    np.save(bboxPath, mask_bbox)

    print("[OK] Step 1 finished:", args.out_dir)
    print(f"query bounding box : {mask_bbox}")


def parse_args():
    p = argparse.ArgumentParser(description="Render gallery images from custom poses for GS model")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--query_img", type=str, default=None)
    p.add_argument("--yolo_weights", type=str, default=None)
    p.add_argument("--sam2_checkpoint", type=str, default=None)
    p.add_argument("--sam2_repo", type=str, default=None)
    p.add_argument("--sam2_config", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--yolo_conf", type=float, default=0.25)
    p.add_argument("--bbox_margin", type=int, default=10)
    p.add_argument("--use_manual_fallback", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_step1_query_extraction(args)