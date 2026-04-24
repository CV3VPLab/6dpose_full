import os
import sys
import cv2
import numpy as np


def load_yolo_model(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)


def detect_with_yolo(model, image_bgr, conf_thres=0.25):
    results = model.predict(source=image_bgr, conf=conf_thres, verbose=False)
    if len(results) == 0:
        return None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_i = None
    best_conf = -1.0
    for i in range(len(boxes)):
        conf = float(boxes.conf[i].item())
        if conf > best_conf:
            best_conf = conf
            best_i = i

    xyxy = boxes.xyxy[best_i].detach().cpu().numpy().tolist()
    xyxy = [int(round(v)) for v in xyxy]
    cls_id = int(boxes.cls[best_i].item()) if boxes.cls is not None else -1
    return {
        "bbox_xyxy": xyxy,
        "conf": best_conf,
        "cls_id": cls_id,
    }


def load_sam2_predictor(sam2_repo, checkpoint_path, config_path, device="cuda"):
    if sam2_repo not in sys.path:
        sys.path.insert(0, sam2_repo)

    # Common SAM2 import paths; adjust if repo differs.
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as e:
        raise ImportError(
            "Failed to import SAM2. Check sam2_repo path and repo structure. "
            f"Original error: {e}"
        )

    model = build_sam2(config_path, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor


def segment_from_bbox(predictor, image_bgr, bbox_xyxy):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    box = np.array(bbox_xyxy, dtype=np.float32)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=True,
    )
    if masks is None or len(masks) == 0:
        raise RuntimeError("SAM2 failed to produce a mask.")

    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].astype(np.uint8) * 255
    return mask, float(scores[best_idx])
