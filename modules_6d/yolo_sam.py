import os
import sys
import cv2
import numpy as np


def load_yolo_model(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)


def detect_with_yolo(model, image_bgr, conf_thres=0.25, imgsz=1280):
    results = model.predict(source=image_bgr, conf=conf_thres, imgsz=imgsz, verbose=False)
    if len(results) == 0:
        return None
    
    if isinstance(image_bgr, np.ndarray):
        nImages = 1
    else:
        nImages = len(image_bgr)

    assert nImages == len(results), f"Number of images ({nImages}) does not match number of results ({len(results)})"

    resList = []
    for i in range(nImages):
        boxes = results[i].boxes
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
        resList.append( 
            { "bbox_xyxy": xyxy,
              "conf": best_conf,
              "cls_id": cls_id } )
        
    return resList


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


def segment_from_bbox(predictor, image_rgb, bbox_xyxy):
    box = np.array(bbox_xyxy, dtype=np.float32)

    if isinstance(image_rgb, list):
        assert len(image_rgb) == len(bbox_xyxy)
        nImages = len(image_rgb)
        predictor.set_image_batch(image_rgb)
        masks_batch, scores_batch, logits_batch = predictor.predict_batch(
            box_batch=box,
            multimask_output=True 
        )
        if masks_batch is None or len(masks_batch) == 0:
            raise RuntimeError("SAM2 failed to produce a mask.")
        best_idx = np.argmax(scores_batch, axis=1)
        masks = [masks_batch[i][best_idx[i]].astype(np.uint8) * 255 for i in range(nImages)]
        scores = [float(scores_batch[i][best_idx[i]]) for i in range(nImages)]

        return masks, scores
    
    else:
        assert isinstance(image_rgb, np.ndarray)
        predictor.set_image(image_rgb)

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
    
    
