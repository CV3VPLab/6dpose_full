import cv2


def select_bbox_opencv(image, window_name="Draw BBox and press ENTER"):
    disp = image.copy()
    roi = cv2.selectROI(window_name, disp, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise RuntimeError("No valid manual bbox selected.")
    return [int(x), int(y), int(x + w), int(y + h)]
