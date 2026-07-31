import locale
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
try:
    locale.setlocale(locale.LC_CTYPE, "C.UTF-8")
except locale.Error:
    pass

try:
    import torch.utils._cpp_extension_versioner as _torch_ext_versioner

    def _hash_source_files_utf8(hash_value, source_files):
        for filename in source_files:
            with open(filename, encoding="utf-8", errors="ignore") as file:
                hash_value = _torch_ext_versioner.update_hash(hash_value, file.read())
        return hash_value

    _torch_ext_versioner.hash_source_files = _hash_source_files_utf8
except Exception:
    pass

from pathlib import Path
from modules_6d.yolo_sam import load_yolo_model, detect_with_yolo, load_sam2_predictor, segment_from_bbox
from modules_6d.retrieval_dino import DinoV2Extractor, DinoV2ExtractorTRT, load_dino_trt_session
from modules_6d.retrieval_dino_loftr import compute_loftr_matches
from modules_6d.retrieval_edm import (
    compute_edm_matches,
    compute_edm_trt_matches,
    load_edm_model,
    load_edm_trt_session,
    warmup_edm_trt_session,
)
from modules_6d.step6_translation import get_initial_pose, get_gallery_pose, solve_pose_pnp
from refine_pose import (
    GaussianRenderer, CosineWarmupScheduler, 
    dssim_loss, dms_ssim_loss, 
    so3_exp_map, crop_chw_with_bbox
)
from tqdm import tqdm
import time

from utils.io_utils import (
    load_json, save_json, 
    load_intrinsics, params_to_K,
    resolve_ply_path 
)
from utils.image_utils import (
    load_rgb, render_to_image,
    expand_bbox, compute_bbox, get_bbox_size, square_bbox, crop_with_bbox, square_pad_resize, unmap_to_full_image,     
    make_gallery_square, construct_galleryInfo, 
    get_specular_mask, apply_mask, get_mask_inlier_indices,
    erode_binary_tensor,    
    scale_image_draw_maskcontour, draw_contour, imshow_tensor,
    dice_loss, gradient_matching_loss
)
from utils.image_utils import tensor_to_np as t2np, np_to_tensor as np2t

from gaussian_renderer import GaussianModel
from utils.geom_utils import (
    depth_tensor_to_xyz_map, depth_tensor_to_xyz_map2, depth_sample_to_xyz, 
    is_valid_point3d,
    T2Rt, mesh_from_depth_grid
)


FRUSTUM_N = 0.03  # near frustum : 3cm

def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

class ECCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(ECCLoss, self).__init__()
        self.eps = eps # 0으로 나누어지는 것을 방지하기 위한 작은 값

    def forward(self, img1, img2):
        """
        img1, img2: (B, C, H, W) 형태의 텐서
        """
        # 1. 각 이미지의 공간 차원(H, W)에 대한 평균을 구하고 빼줍니다 (Zero-mean)
        img1_mean = img1 - torch.mean(img1, dim=[-2, -1], keepdim=True)
        img2_mean = img2 - torch.mean(img2, dim=[-2, -1], keepdim=True)

        # 2. 분자: 두 이미지의 공분산 (Covariance)
        cov = torch.mean(img1_mean * img2_mean, dim=[-2, -1], keepdim=True)

        # 3. 분모: 각 이미지의 표준편차 (Standard Deviation)
        std1 = torch.std(img1_mean, dim=[-2, -1], keepdim=True)
        std2 = torch.std(img2_mean, dim=[-2, -1], keepdim=True)

        # 4. 정규화된 교차 상관계수 (ECC)
        ecc = cov / (std1 * std2 + self.eps)

        # 5. Loss 구성 (1 - ECC) 및 배치 전체 평균 반환
        loss = 1.0 - ecc
        return loss.mean()
    

def init_gaussians(config, ply_path, scale=1.0):
    options = config["options"]
    gaussians = GaussianModel(options["sh_degree"])
    gaussians.load_ply(ply_path, scale, use_train_test_exp=False)
    gaussians.freeze_except_pose()

    return gaussians


def load_xyz_map(xyz_dir, idx):
    xyz_map_path = xyz_dir / f"{idx:04d}.npy"
    return np.load(str(xyz_map_path)).astype(np.float64)


def load_detector(config):
    assert config["name"] == "yolo"
    return load_yolo_model(config["weights"]), config["options"]["conf_thr"]


def load_segmentator(config):
    assert config["name"] == "sam2"

    weights = config["weights"]
    options = config["options"]
    return load_sam2_predictor(
        sam2_repo       = options["repo"],
        checkpoint_path = weights,
        config_path     = options["config"],
        device          = 'cuda'
    )


def load_extractor(config):
    assert config["name"] == "DINOv2"

    options = config['options']
    if options.get("onnx"):
        sess = load_dino_trt_session(
            options["onnx"],
            trt_cache_dir=options.get("trt_cache_dir"),
            require_gpu=options.get("require_gpu", True),
        )
        extractor = DinoV2ExtractorTRT(options["model"], sess)
        if options.get("warmup", True):
            t0 = sync_time()
            dummy_rgb = np.zeros((int(options.get("input_size", 224)), int(options.get("input_size", 224)), 3), dtype=np.uint8)
            for _ in range(int(options.get("warmup_runs", 1))):
                extractor.encode_rgb(dummy_rgb)
            t1 = sync_time()
            print(f"  [DINO TRT] Warmup done ({(t1 - t0):.3f}s, runs={options.get('warmup_runs', 1)})")
        return extractor, options

    return DinoV2Extractor(options["model"], device='cuda'), options


def load_matcher(config):
    name = config["name"]
    options = config['options']

    if name == "LoFTR":
        import kornia.feature as KF
        model = KF.LoFTR(pretrained=options["pretrained"]).to('cuda').eval()
        return {"name": name, "model": model, "options": options}

    if name == "EDM":
        model = load_edm_model(
            edm_repo=config["repo"],
            ckpt_path=config["weights"],
            device="cuda",
        )
        return {"name": name, "model": model, "options": options}

    if name == "EDM_TRT":
        model = load_edm_trt_session(
            onnx_path=config["onnx"],
            trt_cache_dir=config.get("trt_cache_dir"),
            require_gpu=options.get("require_gpu", True),
        )
        if options.get("warmup", True):
            input_size = int(options["input_size"])
            n_runs = int(options.get("warmup_runs", 1))
            t0 = sync_time()
            warmup_edm_trt_session(model, input_size, input_size, n_runs=n_runs)
            t1 = sync_time()
            print(f"  [EDM TRT] Warmup done ({(t1 - t0):.3f}s, runs={n_runs})")
        return {"name": name, "model": model, "options": options}

    raise ValueError(f"Unsupported matcher: {name}")


def load_networks(config):
    detector    = load_detector(config["detector"])
    segmentator = load_segmentator(config["segmentator"])
    extractor   = load_extractor(config["feat_extractor"])
    matcher     = load_matcher(config["matcher"])

    assert detector    is not None  
    assert segmentator is not None  
    assert extractor   is not None  
    assert matcher     is not None  
    return detector, segmentator, extractor, matcher


def get_query_paths(config):
    config_input = config["input"]
    assert config_input["type"] == "file" or config_input["type"] == "dir"

    obj_num = config["object"]
    obj_name = config["objects"][obj_num][0]
    file_path = Path("data/object")
    file_path = file_path / obj_name / "query"

    if config_input["type"] == "file":
        return [file_path / config_input["name"]]
    elif config_input["type"] == "dir":
        qpaths = [f for f in file_path.iterdir() if f.is_file()]
        qpaths.sort()
        return qpaths
    else:
        assert False    
    
    
def get_query(config):
    config_input = config["input"]
    assert config_input["type"] == "file" or config_input["type"] == "dir"

    obj_num = config["object"]
    obj_name = config["objects"][obj_num][0]
    file_path = Path("data/object")
    file_path = file_path / obj_name / "query"
    
    if config_input["type"] == "file":
        image = load_rgb(file_path / config_input["name"])
        assert image is not None
        return image
    elif config_input["type"] == "dir":
        images = []
        for f in file_path.iterdir():
            if f.is_file():
                images.append( load_rgb(f) )
        return images


def get_obj_path(obj_name, kind):
    # kind : {"output", "object", "gallery", "xyz", "model"}
    if kind == "output":
        return Path("data/output") / obj_name
    
    out_path = Path("data/object") / obj_name
    if kind == "object":
        return out_path 
    
    return out_path / kind
    

def get_K_path(config):
    return Path("data/camera") / config["cam_type"] / config["K_filename"]


def project_obj_axes(K, R, t, axis_len_m=0.1):
    obj_pts = np.array([
        [0, 0, 0],
        [axis_len_m, 0, 0],
        [0, axis_len_m, 0],
        [0, 0, axis_len_m],
        [-axis_len_m, 0, 0],
        [0, -axis_len_m, 0],
        [0, 0, -axis_len_m],
    ], dtype=np.float32)

    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec = t.reshape(3, 1).astype(np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    imgpts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K.astype(np.float64), dist)
    imgpts = np.round(imgpts.reshape(-1, 2)).astype(int)

    return imgpts


def make_comp_image(q_crop, q_mask, q_bbox, gaussian_proxy:GaussianRenderer, R0, t0):
    q_crop_annotation = q_crop.copy()
    
    _r, _a, _ = gaussian_proxy.render_no_grad()
    r_crop = crop_with_bbox(render_to_image(_r[0]), q_bbox)
        
    R, t = gaussian_proxy.get_T()
    R_np = t2np(R)
    t_np = t2np(t)
    K = gaussian_proxy.K_mat.cpu().numpy()[0]
    imgpts = project_obj_axes(K, R_np, t_np, 0.06)
    imgpts = imgpts - q_bbox[:2]
    o = tuple(imgpts[0])
    cv2.line(q_crop_annotation, o, tuple(imgpts[1]), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.line(q_crop_annotation, o, tuple(imgpts[2]), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(q_crop_annotation, o, tuple(imgpts[3]), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(q_crop_annotation, o, tuple(imgpts[4]), (255, 0, 0), 1, cv2.LINE_AA)
    cv2.line(q_crop_annotation, o, tuple(imgpts[5]), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(q_crop_annotation, o, tuple(imgpts[6]), (0, 0, 255), 1, cv2.LINE_AA)
    cv2.circle(q_crop_annotation, o, 6, (255, 255, 255), -1, cv2.LINE_AA)
    
    gaussian_proxy.set_T(R0, t0)
    _, _a0, _ = gaussian_proxy.render_no_grad()
    draw_contour(q_crop_annotation, crop_with_bbox(t2np(_a0[0].squeeze(2)), q_bbox) > 0.5, (255, 0, 0))    
    draw_contour(q_crop_annotation, crop_with_bbox(t2np(_a[0].squeeze(2)), q_bbox) > 0.5)
    
    q_w = q_crop.shape[1]
    alpha = 0.6
    overlay_crop = cv2.addWeighted(q_crop, alpha, r_crop, 1.0 - alpha, 0)
    res_img = np.zeros((q_crop.shape[0], q_crop.shape[1] * 4, 3), dtype=np.uint8)
    draw_contour(q_crop, q_mask)
    res_img[:, :q_w] = q_crop
    res_img[:, q_w:q_w*2] = r_crop
    res_img[:, q_w*2:q_w*3] = overlay_crop
    res_img[:, q_w*3:] = q_crop_annotation
    cv2.putText(res_img, "Query",   (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(res_img, "Render",  (q_w + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(res_img, "Overlay", (q_w*2 + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(res_img, "Query+Render contours",   (q_w*3 + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return res_img


def make_pose_validation_image(query, gaussian_proxy:GaussianRenderer, q_bbox, dra = 5.0, dt = 0.005):
    # 5 dgree, 5 mm perturbation
    q_crop = crop_with_bbox(query, q_bbox)
    q_crop_t = q_crop.copy()

    R, t = gaussian_proxy.get_T()
    
    _, _a, _ = gaussian_proxy.render_no_grad()    
    mask_crop = crop_with_bbox(t2np(_a[0].squeeze(2)), q_bbox) > 0.5
    draw_contour(q_crop, mask_crop)
    draw_contour(q_crop_t, mask_crop)

    colors = [(217, 83, 25), (0, 114, 189), (77, 190, 238), (237, 177, 32), (119, 172, 38), (126, 47, 142)]
    dr = np.deg2rad(dra)    # 5 degree
    cos_r = np.cos(dr)
    sin_r = np.sin(dr)
    str_angle = f"{dra:.1f} deg"
    str_t = f"{dt*1000} mm"

    Rx = torch.tensor([[1, 0, 0],[0, cos_r, -sin_r], [0, sin_r, cos_r]], device=R.device, dtype=torch.float32)
    gaussian_proxy.set_T(Rx @ R, t)
    _, _a, _ = gaussian_proxy.render_no_grad()
    mask_crop = crop_with_bbox(t2np(_a[0].squeeze(2)), q_bbox) > 0.5
    contours, _ = cv2.findContours( mask_crop.astype(np.uint8) * 255, 
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cv2.drawContours(q_crop, contours, -1, colors[0], 1)
    cv2.putText(q_crop, f"Rx {str_angle}", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0], 1)
    
    Ry = torch.tensor([[cos_r, 0, sin_r],[0, 1, 0], [-sin_r, 0, cos_r]], device=R.device, dtype=torch.float32)
    gaussian_proxy.set_T(Ry @ R, t)
    _, _a, _ = gaussian_proxy.render_no_grad()
    mask_crop = crop_with_bbox(t2np(_a[0].squeeze(2)), q_bbox) > 0.5
    contours, _ = cv2.findContours( mask_crop.astype(np.uint8) * 255, 
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cv2.drawContours(q_crop, contours, -1, colors[1], 1)
    cv2.putText(q_crop, f"Ry {str_angle}", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[1], 1)

    Rz = torch.tensor([[cos_r, -sin_r, 0],[sin_r, cos_r, 0], [0, 0, 1]], device=R.device, dtype=torch.float32)
    gaussian_proxy.set_T(Rz @ R, t)
    _, _a, _ = gaussian_proxy.render_no_grad()
    mask_crop = crop_with_bbox(t2np(_a[0].squeeze(2)), q_bbox) > 0.5
    contours, _ = cv2.findContours( mask_crop.astype(np.uint8) * 255, 
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cv2.drawContours(q_crop, contours, -1, colors[2], 1)
    cv2.putText(q_crop, f"Rz {str_angle}", (12, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[2], 1)

    t[0] += dt
    gaussian_proxy.set_T(R, t)
    _, _a, _ = gaussian_proxy.render_no_grad()
    mask_crop = crop_with_bbox(t2np(_a[0].squeeze(2)), q_bbox) > 0.5
    contours, _ = cv2.findContours( mask_crop.astype(np.uint8) * 255, 
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cv2.drawContours(q_crop_t, contours, -1, colors[3], 1)
    cv2.putText(q_crop_t, f"tx {str_t}", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[3], 1)

    t[0] -= dt
    t[1] += dt
    gaussian_proxy.set_T(R, t)
    _, _a, _ = gaussian_proxy.render_no_grad()
    mask_crop = crop_with_bbox(t2np(_a[0].squeeze(2)), q_bbox) > 0.5
    contours, _ = cv2.findContours( mask_crop.astype(np.uint8) * 255, 
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cv2.drawContours(q_crop_t, contours, -1, colors[4], 1)
    cv2.putText(q_crop_t, f"ty {str_t}", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[4], 1)

    t[1] -= dt
    t[2] += dt
    gaussian_proxy.set_T(R, t)
    _r, _a, _ = gaussian_proxy.render_no_grad()
    r_crop = crop_with_bbox(render_to_image(_r[0]), q_bbox)
    # gray = cv2.cvtColor(r_crop, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours( mask_crop.astype(np.uint8) * 255, 
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cv2.drawContours(q_crop_t, contours, -1, colors[5], 1)
    cv2.putText(q_crop_t, f"tz {str_t}", (12, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[5], 1)

    return np.hstack((q_crop, q_crop_t))


# processes
def detect_segment(query, nets):
    detector, det_conf_thr = nets[0]
    segmentator = nets[1]
    
    h, w = query.shape[:2]
    det = detect_with_yolo(detector, query, conf_thres=det_conf_thr)
    assert det is not None 
    det_conf = det["conf"]
    
    bbox = expand_bbox(det["bbox_xyxy"], 12, w, h)
    mask, seg_score = segment_from_bbox(segmentator, query, bbox)
    
    print(f"detection score: {det_conf:.3f},  segmentation score: {seg_score:.3f}, bounding box: {bbox}") 
    return mask


def retrieve_best(queryInfo, galleryInfo, extractor):
    # KSCHOI TODO: occlusion 상황에서 topk가 의미 있는지 확인 (현재는 DINO feature similarity top-1만 사용함)
    g_feats = galleryInfo["feats"]
    g_bbox_size = galleryInfo["bbox_size"]

    q_bbox = queryInfo["bbox"]
    # query bounding box can be larger than gallery bbox, so we use the max of both for square cropping
    q_bbox_size = max( g_bbox_size, q_bbox[2] - q_bbox[0], q_bbox[3] - q_bbox[1] )
    q_bbox_ext = square_bbox(q_bbox, q_bbox_size)
    masked_query_crop = crop_with_bbox(queryInfo["masked_query"], q_bbox_ext)    

    # masked_query_crop = cv2.detailEnhance(masked_query_crop, sigma_s=10, sigma_r=0.15)     

    # A cropped query image for extracting DINOv2 feature
    ext_net = extractor[0]
    ext_opts = extractor[1]
    dino_size = ext_opts["input_size"]
    query_dino_in = square_pad_resize(masked_query_crop, dino_size * 2)
    if g_feats.shape[1] == 384:
        qfeat = ext_net.encode_rgb(query_dino_in)    
    elif g_feats.shape[1] == 384 * 4:
        qfeat = ext_net.encode_4rgb(query_dino_in)
        
    scores = (g_feats @ qfeat).numpy()
    best_item = np.argmax(scores)
    
    return best_item, scores[best_item], masked_query_crop, q_bbox_ext


def retrieve_topk(queryInfo, galleryInfo, extractor, k=3):
    # KSCHOI TODO: occlusion 상황에서 topk가 의미 있는지 확인 (현재는 DINO feature similarity top-1만 사용함)
    g_feats = galleryInfo["feats"]
    g_bbox_size = galleryInfo["bbox_size"]

    q_bbox = queryInfo["bbox"]
    # query bounding box can be larger than gallery bbox, so we use the max of both for square cropping
    q_bbox_size = max( g_bbox_size, q_bbox[2] - q_bbox[0], q_bbox[3] - q_bbox[1] )
    q_bbox_ext = square_bbox(q_bbox, q_bbox_size)
    masked_query_crop = crop_with_bbox(queryInfo["masked_query"], q_bbox_ext)    

    # A cropped query image for extracting DINOv2 feature
    ext_net = extractor[0]
    ext_opts = extractor[1]
    dino_size = ext_opts["input_size"]
    query_dino_in = square_pad_resize(masked_query_crop, dino_size * 2)
    if g_feats.shape[1] == 384:
        qfeat = ext_net.encode_rgb(query_dino_in)    
    elif g_feats.shape[1] == 384 * 4:
        qfeat = ext_net.encode_4rgb(query_dino_in)
        
    scores = (g_feats @ qfeat).numpy()    
    topk_items = np.argsort(scores)[::-1][:k]    
    
    return topk_items, scores[topk_items], masked_query_crop, q_bbox_ext


def compute_matches(query, gallery, matcher):
    match_name = matcher["name"]
    match_net = matcher["model"]
    match_opts = matcher["options"]
    in_size = match_opts["input_size"]
    conf_thr = match_opts["conf_thr"]
    
    query_m = square_pad_resize(query, in_size)
    gallery_m = square_pad_resize(gallery, in_size)
    
    if match_name == "LoFTR":
        mkpts0, mkpts1, conf = compute_loftr_matches(match_net, query_m, gallery_m, device="cuda")
    elif match_name == "EDM":
        mkpts0, mkpts1, conf = compute_edm_matches(match_net, query_m, gallery_m, device="cuda")
    elif match_name == "EDM_TRT":
        mkpts0, mkpts1, conf = compute_edm_trt_matches(
            match_net,
            query_m,
            gallery_m,
            conf_thr=0.0,
        )
    else:
        raise ValueError(f"Unsupported matcher: {match_name}")

    n_raw = len(conf)
    valid = conf >= conf_thr
    mkpts0, mkpts1, conf = mkpts0[valid], mkpts1[valid], conf[valid]
    print(f"  [{match_name}] matches after conf>={conf_thr}: {len(conf)} / {n_raw}")

    mkpts0 = unmap_square_pad_resize(mkpts0, query.shape[:2], query_m.shape[:2])
    mkpts1 = unmap_square_pad_resize(mkpts1, gallery.shape[:2], gallery_m.shape[:2])

    # from utils.image_utils import draw_matches
    # match_img = draw_matches(query_l, gallery_l, mkpts0, mkpts1, conf, None)

    return mkpts0, mkpts1, conf


def unmap_square_pad_resize(pts, org_hw, resized_hw):
    # pts: (N, 2) in resized_hw coordinates
    # org_hw: (H, W) original image dimensions
    # resized_hw: (H', W') resized image dimensions
    h, w = org_hw
    side = max(h, w)
    y0 = (side - h) // 2
    x0 = (side - w) // 2

    h_ratio = side / resized_hw[0]
    w_ratio = side / resized_hw[1]

    pts_unmapped = pts.copy()
    pts_unmapped[:, 0] *= w_ratio
    pts_unmapped[:, 1] *= h_ratio
    pts_unmapped[:, 0] -= x0
    pts_unmapped[:, 1] -= y0

    return pts_unmapped


def unmap_inlier_matches(matching_results, bboxes, mask = None):
    # 마스크 필터링: query crop coords → full image coords → mask 체크
    pts0_full = unmap_to_full_image(matching_results[0], bboxes[0])
    pts1_full = unmap_to_full_image(matching_results[1], bboxes[1])
    conf = matching_results[2]
    if mask is not None:
        mask_keep = get_mask_inlier_indices(pts0_full, mask)
        pts0_full = pts0_full[mask_keep]
        pts1_full = pts1_full[mask_keep]
        conf = conf[mask_keep]

    return pts0_full, pts1_full, conf


from utils.image_utils import draw_crop_matches, draw_matches
def get_T0(query_info, gallery_info, nets, K, reproj_thr):
    best_idx, score, query_crop, q_bbox_ext = retrieve_best(query_info, gallery_info, nets[2])
    # prepare the best reference for matcher 
    gallery_crop, g_bbox_ext = make_gallery_square(gallery_info, best_idx, query_crop.shape[0])

    matching_results = compute_matches( query_crop, gallery_crop, nets[3] )
    # matcher input size in the matcher options
    pts0, pts1, conf = unmap_inlier_matches( matching_results, (q_bbox_ext, g_bbox_ext), query_info["mask"])
    # match_img = draw_crop_matches(gallery_crop, query_crop, g_bbox_ext, q_bbox_ext, pts0, pts1, conf, None)

    # Get initial pose using the matched 2D-3D correspondences and PnP
    R_g, t_g = get_gallery_pose(gallery_info["poses"], best_idx)
    g_bbox = gallery_info["bboxes"][best_idx]

    xyz_dir = gallery_info["path"].parent / "xyz"
    xyz_map = load_xyz_map(xyz_dir, best_idx)
    pts2d_xyz = pts1 - [g_bbox[:2]]
    
    R0, t0, pts3d, reproj_err, inlier_idx, match_counts = get_initial_pose(
        xyz_map, pts2d_xyz, pts0, conf, 
        K, R_g, reproj_thr
    )

    n_valid = match_counts[-1][1]
    assert n_valid == len(inlier_idx), "Length mismatch after get_initial_pose"
    
    # reproj_stats_after_inliers = compute_mean_reprojection_error(
    #     pts0[inlier_idx], pts3d[inlier_idx], K, R0, t0
    # )
    return R0, t0, n_valid


def get_T0_stereo(query_infos, gallery_info, nets, K, reproj_thr):
    best_idx_l, score_l, query_crop_l, q_bbox_ext_l = retrieve_best(query_infos[0], gallery_info, nets[2])
    best_idx_r, score_r, query_crop_r, q_bbox_ext_r = retrieve_best(query_infos[1], gallery_info, nets[2])
    # prepare the best reference for matcher 
    gallery_crop_l, g_bbox_ext_l = make_gallery_square(gallery_info, best_idx_l, query_crop_l.shape[0])
    gallery_crop_r, g_bbox_ext_r = make_gallery_square(gallery_info, best_idx_r, query_crop_r.shape[0])

    matching_results_l = compute_matches( query_crop_l, gallery_crop_l, nets[3] )
    matching_results_r = compute_matches( query_crop_r, gallery_crop_r, nets[3] )

    if len(matching_results_l[0]) > len(matching_results_r[0]):
        matching_results = matching_results_l
        q_bbox_ext = q_bbox_ext_l
        g_bbox_ext = g_bbox_ext_l
        query_info = query_infos[0]
        best_idx = best_idx_l
        which_query = 0
    else:
        matching_results = matching_results_r
        q_bbox_ext = q_bbox_ext_r
        g_bbox_ext = g_bbox_ext_r
        query_info = query_infos[1]
        best_idx = best_idx_r
        which_query = 1

    # matcher input size in the matcher options
    pts0, pts1, conf = unmap_inlier_matches( matching_results, (q_bbox_ext, g_bbox_ext), query_info["mask"])
    # match_img = draw_crop_matches(gallery_crop, query_crop, g_bbox_ext, q_bbox_ext, pts0, pts1, conf, None)

    # Get initial pose using the matched 2D-3D correspondences and PnP
    R_g, t_g = get_gallery_pose(gallery_info["poses"], best_idx)
    g_bbox = gallery_info["bboxes"][best_idx]

    xyz_dir = gallery_info["path"].parent / "xyz"
    xyz_map = load_xyz_map(xyz_dir, best_idx)
    pts2d_xyz = pts1 - [g_bbox[:2]]
    
    R0, t0, pts3d, reproj_err, inlier_idx, match_counts = get_initial_pose(
        xyz_map, pts2d_xyz, pts0, conf, 
        K, R_g, reproj_thr
    )

    n_valid = match_counts[-1][1]
    assert n_valid == len(inlier_idx), "Length mismatch after get_initial_pose"
    
    # reproj_stats_after_inliers = compute_mean_reprojection_error(
    #     pts0[inlier_idx], pts3d[inlier_idx], K, R0, t0
    # )
    return R0, t0, which_query


def get_T0_stereo1(query_infos, gallery_info, nets, K, reproj_thr):
    best_inds_l, score_l, query_crop_l, q_bbox_ext_l = retrieve_topk(query_infos[0], gallery_info, nets[2])
    best_inds_r, score_r, query_crop_r, q_bbox_ext_r = retrieve_topk(query_infos[1], gallery_info, nets[2])
    
    # prepare the references for matcher
    nMatches = 0 
    for i in range(len(best_inds_l)):
        gallery_crop_l, g_bbox_ext_l = make_gallery_square(gallery_info, best_inds_l[i], query_crop_l.shape[0])
        matching_results_l = compute_matches( query_crop_l, gallery_crop_l, nets[3] )
        if nMatches < len(matching_results_l[0]):
            nMatches = len(matching_results_l[0])
            matching_results = matching_results_l
            gallery_crop, g_bbox_ext = gallery_crop_l, g_bbox_ext_l
            best_idx = best_inds_l[i]
            which_query = 0
    
    for i in range(len(best_inds_r)):
        gallery_crop_r, g_bbox_ext_r = make_gallery_square(gallery_info, best_inds_r[i], query_crop_r.shape[0])
        matching_results_r = compute_matches( query_crop_r, gallery_crop_r, nets[3] )
        if nMatches < len(matching_results_r[0]):
            nMatches = len(matching_results_r[0])
            matching_results = matching_results_r
            gallery_crop, g_bbox_ext = gallery_crop_r, g_bbox_ext_r
            best_idx = best_inds_r[i]
            which_query = 1

    if which_query == 0:
        q_bbox_ext = q_bbox_ext_l        
    else:
        q_bbox_ext = q_bbox_ext_r        

    # matcher input size in the matcher options
    pts0, pts1, conf = unmap_inlier_matches( matching_results, (q_bbox_ext, g_bbox_ext), query_infos[which_query]["mask"])
    # match_img = draw_crop_matches(gallery_crop, query_crop, g_bbox_ext, q_bbox_ext, pts0, pts1, conf, None)

    # Get initial pose using the matched 2D-3D correspondences and PnP
    R_g, t_g = get_gallery_pose(gallery_info["poses"], best_idx)
    g_bbox = gallery_info["bboxes"][best_idx]

    xyz_dir = gallery_info["path"].parent / "xyz"
    xyz_map = load_xyz_map(xyz_dir, best_idx)
    pts2d_xyz = pts1 - [g_bbox[:2]]
    
    R0, t0, pts3d, reproj_err, inlier_idx, match_counts = get_initial_pose(
        xyz_map, pts2d_xyz, pts0, conf, 
        K, R_g, reproj_thr
    )

    n_valid = match_counts[-1][1]
    assert n_valid == len(inlier_idx), "Length mismatch after get_initial_pose"
    
    # reproj_stats_after_inliers = compute_mean_reprojection_error(
    #     pts0[inlier_idx], pts3d[inlier_idx], K, R0, t0
    # )
    return R0, t0, which_query


def project_object_points(points_3d: torch.Tensor, R: torch.Tensor, t: torch.Tensor, K: torch.Tensor, bbox):
    W, H = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # points_3d: (3, N), R: (3, 3), t: (3) -> points_cam: (3, N)
    points_cam = R @ points_3d + t.unsqueeze(1)
    # projection: K @ X_cam
    points_pixel_homo = K @ points_cam          # (3, N)
    
    # 원근 나눗셈 (Homogeneous -> 2D 픽셀)
    x_2d = points_pixel_homo[0, :] / (points_pixel_homo[2, :] + 1e-8) - bbox[0]
    y_2d = points_pixel_homo[1, :] / (points_pixel_homo[2, :] + 1e-8) - bbox[1]
    
    # PyTorch grid_sample용 [-1, 1] 정규화
    grid_x_norm = 2.0 * (x_2d / (W - 1)) - 1.0
    grid_y_norm = 2.0 * (y_2d / (H - 1)) - 1.0
    
    # grid_sample 입력 형태 맞추기: (1, 1, N, 2)
    grid = torch.stack([grid_x_norm, grid_y_norm], dim=-1)
    
    return grid


def refine_pose_GS(query_info, gProxy:GaussianRenderer, options):
    device = 'cuda'
    
    # query bounding box 
    q_bbox = np.array(query_info["bbox"])
    q_side = max(get_bbox_size(q_bbox))
    bbox_fixed = square_bbox(q_bbox, int(q_side * 1.2) )

    # crop the mask and masked query image for the refinement step
    qm = crop_with_bbox(query_info["mask"], bbox_fixed)
    query_mask = torch.from_numpy(qm).float() / 255.0
    query_mask = query_mask.to(device)

    mq = crop_with_bbox(query_info["masked_query"], bbox_fixed)
    query_crop = torch.from_numpy(mq).float().permute(2,0,1) / 255.0
    query_crop = query_crop.to(device)

    R0, t0 = gProxy.get_T()

    # ──────────────────────────────────────────────────────
    # 4. Optimization
    # ──────────────────────────────────────────────────────
    delta_r = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)    # rodrigues vector for rotation update
    delta_t = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)    # translation update

    optimizer = torch.optim.AdamW([
        {"params": [delta_r], "lr": options["lr_rot"]},
        {"params": [delta_t], "lr": options["lr_trans"]},
    ])

    scheduler = CosineWarmupScheduler(
        optimizer,
        total_steps  = options["iters"],
        warmup_steps = options["warmup_steps"],
        max_lr       = options["lr_rot"],
        min_lr       = options["lr_rot"] * 0.01,
    )

    losses = []
    best_loss  = 1e9
    best_state = {
        "R": R0.detach(),
        "t": t0.detach(),
        "iter": 0,
    }

    ecc_loss_fn = ECCLoss()

    for it in range(options["iters"]):
        optimizer.zero_grad()

        dR    = so3_exp_map(delta_r)
        R_cur = dR @ R0
        # tz clamp: init_t의 ±40% 범위로 제한 (frustum 이탈 방지)
        t0z = float(t0[2].item())
        t_raw = t0 + delta_t
        t_cur = torch.stack( [t_raw[0],  t_raw[1],  torch.clamp(t_raw[2], min=t0z*0.6, max=t0z*1.4)] )

        # full resolution render with identity cam (gsplat)
        gProxy.set_T(R_cur, t_cur)
        _r, _, _ = gProxy.render(width = options["width"], height=options["height"])
        
        render_full = _r[0].permute(2, 0, 1).clamp(0, 1)
        render_crop = crop_chw_with_bbox(render_full, bbox_fixed)

        # 1. Silhouette (Mask) Loss
        masked_render_crop = render_crop * query_mask.unsqueeze(0)
        render_alpha = (masked_render_crop.sum(dim=0, keepdim=True) > 0.05).float()
        loss_mask = F.l1_loss(render_alpha.squeeze(), query_mask)

        # 2. RGB Loss (SSIM, L1)
        loss_ssim = dssim_loss(masked_render_crop.unsqueeze(0), query_crop.unsqueeze(0)) 
        loss_ssim += dms_ssim_loss(masked_render_crop.unsqueeze(0), query_crop.unsqueeze(0))
        loss_l1_rgb = F.l1_loss(masked_render_crop, query_crop)

        # 3. Blur Loss
        blur_target = F.avg_pool2d(query_crop, kernel_size=9, stride=1, padding=4)
        blur_render = F.avg_pool2d(masked_render_crop, kernel_size=9, stride=1, padding=4)
        loss_blur = F.l1_loss(blur_render, blur_target)

        ecc_loss = ecc_loss_fn(masked_render_crop, query_crop)

        # 4. [핵심] Dynamic Weighting (Coarse-to-Fine)
        # 초반: Blur 중심 (크게 돌리기) / 후반: SSIM 중심 (칼같이 맞추기)
        progress = it / max(1, options["iters"])    # 현재 학습 진행도 (0.0 ~ 1.0)
        weight_blur = 1.0 * (1.0 - progress)        # 1.0에서 0.0으로 서서히 감소
        weight_ssim = 0.1 + 0.9 * progress          # 0.1에서 1.0으로 서서히 증가
        weight_l1   = 0.5                           # 기본 위치 유지를 위해 고정ge
        weight_mask = 1.0                           # 크기 유지를 위해 고정

        loss = (weight_blur * loss_blur) + ecc_loss + (weight_mask * loss_mask) # (weight_ssim * loss_ssim) + (weight_l1 * loss_l1_rgb) +

        loss.backward()

        torch.nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

        optimizer.step()
        scheduler.step()

        tracking_loss = float(loss_ssim.item() + loss_l1_rgb.item() + loss_blur.item() + loss_mask.item() + ecc_loss.item()) #  
        loss_val = float(loss.item())
        losses.append({"iter": it, "loss": loss_val, "track_loss": tracking_loss})

        # 이제 loss_val이 아닌 tracking_loss 기준으로 최고를 갱신합니다.
        if tracking_loss < best_loss:
            best_loss  = tracking_loss
            best_state = {
                "R": R_cur.detach(),
                "t": t_cur.detach(),
                "iter": it,
                "loss": loss_val,
                "track_loss": tracking_loss
            }

        # Early stopping
        if it >= options["early_stop_steps"]:
            loss_vals  = torch.tensor([l["loss"] for l in losses])
            loss_grads = (loss_vals[1:] - loss_vals[:-1]).abs()
            recent_grad = loss_grads[-options["early_stop_steps"]:].mean().item()
            if recent_grad < options["early_stop_thr"]:
                print(f"  [EarlyStop] iter={it}  grad_norm={recent_grad:.2e}")
                break

    # ──────────────────────────────────────────────────────
    # 5. 저장
    # ──────────────────────────────────────────────────────
    best_R = best_state["R"].cpu().numpy().copy()
    best_t = best_state["t"].cpu().numpy().copy()
    
    return best_R, best_t, best_state["track_loss"], bbox_fixed
    

def adjust_stereo_bbox( bbox_l, bbox_r ):
    bbox_l = np.array(bbox_l)
    bbox_r = np.array(bbox_r)
    # correspondences in rectified stereo have the same vertical positions
    sy = min(bbox_l[1], bbox_r[1])
    ey = max(bbox_l[3], bbox_r[3])
    bbox_l[1], bbox_l[3] = sy, ey   
    bbox_r[1], bbox_r[3] = sy, ey
    q_side = max(get_bbox_size(bbox_l))
    q_r_side = max(get_bbox_size(bbox_r))
    q_side = max(q_side, q_r_side)      # the largest side (left & right)
    
    return square_bbox(bbox_l, int(q_side * 1.2)), square_bbox(bbox_r, int(q_side * 1.2))


def refine_pose_stereo_GS(query_l_info, query_r_info, 
                          gProxy:GaussianRenderer, options, 
                          R_lr, t_lr):
    device = torch.device('cuda')
    
    # query bounding box 
    bbox_l, bbox_r = adjust_stereo_bbox(query_l_info["bbox"], query_r_info["bbox"]) 
    query_l,  query_r  = {"bbox": bbox_l}, {"bbox": bbox_r}
    render_l, render_r = {"bbox": bbox_l}, {"bbox": bbox_r}    
    
    # crop the mask and masked query image for the refinement step
    # c_ : cropped, m_ : masked
    query_l["c_mask"] = np2t(crop_with_bbox(query_l_info["mask"], bbox_l)) / 255.0    
    query_r["c_mask"] = np2t(crop_with_bbox(query_r_info["mask"], bbox_r)) / 255.0
    query_l["m_crop"] = np2t(crop_with_bbox(query_l_info["masked_query"], bbox_l)) / 255.0
    query_r["m_crop"] = np2t(crop_with_bbox(query_r_info["masked_query"], bbox_r)) / 255.0
    # qc = npf2img(t2np(query_l["crop"]))
    # qcr = npf2img(t2np(query_r["crop"]))
    # plt.imshow( np.hstack(qc, qcr) )

    # ──────────────────────────────────────────────────────
    # 4. Optimization
    # ──────────────────────────────────────────────────────
    delta_r = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)    # rodrigues vector for rotation update
    delta_t = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)    # translation update

    optimizer = torch.optim.AdamW([
        {"params": [delta_r], "lr": options["lr_rot"]},
        {"params": [delta_t], "lr": options["lr_trans"]},
    ])

    scheduler = CosineWarmupScheduler(
        optimizer,
        total_steps  = options["iters"],
        warmup_steps = options["warmup_steps"],
        max_lr       = options["lr_rot"],
        min_lr       = options["lr_rot"] * 0.01,
    )

    R0, t0 = gProxy.get_T()

    losses = []
    best_loss  = 1e9
    best_state = { "R": R0.detach(), "t": t0.detach(), "iter": 0 }

    ecc_loss_fn = ECCLoss()

    for it in range(options["iters"]):
        optimizer.zero_grad()

        # so3_exp_map(delta_r) : dR
        R_cur = so3_exp_map(delta_r) @ R0
        # tz clamp: init_t의 ±40% 범위로 제한 (frustum 이탈 방지)
        t0z = float(t0[2].item())
        t_raw = t0 + delta_t
        t_cur = torch.stack( [t_raw[0],  t_raw[1],  torch.clamp(t_raw[2], min=t0z*0.6, max=t0z*1.4)] )

        # current pose of the right cam
        R_r_cur = R_lr @ R_cur
        t_r_cur = t_cur @ R_lr.T + t_lr 

        # full resolution render with identity cam (gsplat)
        gProxy.set_T(R_cur, t_cur)
        _r, _a, _ = gProxy.render()        
        render_l_f = _r[0].permute(2,0,1).clamp(0,1)
        render_l["crop"] = crop_chw_with_bbox(render_l_f, render_l["bbox"])

        gProxy.set_T(R_r_cur, t_r_cur)
        _r_r, _a_r, _ = gProxy.render()
        render_r_f = _r_r[0].permute(2,0,1).clamp(0,1)
        render_r["crop"] = crop_chw_with_bbox(render_r_f, render_r["bbox"])

        loss_l = {}
        loss_r = {}
        # 1. Silhouette (Mask) Loss - render_crop:chw, query_mask: hw
        render_l["c_mask"] = crop_with_bbox((_a[0].squeeze() > 0.5).float(), render_l["bbox"])
        loss_l["mask"] = dice_loss(render_l["c_mask"], query_l["c_mask"])
        render_l["c_mask"] = erode_binary_tensor(render_l["c_mask"].unsqueeze(0), 3).squeeze(0)
        render_l["ci_mask"] = render_l["c_mask"] * query_l["c_mask"]
        render_l["mi_crop"] = render_l["crop"] * render_l["ci_mask"]        
        
        render_r["c_mask"] = crop_with_bbox((_a_r[0].squeeze() > 0.5).float(), render_r["bbox"])
        loss_r["mask"] = dice_loss(render_r["c_mask"], query_r["c_mask"])
        render_r["c_mask"] = erode_binary_tensor(render_r["c_mask"].unsqueeze(0), 3).squeeze(0)
        render_r["ci_mask"] = render_r["c_mask"] * query_r["c_mask"]
        render_r["mi_crop"] = render_r["crop"] * render_r["ci_mask"]
        
        r_tensors = torch.stack((render_l["mi_crop"], render_r["mi_crop"]))
        q_tensors = torch.stack((query_l["m_crop"], query_r["m_crop"]))

        # 2. RGB Loss (SSIM, L1)
        loss_ssim = (dssim_loss(r_tensors, q_tensors) + dms_ssim_loss(r_tensors, q_tensors)) * 2
        loss_l1_rgb = F.l1_loss(r_tensors, q_tensors) * 2

        # # 3. Blur Loss
        r_blur = F.avg_pool2d(r_tensors, kernel_size=5, stride=1, padding=2)
        q_blur = F.avg_pool2d(q_tensors, kernel_size=5, stride=1, padding=2)
        loss_l["blur"] = F.l1_loss(r_blur, q_blur) * 2
        loss_r["blur"] = 0

        loss_l["ecc"] = ecc_loss_fn(render_l["mi_crop"], query_l["m_crop"])
        loss_r["ecc"] = ecc_loss_fn(render_r["mi_crop"], query_r["m_crop"])

        loss_l["grad"] = gradient_matching_loss(
                                r_tensors, q_tensors, 
                                torch.stack((render_l["ci_mask"], render_r["ci_mask"])).unsqueeze(1)) * 2
        loss_r["grad"] = 0 

        # 4. [핵심] Dynamic Weighting (Coarse-to-Fine)
        # 초반: Blur 중심 (크게 돌리기) / 후반: SSIM 중심 (칼같이 맞추기)
        progress = it / max(1, options["iters"])    # 현재 학습 진행도 (0.0 ~ 1.0)
        weight_blur = 1.0 * (1.0 - progress)        # 1.0에서 0.0으로 서서히 감소
        weight_ssim = 0.1 + 0.9 * progress          # 0.1에서 1.0으로 서서히 증가
        weight_l1   = 2.5                           # 기본 위치 유지를 위해 고정ge
        weight_mask = 1.0                           # 크기 유지를 위해 고정

        loss_l["ecc"] *= 2
        loss_r["ecc"] *= 2 
        loss_l["grad"] *= 4
        loss_r["grad"] *= 4
        loss_l["blur"] *= 5
        loss_r["blur"] *= 5
        loss = (loss_l["ecc"] + loss_r["ecc"]) + (loss_l["grad"] + loss_r["grad"]) + loss_ssim
        loss = loss + (weight_blur * (loss_l["mask"] + loss_r["mask"])) + (weight_l1 * loss_l1_rgb)
        loss = loss + (weight_blur * (loss_l["blur"] + loss_r["blur"]))
        loss.backward()

        torch.nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

        optimizer.step()
        scheduler.step()

        tracking_loss = (loss_l["ecc"] + loss_r["ecc"]) + (loss_l["grad"] + loss_r["grad"]) + loss_ssim
        tracking_loss = tracking_loss + (loss_l["mask"] + loss_r["mask"]) + (weight_l1 * loss_l1_rgb)
        tracking_loss = tracking_loss + (loss_l["blur"] + loss_r["blur"])
        loss_val = float(tracking_loss.item())

        losses.append({"iter": it, "ecc_loss": (loss_l["ecc"] + loss_r["ecc"]).item(), 
                       "grad_loss": (loss_l["grad"] + loss_r["grad"]).item(), 
                       "blur_loss": (loss_l["blur"] + loss_r["blur"]).item(),
                       "mask_loss": (loss_l["mask"] + loss_r["mask"]).item(),
                       "ssim_loss": loss_ssim.item(),
                       "rgb_loss": loss_l1_rgb.item(),
                       "loss": loss_val})
        
        # 이제 loss_val이 아닌 tracking_loss 기준으로 최고를 갱신합니다.
        if loss_val < best_loss:
            best_loss  = loss_val
            best_state = {
                "R": R_cur.detach(),
                "t": t_cur.detach(),
                "iter": it,
                "loss": loss_val,                
                # "track_loss": tracking_loss
            }

        # Early stopping
        if it >= options["early_stop_steps"]:
            loss_vals  = torch.tensor([l["loss"] for l in losses])
            loss_grads = (loss_vals[1:] - loss_vals[:-1]).abs()
            recent_grad = loss_grads[-options["early_stop_steps"]:].mean().item()
            if recent_grad < options["early_stop_thr"]:
                print(f"  [EarlyStop] iter={it}  grad_norm={recent_grad:.2e}")
                break

    # ──────────────────────────────────────────────────────
    # 5. 저장
    # ──────────────────────────────────────────────────────
    best_R = best_state["R"].cpu().numpy().copy()
    best_t = best_state["t"].cpu().numpy().copy()
    
    return best_R, best_t, best_state["loss"], (query_l, query_r), losses


def refine_pose_rerender(query_info, gProxy:GaussianRenderer, options):
    device = torch.device('cuda')
    
    # query bounding box 
    q_bbox = np.array(query_info["bbox"])
    q_side = max(get_bbox_size(q_bbox))
    query = {"bbox": square_bbox(q_bbox, int(q_side * 1.2) )}
    render = {"bbox": query["bbox"]}
    
    # crop the mask and masked query image for the refinement step
    # c_ : cropped, m_ : masked
    query["c_mask"] = np2t(crop_with_bbox(query_info["mask"], query["bbox"])) / 255.0
    query["m_crop"] = np2t(crop_with_bbox(query_info["masked_query"], query["bbox"])) / 255.0
    
    # ──────────────────────────────────────────────────────
    # 4. Optimization
    # ──────────────────────────────────────────────────────
    delta_r = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)    # rodrigues vector for rotation update
    delta_t = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)    # translation update

    optimizer = torch.optim.AdamW([
                    {"params": [delta_r], "lr": options["lr_rot"]},
                    {"params": [delta_t], "lr": options["lr_trans"]},
                ])

    scheduler = CosineWarmupScheduler(
                    optimizer,
                    total_steps  = options["iters"],
                    warmup_steps = options["warmup_steps"],
                    max_lr       = options["lr_rot"],
                    min_lr       = options["lr_rot"] * 0.01,
                )

    ecc_loss_fn = ECCLoss()

    # initial pose for left and right cam
    R0, t0 = gProxy.get_T()

    K = gProxy.K_mat.squeeze(0)
    fx, fy = t2np(K.diag()[:2])
    cx, cy = t2np(K[:2, 2])

    losses = []
    best_loss  = 1e9
    best_state = {"R": R0.detach(), "t": t0.detach(), "iter": 0}

    _r, _a, _ = gProxy.render_no_grad("RGB+ED")
    render_chw = _r[0][:3]
    depth_hw = _r[0][3]
    # with torch.no_grad():
    #     render_chw, depth_hw = render_with_gsplat(
    #         gaussians = gProxy.base,
    #         R_obj_to_cam = R0, t_obj_to_cam = t0,
    #         width = options["width"], height=options["height"],
    #         fx=fx, fy=fy, cx=cx, cy=cy,
    #         bg_color_str="0,0,0", device=device,
    #         render_depth=True,
    #     )        
    xyz_obj, _ = depth_tensor_to_xyz_map(depth_hw,
            fx = fx, fy = fy, cx = cx, cy = cy,
            R = R0, t = t0)
    
    st = time.perf_counter()    
    render["crop0"] = crop_with_bbox(render_chw, render["bbox"])
    render["c_mask"] = crop_with_bbox(depth_hw > FRUSTUM_N, render["bbox"])    
    render["ci_mask"] = render["c_mask"] & query["c_mask"].bool()
    render["cif_mask"] = render["ci_mask"].float()

    xyz = crop_with_bbox(xyz_obj, render["bbox"])
    xyz = xyz[:, render["ci_mask"]]
    xyz_colors = render["crop0"][:, render["ci_mask"]]
    
    for it in range(options["iters"]):
        optimizer.zero_grad()

        # so3_exp_map(delta_r) : dR
        R_cur = so3_exp_map(delta_r) @ R0
        # tz clamp: init_t의 ±40% 범위로 제한 (frustum 이탈 방지)
        t0z = float(t0[2].item())
        t_raw = t0 + delta_t
        t_cur = torch.stack( [t_raw[0],  t_raw[1],  torch.clamp(t_raw[2], min=t0z*0.6, max=t0z*1.4)] )

        pts2d = project_object_points(xyz, R_cur, t_cur, K, render["bbox"])
        sampled_colors = F.grid_sample(query["m_crop"].unsqueeze(0), pts2d.unsqueeze(0).unsqueeze(0),
                                       mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_colors = sampled_colors.squeeze(0).squeeze(1)

        # reconstructed masked cropped query
        query["rm_crop"] = torch.zeros_like(query["m_crop"])
        query["rm_crop"][:, render["ci_mask"]] = sampled_colors

        # # 4. [핵심] Dynamic Weighting (Coarse-to-Fine)
        # # 초반: Blur 중심 (크게 돌리기) / 후반: SSIM 중심 (칼같이 맞추기)
        progress = it / max(1, options["iters"])    # 현재 학습 진행도 (0.0 ~ 1.0)
        weight_blur = 1.0 * (1.0 - progress)        # 1.0에서 0.0으로 서서히 감소
        weight_l1   = 2.5

        loss = {}
        loss["ssim"] = dssim_loss(render["crop0"].unsqueeze(0), query["rm_crop"].unsqueeze(0))
        loss["ssim"] += dms_ssim_loss(render["crop0"].unsqueeze(0), query["rm_crop"].unsqueeze(0))
        loss["rgb"] = F.l1_loss(sampled_colors, xyz_colors)

        blur_target = F.avg_pool2d(query["rm_crop"], kernel_size=5, stride=1, padding=2)
        blur_render = F.avg_pool2d(render["crop0"], kernel_size=5, stride=1, padding=2)
        loss["blur"] = F.l1_loss(blur_render, blur_target)

        loss["grad"] = gradient_matching_loss(render["crop0"], query["rm_crop"], render["cif_mask"])

        loss["ecc"] = ecc_loss_fn(sampled_colors, xyz_colors) * 2
        loss["grad"] *= 4
        loss["blur"] *= 5
        total_loss = loss["ecc"] + loss["grad"] + loss["ssim"] 
        total_loss = total_loss + (weight_blur * loss["blur"]) + (weight_l1 * loss["rgb"]) 

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

        optimizer.step()
        scheduler.step()

        tracking_loss = loss["ecc"] + loss["grad"] + loss["ssim"] 
        tracking_loss = tracking_loss + (weight_l1 * loss["rgb"])
        tracking_loss = tracking_loss + loss["blur"]
        loss_val = float(tracking_loss.item())

        losses.append({"iter": it, "ecc_loss": loss["ecc"].item(), 
                       "grad_loss": loss["grad"].item(),
                       "blur_loss": loss["blur"].item(),
                       "ssim_loss": loss["ssim"].item(),
                       "rgb_loss": loss["rgb"].item(),
                       "loss": loss_val})
        
        if loss_val < best_loss:
            best_loss = loss_val
            best_state["R"] = R_cur.detach()
            best_state["t"] = t_cur.detach()
            best_state["loss"] = best_loss
            best_state["iter"] = it

        # Early stopping
        if it >= options["early_stop_steps"]:
            loss_vals  = torch.tensor([l["loss"] for l in losses])
            loss_grads = (loss_vals[1:] - loss_vals[:-1]).abs()
            recent_grad = loss_grads[-options["early_stop_steps"]:].mean().item()
            if recent_grad < options["early_stop_thr"]:
                print(f"  [EarlyStop] iter={it}  grad_norm={recent_grad:.2e}")
                break

    et = time.perf_counter()    
    print(f"  [Timing] {options['iters']} projections: {(et - st):.3f}s")

    # ──────────────────────────────────────────────────────
    # 5. 저장
    # ──────────────────────────────────────────────────────
    best_R = best_state["R"].cpu().numpy().copy()
    best_t = best_state["t"].cpu().numpy().copy()
    
    return best_R, best_t, best_state["loss"], query["bbox"], losses


def refine_pose_stereo_rerender(query_l_info, query_r_info, 
                                gProxy:GaussianRenderer, options, 
                                R_lr, t_lr):
    device = torch.device('cuda')
    
    # query bounding box 
    bbox_l, bbox_r = adjust_stereo_bbox(query_l_info["bbox"], query_r_info["bbox"]) 
    query_l,  query_r  = {"bbox": bbox_l}, {"bbox": bbox_r}
    render_l, render_r = {"bbox": bbox_l}, {"bbox": bbox_r}
    
    # crop the mask and masked query image for the refinement step
    # c_ : cropped, m_ : masked
    query_l["c_mask"] = np2t(crop_with_bbox(query_l_info["mask"], bbox_l)) / 255.0    
    query_r["c_mask"] = np2t(crop_with_bbox(query_r_info["mask"], bbox_r)) / 255.0
    query_l["m_crop"] = np2t(crop_with_bbox(query_l_info["masked_query"], bbox_l)) / 255.0
    query_r["m_crop"] = np2t(crop_with_bbox(query_r_info["masked_query"], bbox_r)) / 255.0
    # qc = npf2img(t2np(query_l["crop"]))
    # qcr = npf2img(t2np(query_r["crop"]))
    # plt.imshow( np.hstack(qc, qcr) )

    # ──────────────────────────────────────────────────────
    # 4. Optimization
    # ──────────────────────────────────────────────────────
    delta_r = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)    # rodrigues vector for rotation update
    delta_t = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)    # translation update

    optimizer = torch.optim.AdamW([
        {"params": [delta_r], "lr": options["lr_rot"]},
        {"params": [delta_t], "lr": options["lr_trans"]},
    ])

    scheduler = CosineWarmupScheduler(
        optimizer,
        total_steps  = options["iters"],
        warmup_steps = options["warmup_steps"],
        max_lr       = options["lr_rot"],
        min_lr       = options["lr_rot"] * 0.01,
    )

    _r, _a, _ = gProxy.render_no_grad("RGB+ED")

    # initial pose for left and right cam
    R0, t0 = gProxy.get_T()
    R_r0 = R_lr @ R0
    t_r0 = t0 @ R_lr.T + t_lr 

    K = gProxy.K_mat.squeeze(0)
    fx, fy = t2np(K.diag()[:2])
    cx, cy = t2np(K[:2, 2])

    losses = []
    best_loss  = 1e9
    best_state = { "R": R0.detach(), "t": t0.detach(), "iter": 0 }

    render_l_chw = _r[0][..., :3].permute(2,0,1).clamp(0.0, 1.0)
    depth_l_hw = _r[0][..., 3]
    render_r_chw = _r[1][..., :3].permute(2,0,1).clamp(0.0, 1.0)
    depth_r_hw = _r[1][..., 3]    
    
    xyz_obj_l, _ = depth_tensor_to_xyz_map(depth_l_hw, R = R0, t = t0,
                                            fx = fx, fy = fy, cx = cx, cy = cy)
    xyz_obj_r, _ = depth_tensor_to_xyz_map(depth_r_hw, R = R_r0, t = t_r0,
                                            fx = fx, fy = fy, cx = cx, cy = cy)
    
    render_l["crop0"] = crop_with_bbox(render_l_chw, render_l["bbox"])
    render_l["c_mask"] = crop_with_bbox((_a[0].squeeze() > 0.5), render_l["bbox"])
    render_l["ci_mask"] = render_l["c_mask"] & query_l["c_mask"].bool()
    render_l["cif_mask"] = render_l["ci_mask"].float()

    render_r["crop0"] = crop_with_bbox(render_r_chw, render_r["bbox"])
    render_r["c_mask"] = crop_with_bbox((_a[1].squeeze() > 0.5), render_r["bbox"])
    render_r["ci_mask"] = render_r["c_mask"] & query_r["c_mask"].bool()
    render_r["cif_mask"] = render_r["ci_mask"].float()

    # xyz, xyz_colors : tensors
    xyz_l = crop_with_bbox(xyz_obj_l, render_l["bbox"])
    xyz_l = xyz_l[:, render_l["ci_mask"]]                       # 물체 픽셀별 xyz
    xyz_colors_l = render_l["crop0"][:, render_l["ci_mask"]]    # 물체 픽셀별 color
    
    xyz_r = crop_with_bbox(xyz_obj_r, render_r["bbox"])
    xyz_r = xyz_r[:, render_r["ci_mask"]]
    xyz_colors_r = render_r["crop0"][:, render_r["ci_mask"]]

    for it in range(options["iters"]):
        optimizer.zero_grad()

        # so3_exp_map(delta_r) : dR
        R_cur = so3_exp_map(delta_r) @ R0
        # tz clamp: init_t의 ±40% 범위로 제한 (frustum 이탈 방지)
        t0z = float(t0[2].item())
        t_raw = t0 + delta_t
        t_cur = torch.stack( [t_raw[0],  t_raw[1],  torch.clamp(t_raw[2], min=t0z*0.6, max=t0z*1.4)] )

        # current pose of the right cam
        R_r_cur = R_lr @ R_cur
        t_r_cur = t_cur @ R_lr.T + t_lr 

        # 물체 3차원 점의 2차원 프로젝션, normalized position
        # xyz_l이 cropped이기 때문에, bbox offset을 고려해 pts2d_l은 unmapped된 상태임
        pts2d_l = project_object_points(xyz_l, R_cur, t_cur, K, render_l["bbox"])   
        sampled_colors_l = F.grid_sample(query_l["m_crop"].unsqueeze(0), pts2d_l.unsqueeze(0).unsqueeze(0),
                                       mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_colors_l = sampled_colors_l.squeeze(0).squeeze(1)

        pts2d_r = project_object_points(xyz_r, R_r_cur, t_r_cur, K, render_r["bbox"])
        sampled_colors_r = F.grid_sample(query_r["m_crop"].unsqueeze(0), pts2d_r.unsqueeze(0).unsqueeze(0),
                                       mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_colors_r = sampled_colors_r.squeeze(0).squeeze(1)
        
        query_l["rm_crop"] = torch.zeros_like(query_l["m_crop"])  # 만약에 query 색으로 재구성된 query. Rt가 잘 맞는다면, query에서 뽑혀진 sampled_colors가 query에 잘 그려짐
        query_l["rm_crop"][:, render_l["ci_mask"]] = sampled_colors_l
        query_r["rm_crop"] = torch.zeros_like(query_r["m_crop"])
        query_r["rm_crop"][:, render_r["ci_mask"]] = sampled_colors_r

        # 초반: Blur 중심 (크게 돌리기) / 후반: SSIM 중심 (칼같이 맞추기)
        progress = it / max(1, options["iters"])    # 현재 학습 진행도 (0.0 ~ 1.0)
        weight_blur = 1.0 * (1.0 - progress)        # 1.0에서 0.0으로 서서히 감소
        weight_l1   = 2.5

        total_loss, loss_sub = calc_losses_rerender(
                torch.stack((render_l["crop0"], render_r["crop0"])), 
                torch.stack((query_l["rm_crop"], query_r["rm_crop"])),
                [xyz_colors_l, xyz_colors_r], [sampled_colors_l, sampled_colors_r], 
                torch.stack((render_l["cif_mask"], render_r["cif_mask"])).unsqueeze(1), 
                weight_blur
            )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

        optimizer.step()
        scheduler.step()

        tracking_loss = loss_sub["ecc"] 
        tracking_loss += loss_sub["ssim"]
        tracking_loss += loss_sub["grad"]
        tracking_loss += loss_sub["blur"]
        # tracking_loss += (weight_l1 * loss_sub["rgb"])        
        loss_val = float(tracking_loss.item())

        losses.append({"iter": it, 
                       "ecc_loss": loss_sub["ecc"].item(), 
                       "ssim_loss": loss_sub["ssim"].item(),
                       "grad_loss": loss_sub["grad"].item(), 
                       "blur_loss": loss_sub["blur"].item(),
                    #    "rgb_loss": (loss_l["rgb"]+loss_r["rgb"]).item(),
                       "loss": loss_val})
        
        # 이제 loss_val이 아닌 tracking_loss 기준으로 최고를 갱신합니다.
        if loss_val < best_loss:
            best_loss  = loss_val
            best_state = {
                "R": R_cur.detach(),
                "t": t_cur.detach(),
                "iter": it,
                "loss": loss_val,                
                # "track_loss": tracking_loss
            }

        # Early stopping
        if it >= options["early_stop_steps"]:
            loss_vals  = torch.tensor([l["loss"] for l in losses])
            loss_grads = (loss_vals[1:] - loss_vals[:-1]).abs()
            recent_grad = loss_grads[-options["early_stop_steps"]:].mean().item()
            if recent_grad < options["early_stop_thr"]:
                print(f"  [EarlyStop] iter={it}  grad_norm={recent_grad:.2e}")
                break

    # ──────────────────────────────────────────────────────
    # 5. 저장
    # ──────────────────────────────────────────────────────
    best_R = best_state["R"].cpu().numpy().copy()
    best_t = best_state["t"].cpu().numpy().copy()
    
    return best_R, best_t, best_state["loss"], (query_l, query_r), losses


class SoftLBPLoss(nn.Module):
    def __init__(self, temperature=0.1):
        """
        temperature (tau): 계단 함수를 얼마나 부드럽게 근사할지 결정하는 파라미터.
        값이 작을수록 원래의 계단 함수(0 또는 1)에 가까워지고, 클수록 기울기가 부드러워집니다.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, pred, target):
        # 1. 컬러(RGB) 텐서를 그레이스케일로 변환 [B, C, H, W] -> [B, 1, H, W]
        # RGB 채널 순서라고 가정하고 휘도(Luminosity) 공식을 적용합니다.
        if pred.size(1) == 3:
            pred = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

        # 2. 3x3 패치 추출 (Unfold 사용)
        # 결과 형태: [B, 9, H*W]
        pred_unfold = F.unfold(pred, kernel_size=3, padding=1)
        target_unfold = F.unfold(target, kernel_size=3, padding=1)

        # 3. 중심 픽셀(인덱스 4)과 8개의 이웃 픽셀 분리
        center_idx = 4
        neighbor_indices = [0, 1, 2, 3, 5, 6, 7, 8]

        pred_center = pred_unfold[:, center_idx:center_idx+1, :]
        target_center = target_unfold[:, center_idx:center_idx+1, :]

        pred_neighbors = pred_unfold[:, neighbor_indices, :]
        target_neighbors = target_unfold[:, neighbor_indices, :]

        # 4. 차이 계산 및 Soft Thresholding (Sigmoid 적용)
        # 주변 픽셀 - 중심 픽셀을 한 뒤 시그모이드를 통과시켜 0~1 사이의 값으로 만듭니다.
        pred_diff = (pred_neighbors - pred_center) / self.temperature
        target_diff = (target_neighbors - target_center) / self.temperature

        pred_soft_lbp = torch.sigmoid(pred_diff)
        target_soft_lbp = torch.sigmoid(target_diff)

        # 5. 두 Soft LBP 특징 간의 평균 절대 오차(L1 Loss) 반환
        loss = F.l1_loss(pred_soft_lbp, target_soft_lbp)
        
        return loss
    

def calc_patched_ecc(r, q):
    ecc_loss_fn = ECCLoss()
    if r.ndim == 2:
        assert r.shape[0] == 3
        num_pxls = r.shape[1]
        len_patch = num_pxls // 4
        L = [len_patch, len_patch*2, len_patch*3]
        loss_val  = ecc_loss_fn(r[:,     :L[0]], q[:,     :L[0]])
        loss_val += ecc_loss_fn(r[:, L[0]:L[1]], q[:, L[0]:L[1]])
        loss_val += ecc_loss_fn(r[:, L[1]:L[2]], q[:, L[1]:L[2]])
        loss_val += ecc_loss_fn(r[:, L[2]:    ], q[:, L[2]:    ])
    else:
        assert r.ndim == 3
        H, W = r.shape[1:3]
        H2, W2 = H//2, W//2
        loss_val  = ecc_loss_fn(r[:, :H2, :W2], q[:, :H2, :W2])
        loss_val += ecc_loss_fn(r[:, :H2, W2:], q[:, :H2, W2:])
        loss_val += ecc_loss_fn(r[:, H2:, :W2], q[:, H2:, :W2])
        loss_val += ecc_loss_fn(r[:, H2:, W2:], q[:, H2:, W2:])

    return loss_val
    

def calc_losses_rerender(r_tensors: torch.tensor, q_tensors: torch.tensor, 
                 r_colors: torch.tensor, q_colors: torch.tensor, 
                 masks: torch.tensor, weight_blur):
    weight_l1   = 2.5
    loss_sub = {}    

    loss_sub["ssim"] = dssim_loss(r_tensors, q_tensors) * 2    

    r_blurs = F.avg_pool2d( r_tensors, kernel_size=5, stride=1, padding=2 )
    q_blurs = F.avg_pool2d( q_tensors, kernel_size=5, stride=1, padding=2 )
    loss_sub["blur"] = F.l1_loss(r_blurs, q_blurs) * 10    

    loss_sub["grad"] = gradient_matching_loss(r_tensors, q_tensors, masks) * 8    

    loss_sub["ecc"] = calc_patched_ecc(r_colors[0], q_colors[0])
    loss_sub["ecc"] += calc_patched_ecc(r_colors[1], q_colors[1])

    total_loss = loss_sub["ecc"]
    total_loss += loss_sub["ssim"]
    total_loss += loss_sub["grad"]
    total_loss += (weight_blur * loss_sub["blur"]) 
    # total_loss += (weight_l1 * (loss_l["rgb"]+loss_r["rgb"])) 

    return total_loss, loss_sub


def calc_losses_GS(r_tensors: torch.tensor, q_tensors: torch.tensor, weight_blur):
    loss_sub = {}
    
    # 1. Silhouette (Mask) Loss - render_crop:chw, query_mask: hw
    loss_sub["mask"] = dice_loss(r_tensors[0]["c_mask"], q_tensors[0]["c_mask"])
    loss_sub["mask"] += dice_loss(r_tensors[1]["c_mask"], q_tensors[1]["c_mask"])
    
    # 2. RGB Loss (SSIM, L1)
    r_t4d = torch.stack((r_tensors[0]["mi_crop"], r_tensors[1]["mi_crop"]))
    q_t4d = torch.stack((q_tensors[0]["m_crop"], q_tensors[1]["m_crop"]))
    loss_sub["ssim"] = dssim_loss( r_t4d, q_t4d ) * 2    
    
    # loss_sub["l1_rgb"] = F.l1_loss(r_t4d, q_t4d) * 2

    # # 3. Blur Loss
    r_blur = F.avg_pool2d(r_t4d, kernel_size=5, stride=1, padding=2)
    q_blur = F.avg_pool2d(q_t4d, kernel_size=5, stride=1, padding=2)
    loss_sub["blur"] = F.l1_loss(r_blur, q_blur) * 10    
    
    loss_sub["ecc"] = calc_patched_ecc(r_tensors[0]["mi_crop"], q_tensors[0]["m_crop"])
    loss_sub["ecc"] += calc_patched_ecc(r_tensors[1]["mi_crop"], q_tensors[1]["m_crop"])

    loss_sub["grad"] = gradient_matching_loss(r_t4d, q_t4d, 
                                              torch.stack((r_tensors[0]["cif_mask"], r_tensors[1]["cif_mask"])).unsqueeze(1)) * 8

    # 4. [핵심] Dynamic Weighting (Coarse-to-Fine)
    weight_l1   = 2.5

    loss = loss_sub["ecc"]
    loss += loss_sub["ssim"]
    loss += loss_sub["grad"]
    loss += (weight_blur * loss_sub["mask"])     
    loss += (weight_blur * loss_sub["blur"])
    # loss += (weight_l1 * (loss_l["l1_rgb"] + loss_r["l1_rgb"]))

    return loss, loss_sub



def solve_render_PnP(render: torch.Tensor, query_crop: torch.Tensor, bboxes, K, R, t, matcher):
    assert render.shape[-1] == 4, "render tensor must have 4 channels (RGB + Depth)"

    xyz_obj, _ = depth_tensor_to_xyz_map(render[..., 3],
                                             fx = K[0, 0], fy = K[1, 1], cx = K[0, 2], cy = K[1, 2],
                                             R = R, t = t)
        
    render_l_f = render_to_image(render[..., :3])
    rgb = crop_with_bbox(render_l_f, bboxes[0])
    matching_results = compute_matches( rgb, query_crop, matcher ) # np
    pts_r, pts_q, conf = unmap_inlier_matches( matching_results, bboxes )
    # match_img = draw_crop_matches(rgb, query_l["m_crop"], render_l["bbox"], query_l["bbox"], pts0, pts1, conf, None)    
    
    R1, t1, _, _, _, _ = get_initial_pose(
                t2np(xyz_obj).astype(np.float64), pts_r, pts_q, conf, 
                t2np(K).astype(np.float64), t2np(R).astype(np.float64), 2.0
            )
    return R1, t1


def solve_renders_PnP(renders: torch.Tensor, viewmats: torch.Tensor, query_crop: np.ndarray, bboxes, 
                      K: np.ndarray, matcher):
    assert renders.shape[-1] == 4, "render tensor must have 4 channels (RGB + Depth)"
    assert renders.shape[0] == viewmats.shape[0], "renders and viewmats must have the same number of views"

    num_views = renders.shape[0]    

    rvecs = np.zeros((num_views, 3), dtype=np.float64)
    tvecs = np.zeros((num_views, 3), dtype=np.float64)

    for iv in range(num_views):
        R, t = T2Rt(viewmats[iv])
        render = renders[iv]
        
        render_crop = crop_with_bbox(render_to_image(render[..., :3]), bboxes[0])
        matching_results = compute_matches( render_crop, query_crop, matcher ) # np
        
        pts_r, pts_q, conf = unmap_inlier_matches( matching_results, bboxes )  

        xyz_obj1 = depth_sample_to_xyz(t2np(render[...,3]), pts_r, K, t2np(R), t2np(t))
        valid_pts_idx = is_valid_point3d(xyz_obj1)
        xyz_obj1 = xyz_obj1[valid_pts_idx]
        pts_r = pts_r[valid_pts_idx]
        pts_q = pts_q[valid_pts_idx]
        conf = conf[valid_pts_idx]

        R1, t1, rep_err, pnp_inlier_idx = solve_pose_pnp(
            pts_q, xyz_obj1.astype(np.float64), K.astype(np.float64), t2np(R).astype(np.float64), 2.0
        )

        rvecs[iv] = cv2.Rodrigues(R1)[0].reshape(-1)
        tvecs[iv] = t1

    return rvecs, tvecs, len(conf)


def refine_pose_stereo_PnP(query_l_info, query_r_info, 
                           gProxy:GaussianRenderer, R_lr, t_lr, matcher):
    assert gProxy.perturbation, "refine_pose_PnP requires gProxy.perturbation=True"

    # query bounding box 
    q_l_bbox = np.array(query_l_info["bbox"])
    q_r_bbox = np.array(query_r_info["bbox"])
    sy = min(q_l_bbox[1], q_r_bbox[1])
    ey = max(q_l_bbox[3], q_r_bbox[3])
    q_l_bbox[1], q_l_bbox[3] = sy, ey
    q_r_bbox[1], q_r_bbox[3] = sy, ey
    q_side = max(get_bbox_size(q_l_bbox))
    q_r_side = max(get_bbox_size(q_r_bbox))
    q_side = max(q_side, q_r_side)      # the largest side (left & right)
    
    query_l = {"bbox": square_bbox(q_l_bbox, int(q_side * 1.2) )}
    query_r = {"bbox": square_bbox(q_r_bbox, int(q_side * 1.2) )}
    render_l = {"bbox": query_l["bbox"]}
    render_r = {"bbox": query_r["bbox"]}
    
    # crop the mask and masked query image for the refinement step
    # c_ : cropped, m_ : masked
    query_l["c_mask"] = np2t(crop_with_bbox(query_l_info["mask"], query_l["bbox"])) / 255.0    
    query_l["m_crop"] = crop_with_bbox(query_l_info["masked_query"], query_l["bbox"])

    query_r["c_mask"] = np2t(crop_with_bbox(query_r_info["mask"], query_r["bbox"])) / 255.0
    query_r["m_crop"] = crop_with_bbox(query_r_info["masked_query"], query_r["bbox"])
    # qc = npf2img(t2np(query_l["crop"]))
    # qcr = npf2img(t2np(query_r["crop"]))
    # plt.imshow( np.hstack(qc, qcr) )

    # ──────────────────────────────────────────────────────
    # 4. Optimization
    # ──────────────────────────────────────────────────────
    
    K = t2np(gProxy.K_mat.squeeze(0))
    
    viewmats = gProxy.viewmats    
    _r, _, _ = gProxy.render_no_grad(render_mode="RGB+ED")

    nv = 5 if gProxy.perturbation else 1

    # left     
    bboxes = (render_l["bbox"], query_l["bbox"])
    rvecs_1, tvecs_1 = solve_renders_PnP(_r[:nv], viewmats[:nv], query_l["m_crop"], bboxes, K, matcher)    
    # right
    bboxes = (render_r["bbox"], query_r["bbox"])
    rvecs_2, tvecs_2 = solve_renders_PnP(_r[nv:], viewmats[nv:], query_r["m_crop"], bboxes, K, matcher) 

    R_rl = t2np(R_lr.T)
    t_rl = -R_rl @ t2np(t_lr)

    for i in range(rvecs_2.shape[0]):
        R2 = R_rl @ cv2.Rodrigues(rvecs_2[i])[0]
        t2 = R_rl @ t_rl + tvecs_2[i]
        rvecs_2[i] = cv2.Rodrigues(R2)[0].reshape(-1) 
        tvecs_2[i] = t2

    rvec = np.vstack((rvecs_1, rvecs_2)).mean(axis=0)
    tvec = np.vstack((tvecs_1, tvecs_2)).mean(axis=0)
    # rvec = rvecs_2.mean(axis=0)
    # tvec = tvecs_2.mean(axis=0)    
    # rvec = rvecs_2[0]
    # tvec = tvecs_2[0]

    return cv2.Rodrigues(rvec)[0], tvec, query_l["bbox"], query_r["bbox"]


def refine_pose_stereo_PnP_GS(query_l_info, query_r_info, 
                              gProxy:GaussianRenderer, options, R_lr, t_lr, matcher):
    assert gProxy.perturbation == False, "refine_pose_PnP requires gProxy.perturbation=False"

    device = torch.device('cuda')
    
    # query bounding box 
    q_l_bbox = np.array(query_l_info["bbox"])
    q_r_bbox = np.array(query_r_info["bbox"])
    sy = min(q_l_bbox[1], q_r_bbox[1])
    ey = max(q_l_bbox[3], q_r_bbox[3])
    q_l_bbox[1], q_l_bbox[3] = sy, ey
    q_r_bbox[1], q_r_bbox[3] = sy, ey
    q_side = max(get_bbox_size(q_l_bbox))
    q_r_side = max(get_bbox_size(q_r_bbox))
    q_side = max(q_side, q_r_side)      # the largest side (left & right)
    
    query_l = {"bbox": square_bbox(q_l_bbox, int(q_side * 1.2) )}
    query_r = {"bbox": square_bbox(q_r_bbox, int(q_side * 1.2) )}
    render_l = {"bbox": query_l["bbox"]}
    render_r = {"bbox": query_r["bbox"]}
    
    # ndarray
    query_l["m_crop"] = crop_with_bbox(query_l_info["masked_query"], query_l["bbox"])
    query_r["m_crop"] = crop_with_bbox(query_r_info["masked_query"], query_r["bbox"])
    # tensor
    query_l["c_mask"] = np2t(crop_with_bbox(query_l_info["mask"], query_l["bbox"])) / 255.0    
    query_r["c_mask"] = np2t(crop_with_bbox(query_r_info["mask"], query_r["bbox"])) / 255.0
    
    # ──────────────────────────────────────────────────────
    # re-PnP 
    # ──────────────────────────────────────────────────────
    K = t2np(gProxy.K_mat.squeeze(0))
    
    R_rl = t2np(R_lr.T)
    t_rl = -R_rl @ t2np(t_lr)
    
    viewmats = gProxy.viewmats    
    _r, _, _ = gProxy.render_no_grad(render_mode="RGB+ED")

    # left
    # bboxes = (render_l["bbox"], query_l["bbox"])
    # rvec_1, tvec_1 = solve_renders_PnP(_r[:1], viewmats[:1], query_l["m_crop"], bboxes, K, matcher)    
    # right
    bboxes = (render_r["bbox"], query_r["bbox"])
    rvec_2, tvec_2 = solve_renders_PnP(_r[1:], viewmats[1:], query_r["m_crop"], bboxes, K, matcher)    

    rvec_2 = cv2.Rodrigues(R_rl @ cv2.Rodrigues(rvec_2)[0])[0]
    # rvec = (rvec_1 + rvec_2.T) / 2
    # tvec = (tvec_1 + (R_rl @ t_rl + tvec_2)) / 2
    rvec = rvec_2
    tvec = R_rl @ t_rl + tvec_2

    # ──────────────────────────────────────────────────────
    # Optimization
    # ──────────────────────────────────────────────────────
    R0, t0 = np2t(cv2.Rodrigues(rvec)[0]), np2t(tvec[0])
    gProxy.set_T( R0, t0 )
    query_l["m_crop"] = np2t(query_l["m_crop"]) / 255.0
    query_r["m_crop"] = np2t(query_r["m_crop"]) / 255.0

    delta_r = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)    # rodrigues vector for rotation update
    delta_t = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)    # translation update

    optimizer = torch.optim.AdamW([
        {"params": [delta_r], "lr": options["lr_rot"]},
        {"params": [delta_t], "lr": options["lr_trans"]},
    ])

    scheduler = CosineWarmupScheduler(
        optimizer,
        total_steps  = options["iters"],
        warmup_steps = options["warmup_steps"],
        max_lr       = options["lr_rot"],
        min_lr       = options["lr_rot"] * 0.01,
    )

    losses = []
    best_loss  = 1e9
    best_state = { "R": R0.detach(), "t": t0.detach(), "iter": 0 }

    ecc_loss_fn = ECCLoss()

    for it in range(options["iters"]):
        optimizer.zero_grad()

        # so3_exp_map(delta_r) : dR
        R_cur = so3_exp_map(delta_r) @ R0
        # tz clamp: init_t의 ±40% 범위로 제한 (frustum 이탈 방지)
        t0z = float(t0[2].item())
        t_raw = t0 + delta_t
        t_cur = torch.stack( [t_raw[0],  t_raw[1],  torch.clamp(t_raw[2], min=t0z*0.6, max=t0z*1.4)] )

        # full resolution render with identity cam (gsplat)
        gProxy.set_T(R_cur, t_cur)
        _r, _, _ = gProxy.render()        # _r has the left & right rendered images
        render_l_f = _r[0].permute(2,0,1).clamp(0,1)
        render_l["crop"] = crop_chw_with_bbox(render_l_f, render_l["bbox"])

        render_r_f = _r[1].permute(2,0,1).clamp(0,1)
        render_r["crop"] = crop_chw_with_bbox(render_r_f, render_r["bbox"])

        loss_l = {}
        loss_r = {}
        # 1. Silhouette (Mask) Loss - render_crop:chw, query_mask: hw
        render_l["c_mask"] = (render_l["crop"].sum(dim=0) > 0.05).float()
        loss_l["mask"] = dice_loss(render_l["c_mask"], query_l["c_mask"])
        render_l["c_mask"] = erode_binary_tensor(render_l["c_mask"].unsqueeze(0), 3).squeeze(0)
        render_l["ci_mask"] = render_l["c_mask"] * query_l["c_mask"]
        render_l["mi_crop"] = render_l["crop"] * render_l["ci_mask"]        
        
        render_r["c_mask"] = (render_r["crop"].sum(dim=0) > 0.05).float()
        loss_r["mask"] = dice_loss(render_r["c_mask"], query_r["c_mask"])
        render_r["c_mask"] = erode_binary_tensor(render_r["c_mask"].unsqueeze(0), 3).squeeze(0)
        render_r["ci_mask"] = render_r["c_mask"] * query_r["c_mask"]
        render_r["mi_crop"] = render_r["crop"] * render_r["ci_mask"]
        
        # 2. RGB Loss (SSIM, L1)
        loss_ssim = dssim_loss(render_l["mi_crop"].unsqueeze(0), query_l["m_crop"].unsqueeze(0)) 
        loss_ssim += dssim_loss(render_r["mi_crop"].unsqueeze(0), query_r["m_crop"].unsqueeze(0))
        loss_ssim +=  dms_ssim_loss(render_l["mi_crop"].unsqueeze(0), query_l["m_crop"].unsqueeze(0))        
        loss_ssim +=  dms_ssim_loss(render_r["mi_crop"].unsqueeze(0), query_r["m_crop"].unsqueeze(0))
        loss_l1_rgb = F.l1_loss(render_l["mi_crop"], query_l["m_crop"])
        loss_l1_rgb = loss_l1_rgb + F.l1_loss(render_r["mi_crop"], query_r["m_crop"])

        # # 3. Blur Loss
        blur_target = F.avg_pool2d(query_l["m_crop"], kernel_size=5, stride=1, padding=2)
        blur_render = F.avg_pool2d(render_l["mi_crop"], kernel_size=5, stride=1, padding=2)
        loss_l["blur"] = F.l1_loss(blur_render, blur_target)
        blur_r_target = F.avg_pool2d(query_r["m_crop"], kernel_size=5, stride=1, padding=2)
        blur_r_render = F.avg_pool2d(render_r["mi_crop"], kernel_size=5, stride=1, padding=2)
        loss_r["blur"] = F.l1_loss(blur_r_render, blur_r_target)

        loss_l["ecc"] = ecc_loss_fn(render_l["mi_crop"], query_l["m_crop"])
        loss_r["ecc"] = ecc_loss_fn(render_r["mi_crop"], query_r["m_crop"])

        loss_l["grad"] = gradient_matching_loss(render_l["mi_crop"], query_l["m_crop"], render_l["ci_mask"])
        loss_r["grad"] = gradient_matching_loss(render_r["mi_crop"], query_r["m_crop"], render_r["ci_mask"])

        # 4. [핵심] Dynamic Weighting (Coarse-to-Fine)
        # 초반: Blur 중심 (크게 돌리기) / 후반: SSIM 중심 (칼같이 맞추기)
        progress = it / max(1, options["iters"])    # 현재 학습 진행도 (0.0 ~ 1.0)
        weight_blur = 1.0 * (1.0 - progress)        # 1.0에서 0.0으로 서서히 감소
        weight_ssim = 0.1 + 0.9 * progress          # 0.1에서 1.0으로 서서히 증가
        weight_l1   = 2.5                           # 기본 위치 유지를 위해 고정ge
        weight_mask = 1.0                           # 크기 유지를 위해 고정

        loss_l["ecc"] *= 2
        loss_r["ecc"] *= 2 
        loss_l["grad"] *= 4
        loss_r["grad"] *= 4
        loss_l["blur"] *= 5
        loss_r["blur"] *= 5
        loss = (loss_l["ecc"] + loss_r["ecc"]) + (loss_l["grad"] + loss_r["grad"]) + loss_ssim
        loss = loss + (weight_blur * (loss_l["mask"] + loss_r["mask"])) + (weight_l1 * loss_l1_rgb)
        loss = loss + (weight_blur * (loss_l["blur"] + loss_r["blur"]))
        loss.backward()

        torch.nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

        optimizer.step()
        scheduler.step()

        tracking_loss = (loss_l["ecc"] + loss_r["ecc"]) + (loss_l["grad"] + loss_r["grad"]) + loss_ssim
        tracking_loss = tracking_loss + (loss_l["mask"] + loss_r["mask"]) + (weight_l1 * loss_l1_rgb)
        tracking_loss = tracking_loss + (loss_l["blur"] + loss_r["blur"])
        loss_val = float(tracking_loss.item())

        losses.append({"iter": it, "ecc_loss": (loss_l["ecc"] + loss_r["ecc"]).item(), 
                       "grad_loss": (loss_l["grad"] + loss_r["grad"]).item(), 
                       "blur_loss": (loss_l["blur"] + loss_r["blur"]).item(),
                       "mask_loss": (loss_l["mask"] + loss_r["mask"]).item(),
                       "ssim_loss": loss_ssim.item(),
                       "rgb_loss": loss_l1_rgb.item(),
                       "loss": loss_val})
        
        # 이제 loss_val이 아닌 tracking_loss 기준으로 최고를 갱신합니다.
        if loss_val < best_loss:
            best_loss  = loss_val
            best_state = {
                "R": R_cur.detach(),
                "t": t_cur.detach(),
                "iter": it,
                "loss": loss_val,                
                # "track_loss": tracking_loss
            }

        # Early stopping
        if it >= options["early_stop_steps"]:
            loss_vals  = torch.tensor([l["loss"] for l in losses])
            loss_grads = (loss_vals[1:] - loss_vals[:-1]).abs()
            recent_grad = loss_grads[-options["early_stop_steps"]:].mean().item()
            if recent_grad < options["early_stop_thr"]:
                print(f"  [EarlyStop] iter={it}  grad_norm={recent_grad:.2e}")
                break

    # ──────────────────────────────────────────────────────
    # 5. 저장
    # ──────────────────────────────────────────────────────
    best_R = best_state["R"].cpu().numpy().copy()
    best_t = best_state["t"].cpu().numpy().copy()

    return best_R, best_t, best_state["loss"], query_l["bbox"], query_r["bbox"], losses


def Rt_update(R0, t0, delta_r, delta_t):
    # so3_exp_map(delta_r) : dR
    R_cur = so3_exp_map(delta_r) @ R0
    # tz clamp: init_t의 ±40% 범위로 제한 (frustum 이탈 방지)
    t0z = float(t0[2].item())
    t_raw = t0 + delta_t
    t_cur = torch.stack( [t_raw[0],  t_raw[1],  torch.clamp(t_raw[2], min=t0z*0.6, max=t0z*1.4)] )
    return R_cur, t_cur

# (100, 10), (50, 5), (40, 5), (30, 4), (20, 3)
def refine_pose_stereo_PnP_GS_rerender(query_l_info, query_r_info, 
                                       gProxy:GaussianRenderer, options, R_lr, t_lr, matcher):
    assert gProxy.perturbation == False, "refine_pose_PnP requires gProxy.perturbation=False"

    device = torch.device('cuda')
    
    # query bounding box 
    bbox_l, bbox_r = adjust_stereo_bbox(query_l_info["bbox"], query_r_info["bbox"]) 
    query_l,  query_r  = {"bbox": bbox_l}, {"bbox": bbox_r}
    render_l, render_r = {"bbox": bbox_l}, {"bbox": bbox_r}
    
    # ndarray
    query_l["m_crop"] = crop_with_bbox(query_l_info["masked_query"], bbox_l)
    query_r["m_crop"] = crop_with_bbox(query_r_info["masked_query"], bbox_r)
    # tensor
    query_l["c_mask"] = np2t(crop_with_bbox(query_l_info["mask"], bbox_l)) / 255.0    
    query_r["c_mask"] = np2t(crop_with_bbox(query_r_info["mask"], bbox_r)) / 255.0
    
    # ──────────────────────────────────────────────────────
    # re-PnP 
    # ──────────────────────────────────────────────────────
    K = gProxy.K_mat.squeeze(0)
    fx, fy = t2np(K.diag()[:2])
    cx, cy = t2np(K[:2, 2])

    R_rl = t2np(R_lr.T)
    t_rl = -R_rl @ t2np(t_lr)
    
    RE_PNP = True
    if RE_PNP:
        viewmats = gProxy.viewmats
        _r, _, _ = gProxy.render_no_grad(render_mode="RGB+ED")

        # left
        # bboxes = (render_l["bbox"], query_l["bbox"])
        # rvec_1, tvec_1 = solve_renders_PnP(_r[:1], viewmats[:1], query_l["m_crop"], bboxes, K, matcher)    
        # right
        bboxes = (render_r["bbox"], query_r["bbox"])
        rvec_2, tvec_2 = solve_renders_PnP(_r[1:], viewmats[1:], query_r["m_crop"], bboxes, t2np(K), matcher)    

        rvec_2 = cv2.Rodrigues(R_rl @ cv2.Rodrigues(rvec_2)[0])[0]
        # rvec = (rvec_1 + rvec_2.T) / 2
        # tvec = (tvec_1 + (R_rl @ t_rl + tvec_2)) / 2
        rvec = rvec_2
        tvec = R_rl @ t_rl + tvec_2

        R0, t0 = np2t(cv2.Rodrigues(rvec)[0]), np2t(tvec[0])
        gProxy.set_T( R0, t0 )
    else:
        R0, t0 = gProxy.get_T()

    # ──────────────────────────────────────────────────────
    # Optimization
    # ──────────────────────────────────────────────────────
    query_l["m_crop"] = np2t(query_l["m_crop"]) / 255.0
    query_r["m_crop"] = np2t(query_r["m_crop"]) / 255.0

    delta_r = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)    # rodrigues vector for rotation update
    delta_t = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)    # translation update

    optimizer = torch.optim.AdamW([
        {"params": [delta_r], "lr": options["lr_rot"]},
        {"params": [delta_t], "lr": options["lr_trans"]},
    ])

    scheduler = CosineWarmupScheduler(
        optimizer,
        total_steps  = options["iters"],
        warmup_steps = options["warmup_steps"],
        max_lr       = options["lr_rot"],
        min_lr       = options["lr_rot"] * 0.01,
    )

    losses = []
    best_loss  = 1e9
    best_state = { "R": R0.detach(), "t": t0.detach(), "iter": 0 }

    it_rerender = 9
    it_GS = options["iters"]
    it_GS = max(1, int( it_GS/(it_rerender+1) ))

    for it in range(it_GS):
        optimizer.zero_grad()

        R_cur, t_cur = Rt_update(R0, t0, delta_r, delta_t)
        
        # full resolution render with identity cam (gsplat)
        gProxy.set_T(R_cur, t_cur)
        _r, _alpha, _ = gProxy.render(render_mode="RGB+ED")      
        
        render_l_chw = _r[0][..., :3].permute(2,0,1).clamp(0.0, 1.0)
        depth_l_hw = _r[0][..., 3]
        render_r_chw = _r[1][..., :3].permute(2,0,1).clamp(0.0, 1.0)
        depth_r_hw = _r[1][..., 3]    
        
        # _r has the left & right rendered images    
        # _alpha > 0.5 produces more accurate object masks
        render_l["crop"] = crop_chw_with_bbox(render_l_chw, render_l["bbox"])
        render_r["crop"] = crop_chw_with_bbox(render_r_chw, render_r["bbox"])
        render_l["c_mask"] = crop_with_bbox((_alpha[0].squeeze() > 0.5).float(), render_l["bbox"])
        render_r["c_mask"] = crop_with_bbox((_alpha[1].squeeze() > 0.5).float(), render_r["bbox"])
        # eroded cropped mask
        render_l["ec_mask"] = erode_binary_tensor(render_l["c_mask"].unsqueeze(0), 3).squeeze(0)
        render_r["ec_mask"] = erode_binary_tensor(render_r["c_mask"].unsqueeze(0), 3).squeeze(0)
        # intersecion cropped mask
        render_l["cif_mask"] = render_l["ec_mask"] * query_l["c_mask"]
        render_r["cif_mask"] = render_r["ec_mask"] * query_r["c_mask"]
        # intersection masked cropped image
        render_l["mi_crop"] = render_l["crop"] * render_l["cif_mask"]        
        render_r["mi_crop"] = render_r["crop"] * render_r["cif_mask"]
        render_l["ci_mask"] = render_l["cif_mask"].bool()       
        render_r["ci_mask"] = render_r["cif_mask"].bool()

        # 4. [핵심] Dynamic Weighting (Coarse-to-Fine)
        # 초반: Blur 중심 (크게 돌리기) / 후반: SSIM 중심 (칼같이 맞추기)
        progress = it / it_GS                   # 현재 학습 진행도 (0.0 ~ 1.0)
        weight_blur = 1.0 * (1.0 - progress)    # 1.0에서 0.0으로 서서히 감소        
        weight_l1   = 2.5

        loss, loss_sub = calc_losses_GS(
            [render_l, render_r], [query_l, query_r], weight_blur
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

        optimizer.step()
        scheduler.step()

        tracking_loss = loss_sub["ecc"] + loss_sub["grad"] + loss_sub["ssim"]
        tracking_loss += loss_sub["mask"] # + (weight_l1 * loss_sub["l1_rgb"])
        tracking_loss += loss_sub["blur"]
        loss_val = float(tracking_loss.item())

        losses.append({"iter": it * (it_rerender+1), "ecc_loss": loss_sub["ecc"].item(), 
                       "grad_loss": loss_sub["grad"].item(), 
                       "blur_loss": loss_sub["blur"].item(),
                       "mask_loss": loss_sub["mask"].item(),
                       "ssim_loss": loss_sub["ssim"].item(),
                    #    "rgb_loss": loss_sub["l1_rgb"].item(),
                       "loss": loss_val})
        
        # 이제 loss_val이 아닌 tracking_loss 기준으로 최고를 갱신합니다.
        if loss_val < best_loss:
            best_loss  = loss_val
            best_state = {
                "R": R_cur.detach(),
                "t": t_cur.detach(),
                "iter": it,
                "loss": loss_val,                
                # "track_loss": tracking_loss
            }

        # Early stopping removed
        
        # rerender
        R1, t1 = Rt_update(R0, t0, delta_r, delta_t)
        R1_r = R_lr @ R1
        t1_r = t1 @ R_lr.T + t_lr 

        gProxy.set_T(R1, t1)
        _r, _alpha, _ = gProxy.render_no_grad(render_mode="RGB+ED")   

        render_l_chw = _r[0][..., :3].permute(2,0,1).clamp(0.0, 1.0)
        depth_l_hw = _r[0][..., 3]
        render_r_chw = _r[1][..., :3].permute(2,0,1).clamp(0.0, 1.0)
        depth_r_hw = _r[1][..., 3]    

        xyz_obj_l, _ = depth_tensor_to_xyz_map(depth_l_hw, R = R1, t = t1,
                                                fx = fx, fy = fy, cx = cx, cy = cy)
        xyz_obj_r, _ = depth_tensor_to_xyz_map(depth_r_hw, R = R1_r, t = t1_r,
                                                fx = fx, fy = fy, cx = cx, cy = cy)
        
        render_l["crop0"] = crop_with_bbox(render_l_chw, render_l["bbox"])
        render_l["c_mask"] = crop_with_bbox(_alpha[0].squeeze() > 0.5, render_l["bbox"])
        render_l["ci_mask"] = render_l["c_mask"] & query_l["c_mask"].bool()
        render_l["cif_mask"] = render_l["ci_mask"].float()

        render_r["crop0"] = crop_with_bbox(render_r_chw, render_r["bbox"])
        render_r["c_mask"] = crop_with_bbox(_alpha[1].squeeze() > 0.5, render_r["bbox"])
        render_r["ci_mask"] = render_r["c_mask"] & query_r["c_mask"].bool()
        render_r["cif_mask"] = render_r["ci_mask"].float()

        # xyz, xyz_colors : tensors
        xyz_l = crop_with_bbox(xyz_obj_l, render_l["bbox"])
        xyz_l = xyz_l[:, render_l["ci_mask"]].detach()                       # 물체 픽셀별 xyz
        xyz_colors_l = render_l["crop0"][:, render_l["ci_mask"]].detach()    # 물체 픽셀별 color
        
        xyz_r = crop_with_bbox(xyz_obj_r, render_r["bbox"])
        xyz_r = xyz_r[:, render_r["ci_mask"]].detach()
        xyz_colors_r = render_r["crop0"][:, render_r["ci_mask"]].detach()

        best_loss_in  = 1e9
        for itr in range(it_rerender):
            # global iteration
            itg = it * (it_rerender+1) + itr + 1

            optimizer.zero_grad()
            
            R_cur, t_cur = Rt_update(R0, t0, delta_r, delta_t)
            R_r_cur = R_lr @ R_cur
            t_r_cur = t_cur @ R_lr.T + t_lr 

            pts2d_l = project_object_points(xyz_l, R_cur, t_cur, K, render_l["bbox"])   
            sampled_colors_l = F.grid_sample(query_l["m_crop"].unsqueeze(0), pts2d_l.unsqueeze(0).unsqueeze(0),
                                        mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled_colors_l = sampled_colors_l.squeeze(0).squeeze(1)

            pts2d_r = project_object_points(xyz_r, R_r_cur, t_r_cur, K, render_r["bbox"])
            sampled_colors_r = F.grid_sample(query_r["m_crop"].unsqueeze(0), pts2d_r.unsqueeze(0).unsqueeze(0),
                                        mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled_colors_r = sampled_colors_r.squeeze(0).squeeze(1)
            
            query_l["rm_crop"] = torch.zeros_like(query_l["m_crop"])  # 만약에 query 색으로 재구성된 query. Rt가 잘 맞는다면, query에서 뽑혀진 sampled_colors가 query에 잘 그려짐
            query_l["rm_crop"][:, render_l["ci_mask"]] = sampled_colors_l
            query_r["rm_crop"] = torch.zeros_like(query_r["m_crop"])
            query_r["rm_crop"][:, render_r["ci_mask"]] = sampled_colors_r

            # 초반: Blur 중심 (크게 돌리기) / 후반: SSIM 중심 (칼같이 맞추기)
            progress = itg / options["iters"]  # 현재 학습 진행도 (0.0 ~ 1.0)
            weight_blur = 1.0 * (1.0 - progress)                        # 1.0에서 0.0으로 서서히 감소
            weight_l1   = 2.5

            total_loss, loss_l, loss_r = calc_losses_rerender(
                [render_l["crop0"], render_r["crop0"]], [query_l["rm_crop"], query_r["rm_crop"]],
                [xyz_colors_l, xyz_colors_r], [sampled_colors_l, sampled_colors_r], 
                [render_l["cif_mask"], render_r["cif_mask"]], weight_blur
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

            optimizer.step()
            scheduler.step()

            tracking_loss = loss_l["ecc"] + loss_r["ecc"] + loss_l["grad"] + loss_r["grad"] + loss_l["ssim"] + loss_r["ssim"] 
            tracking_loss = tracking_loss + (weight_l1 * (loss_l["rgb"]+loss_r["rgb"]))
            tracking_loss = tracking_loss + (loss_l["blur"]+loss_r["blur"])
            loss_val = float(tracking_loss.item())

            losses.append({"iter": itg, "ecc_loss": (loss_l["ecc"] + loss_r["ecc"]).item(), 
                        "grad_loss": (loss_l["grad"] + loss_r["grad"]).item(), 
                        "blur_loss": (loss_l["blur"] + loss_r["blur"]).item(),                       
                        "ssim_loss": (loss_l["ssim"] + loss_r["ssim"]).item(),
                        "rgb_loss": (loss_l["rgb"]+loss_r["rgb"]).item(),
                        "loss": loss_val})
            
            # 이제 loss_val이 아닌 tracking_loss 기준으로 최고를 갱신합니다.
            if loss_val < best_loss_in:
                best_loss_in  = loss_val
                best_state = {
                    "R": R_cur.detach(),
                    "t": t_cur.detach(),
                    "iter": itg,
                    "loss": loss_val,                
                    # "track_loss": tracking_loss
                }

    # ──────────────────────────────────────────────────────
    # 5. 저장
    # ──────────────────────────────────────────────────────
    best_R = best_state["R"].cpu().numpy().copy()
    best_t = best_state["t"].cpu().numpy().copy()

    return best_R, best_t, best_state["loss"], query_l["bbox"], query_r["bbox"], losses


def refine_pose_stereo_PnP_GS_rerender1(query_l_info, query_r_info, which_query,
                                       gProxy:GaussianRenderer, options, R_lr, t_lr, matcher):
    assert gProxy.perturbation == False, "refine_pose_PnP requires gProxy.perturbation=False"

    device = torch.device('cuda')
    
    # query bounding box 
    bbox_l, bbox_r = adjust_stereo_bbox(query_l_info["bbox"], query_r_info["bbox"]) 
    query_l,  query_r  = {"bbox": bbox_l}, {"bbox": bbox_r}
    render_l, render_r = {"bbox": bbox_l}, {"bbox": bbox_r}
    
    # ndarray
    query_l["m_crop"] = crop_with_bbox(query_l_info["masked_query"], bbox_l)
    query_r["m_crop"] = crop_with_bbox(query_r_info["masked_query"], bbox_r)
    # query_l["m_crop"] = cv2.detailEnhance(query_l["m_crop"], sigma_s=10, sigma_r=0.15) 
    # query_r["m_crop"] = cv2.detailEnhance(query_r["m_crop"], sigma_s=10, sigma_r=0.15) 
    # tensor
    query_l["c_mask"] = np2t(crop_with_bbox(query_l_info["mask"], bbox_l)) / 255.0    
    query_r["c_mask"] = np2t(crop_with_bbox(query_r_info["mask"], bbox_r)) / 255.0
    
    # ──────────────────────────────────────────────────────
    # re-PnP 
    # ──────────────────────────────────────────────────────
    K = gProxy.K_mat.squeeze(0)
    fx, fy = t2np(K.diag()[:2])
    cx, cy = t2np(K[:2, 2])

    R_rl = t2np(R_lr.T)
    t_rl = -R_rl @ t2np(t_lr)
    
    RE_PNP = True
    if RE_PNP:
        viewmats = gProxy.viewmats
        _r, _, _ = gProxy.render_no_grad(render_mode="RGB+ED")

        if which_query == 0:
            # left
            bboxes = (render_l["bbox"], query_l["bbox"])
            rvec_1, tvec_1, n_p1 = solve_renders_PnP(_r[:1], viewmats[:1], query_l["m_crop"], bboxes, t2np(K), matcher)    
            rvec = rvec_1
            tvec = tvec_1
        else:
            # right
            bboxes = (render_r["bbox"], query_r["bbox"])
            rvec_2, tvec_2, n_p2 = solve_renders_PnP(_r[1:], viewmats[1:], query_r["m_crop"], bboxes, t2np(K), matcher)    
            rvec_2 = cv2.Rodrigues(R_rl @ cv2.Rodrigues(rvec_2)[0])[0]
            # rvec = (rvec_1 + rvec_2.T) / 2
            # tvec = (tvec_1 + (R_rl @ t_rl + tvec_2)) / 2
            rvec = rvec_2
            tvec = R_rl @ t_rl + tvec_2

        R0, t0 = np2t(cv2.Rodrigues(rvec)[0]), np2t(tvec[0])
        gProxy.set_T( R0, t0 )
    else:
        R0, t0 = gProxy.get_T()

    # ──────────────────────────────────────────────────────
    # Optimization
    # ──────────────────────────────────────────────────────
    query_l["m_crop"] = np2t(query_l["m_crop"]) / 255.0
    query_r["m_crop"] = np2t(query_r["m_crop"]) / 255.0

    delta_r = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)    # rodrigues vector for rotation update
    delta_t = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)    # translation update

    optimizer = torch.optim.AdamW([
        {"params": [delta_r], "lr": options["lr_rot"]},
        {"params": [delta_t], "lr": options["lr_trans"]},
    ])

    scheduler = CosineWarmupScheduler(
        optimizer,
        total_steps  = options["iters"],
        warmup_steps = options["warmup_steps"],
        max_lr       = options["lr_rot"],
        min_lr       = options["lr_rot"] * 0.01,
    )

    losses = []
    best_loss  = 1e9
    best_state = { "R": R0.detach(), "t": t0.detach(), "iter": 0 }

    it_rerender = 9
    it_GS = max(1, int(options["iters"]/(it_rerender+1)))

    for it in range(it_GS):
        optimizer.zero_grad()

        R_cur, t_cur = Rt_update(R0, t0, delta_r, delta_t)
        
        # full resolution render with identity cam (gsplat)
        gProxy.set_T(R_cur, t_cur)
        _r, _alpha, _ = gProxy.render(render_mode="RGB")      
        
        render_l_chw = _r[0].permute(2,0,1).clamp(0.0, 1.0)
        render_r_chw = _r[1].permute(2,0,1).clamp(0.0, 1.0)
        
        # _r has the left & right rendered images    
        # _alpha > 0.5 produces more accurate object masks
        render_l["crop"] = crop_chw_with_bbox(render_l_chw, render_l["bbox"])
        render_r["crop"] = crop_chw_with_bbox(render_r_chw, render_r["bbox"])
        render_l["c_mask"] = crop_with_bbox((_alpha[0].squeeze() > 0.5).float(), render_l["bbox"])
        render_r["c_mask"] = crop_with_bbox((_alpha[1].squeeze() > 0.5).float(), render_r["bbox"])
        # eroded cropped mask for gradient matching (use inner edges not outer boundaries (occlusion issue))
        render_l["ec_mask"] = erode_binary_tensor(render_l["c_mask"].unsqueeze(0), 3).squeeze(0)
        render_r["ec_mask"] = erode_binary_tensor(render_r["c_mask"].unsqueeze(0), 3).squeeze(0)
        # intersecion cropped mask
        render_l["cif_mask"] = render_l["ec_mask"] * query_l["c_mask"]
        render_r["cif_mask"] = render_r["ec_mask"] * query_r["c_mask"]
        # intersection masked cropped image
        render_l["mi_crop"] = render_l["crop"] * render_l["cif_mask"]        
        render_r["mi_crop"] = render_r["crop"] * render_r["cif_mask"]
        
        # 4. [핵심] Dynamic Weighting (Coarse-to-Fine)
        # 초반: Blur 중심 (크게 돌리기) / 후반: SSIM 중심 (칼같이 맞추기)
        progress = it / it_GS                   # 현재 학습 진행도 (0.0 ~ 1.0)
        weight_blur = 1.0 * (1.0 - progress)    # 1.0에서 0.0으로 서서히 감소        
        weight_l1   = 2.5

        loss, loss_sub = calc_losses_GS(
            [render_l, render_r], [query_l, query_r], weight_blur
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

        optimizer.step()
        scheduler.step()

        tracking_loss = loss_sub["ecc"]
        tracking_loss += loss_sub["ssim"]
        tracking_loss += loss_sub["grad"]
        tracking_loss += loss_sub["mask"]
        tracking_loss += loss_sub["blur"]
        # tracking_loss += (weight_l1 * loss_sub["l1_rgb"])
        loss_val = float(tracking_loss.item())

        losses.append({"iter": it * (it_rerender+1), 
                       "ecc_loss": loss_sub["ecc"].item(), 
                       "ssim_loss": loss_sub["ssim"].item(),
                       "grad_loss": loss_sub["grad"].item(), 
                       "blur_loss": loss_sub["blur"].item(),
                       "mask_loss": loss_sub["mask"].item(),                    
                    #    "rgb_loss": loss_sub["l1_rgb"].item(),
                       "loss": loss_val})
        
        # 이제 loss_val이 아닌 tracking_loss 기준으로 최고를 갱신합니다.
        if loss_val < best_loss:
            best_loss  = loss_val
            best_state = {
                "R": R_cur.detach(),
                "t": t_cur.detach(),
                "iter": it,
                "loss": loss_val,                
                # "track_loss": tracking_loss
            }

        # Early stopping removed
        
        # rerender
        R1, t1 = Rt_update(R0, t0, delta_r, delta_t)
        R1_r = R_lr @ R1
        t1_r = t1 @ R_lr.T + t_lr 

        gProxy.set_T(R1, t1)
        _r, _alpha, _ = gProxy.render_no_grad(render_mode="RGB+ED")   

        render_l_chw = _r[0][..., :3].permute(2,0,1).clamp(0.0, 1.0)
        render_r_chw = _r[1][..., :3].permute(2,0,1).clamp(0.0, 1.0)
        depth_l_hw = crop_with_bbox(_r[0][..., 3], render_l["bbox"])
        depth_r_hw = crop_with_bbox(_r[1][..., 3], render_r["bbox"])    

        render_l["crop0"] = crop_with_bbox(render_l_chw, render_l["bbox"])
        render_l["c_mask"] = crop_with_bbox(_alpha[0].squeeze() > 0.5, render_l["bbox"])
        render_l["ci_mask"] = render_l["c_mask"] & query_l["c_mask"].bool()
        render_l["cif_mask"] = render_l["ci_mask"].float()

        render_r["crop0"] = crop_with_bbox(render_r_chw, render_r["bbox"])
        render_r["c_mask"] = crop_with_bbox(_alpha[1].squeeze() > 0.5, render_r["bbox"])
        render_r["ci_mask"] = render_r["c_mask"] & query_r["c_mask"].bool()
        render_r["cif_mask"] = render_r["ci_mask"].float()

        # xyz, xyz_colors : tensors
        # 물체 픽셀별 xyz
        xyz_l = depth_tensor_to_xyz_map2(depth_l_hw, render_l["ci_mask"], 
                                        render_l["bbox"][0], render_l["bbox"][1], fx, fy, cx, cy, R1, t1).detach()
        xyz_r = depth_tensor_to_xyz_map2(depth_r_hw, render_r["ci_mask"], 
                                        render_r["bbox"][0], render_r["bbox"][1], fx, fy, cx, cy, R1_r, t1_r).detach()
        
        # 물체 픽셀별 color
        xyz_colors_l = render_l["crop0"][:, render_l["ci_mask"]].detach()    
        xyz_colors_r = render_r["crop0"][:, render_r["ci_mask"]].detach()

        best_loss_in  = 1e9
        for itr in range(it_rerender):
            # global iteration
            itg = it * (it_rerender+1) + itr + 1

            optimizer.zero_grad()
            
            R_cur, t_cur = Rt_update(R0, t0, delta_r, delta_t)
            R_r_cur = R_lr @ R_cur
            t_r_cur = t_cur @ R_lr.T + t_lr 

            pts2d_l = project_object_points(xyz_l, R_cur, t_cur, K, render_l["bbox"])   
            sampled_colors_l = F.grid_sample(query_l["m_crop"].unsqueeze(0), pts2d_l.unsqueeze(0).unsqueeze(0),
                                        mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled_colors_l = sampled_colors_l.squeeze(0).squeeze(1)

            pts2d_r = project_object_points(xyz_r, R_r_cur, t_r_cur, K, render_r["bbox"])
            sampled_colors_r = F.grid_sample(query_r["m_crop"].unsqueeze(0), pts2d_r.unsqueeze(0).unsqueeze(0),
                                        mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled_colors_r = sampled_colors_r.squeeze(0).squeeze(1)
            
            query_l["rm_crop"] = torch.zeros_like(query_l["m_crop"])  # 만약에 query 색으로 재구성된 query. Rt가 잘 맞는다면, query에서 뽑혀진 sampled_colors가 query에 잘 그려짐
            query_l["rm_crop"][:, render_l["ci_mask"]] = sampled_colors_l
            query_r["rm_crop"] = torch.zeros_like(query_r["m_crop"])
            query_r["rm_crop"][:, render_r["ci_mask"]] = sampled_colors_r

            # 초반: Blur 중심 (크게 돌리기) / 후반: SSIM 중심 (칼같이 맞추기)
            progress = itg / options["iters"]  # 현재 학습 진행도 (0.0 ~ 1.0)
            weight_blur = 1.0 * (1.0 - progress)                        # 1.0에서 0.0으로 서서히 감소
            weight_l1   = 2.5

            total_loss, loss_sub = calc_losses_rerender(
                torch.stack((render_l["crop0"], render_r["crop0"])), 
                torch.stack((query_l["rm_crop"], query_r["rm_crop"])),
                [xyz_colors_l, xyz_colors_r], [sampled_colors_l, sampled_colors_r], 
                torch.stack((render_l["cif_mask"], render_r["cif_mask"])).unsqueeze(1), 
                weight_blur
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

            optimizer.step()
            scheduler.step()

            tracking_loss = loss_sub["ecc"]
            tracking_loss += loss_sub["ssim"]
            tracking_loss += loss_sub["grad"]            
            tracking_loss += loss_sub["blur"]
            # tracking_loss += (weight_l1 * (loss_l["rgb"]+loss_r["rgb"]))
            loss_val = float(tracking_loss.item())

            losses.append({"iter": itg, 
                           "ecc_loss": loss_sub["ecc"].item(), 
                        "grad_loss": loss_sub["grad"].item(), 
                        "blur_loss": loss_sub["blur"].item(),                       
                        "ssim_loss": loss_sub["ssim"].item(),
                        # "rgb_loss": loss_sub["rgb"].item(),
                        "loss": loss_val})
            
            # print(losses[-1])
             
            # 이제 loss_val이 아닌 tracking_loss 기준으로 최고를 갱신합니다.
            if loss_val < best_loss_in:
                best_loss_in  = loss_val
                best_state = {
                    "R": R_cur.detach(),
                    "t": t_cur.detach(),
                    "iter": itg,
                    "loss": loss_val,                
                    # "track_loss": tracking_loss
                }

    # ──────────────────────────────────────────────────────
    # 5. 저장
    # ──────────────────────────────────────────────────────
    best_R = best_state["R"].cpu().numpy().copy()
    best_t = best_state["t"].cpu().numpy().copy()

    return best_R, best_t, best_state["loss"], (query_l, query_r), losses


def estimate_object_pose(query, gallery_info, nets, gProxy:GaussianRenderer, K, pnp_option, render_options):
    total_t0 = sync_time()
    query_bgr = cv2.cvtColor(query, cv2.COLOR_RGB2BGR)
    q_mask = detect_segment(query_bgr, nets[:2])
    q_bbox = compute_bbox(q_mask)  
    t_step1 = sync_time()

    query_info = {
        "rgb": query,    # full query image (RGB)
        "mask": q_mask,   # full query mask (grayscale, 0=background, 255=foreground)
        "masked_query": apply_mask(query, q_mask),
        "bbox": q_bbox  
    }

    # Best gallery selection - Feature matching - PnP
    R0, t0, _ = get_T0(query_info, gallery_info, nets, K, pnp_option["reproj_thr"])
    t_step56 = sync_time()

    # Render & Compare
    gProxy.set_T(R0, t0)
    # R, t, t_loss, bbox, losses = refine_pose_rerender(query_info, gProxy, render_options)
    R, t, t_loss, bbox = refine_pose_GS(query_info, gProxy, render_options)
    t_refine = sync_time()

    # Summary print
    print(f"R: {R} \nt: {t*100}(cm) \ntracking loss: {t_loss}")
    print(f"Timing: detection+segmentation={t_step1 - total_t0:.3f}s, retrieval+matching+PnP={t_step56 - t_step1:.3f}s, refinement={t_refine - t_step56:.3f}s")
    print(f"refinement update: Δr={cv2.Rodrigues(R)[0].T - cv2.Rodrigues(R0)[0].T}, Δt={(t - t0)*100}(cm)")

    return R, t, R0, t0, bbox, t_loss, losses


def estimate_object_pose_stereo(query, query_r, gallery_info, nets, 
                                gProxy:GaussianRenderer, 
                                K, R_lr, t_lr, pnp_option, render_options):
    time_ = list()
    

    time_.append( sync_time() )
    query_bgr = cv2.cvtColor(query, cv2.COLOR_RGB2BGR)
    q_mask = detect_segment(query_bgr, nets[:2])
    q_bbox = compute_bbox(q_mask)  
    q_mask = q_mask & ~get_specular_mask(query)

    query_r_bgr = cv2.cvtColor(query_r, cv2.COLOR_RGB2BGR)
    q_r_mask = detect_segment(query_r_bgr, nets[:2])
    q_r_bbox = compute_bbox(q_r_mask) 
    q_r_mask = q_r_mask & ~get_specular_mask(query_r)
    time_.append( sync_time() )

    query_info = {
        "rgb": query,    # full query image (RGB)
        "mask": q_mask,   # full query mask (grayscale, 0=background, 255=foreground)
        "masked_query": apply_mask(query, q_mask),
        "bbox": q_bbox  
    }

    query_r_info = {
        "rgb": query_r,    # full query image (RGB)
        "mask": q_r_mask,   # full query mask (grayscale, 0=background, 255=foreground)
        "masked_query": apply_mask(query_r, q_r_mask),
        "bbox": q_r_bbox  
    }

    # Best gallery selection - Feature matching - PnP
    R0, t0, which_query = get_T0_stereo([query_info, query_r_info], gallery_info, nets, K, pnp_option["reproj_thr"])
    if which_query == 1:
        R0 = t2np(R_lr.T) @ R0
        t0 = (t0 - t2np(t_lr)) @ t2np(R_lr)

    time_.append( sync_time() )

    METHOD = 'GS' 
    METHOD = 'RERENDER'
    # METHOD = 'MIXED' 
    # Render & Compare
    gProxy.set_T(R0, t0)
    if METHOD == 'GS':
        R, t, t_loss, queries, losses = refine_pose_stereo_GS(query_info, query_r_info, 
                                                              gProxy, render_options, R_lr, t_lr)
    elif METHOD == 'RERENDER':
        R, t, t_loss, queries, losses = refine_pose_stereo_rerender(query_info, query_r_info, 
                                                                         gProxy, render_options, R_lr, t_lr)    
    elif METHOD == 'MIXED':
        R, t, t_loss, queries, losses = refine_pose_stereo_PnP_GS_rerender1(query_info, query_r_info, which_query,
                                                                                gProxy, render_options, R_lr, t_lr, nets[3])
    time_.append( sync_time() )

    # Summary print
    time_proc = np.diff(time_)
    print(f"Timing: detection+segmentation={time_proc[0]:.3f}s, retrieval+matching+PnP={time_proc[1]:.3f}s, refinement={time_proc[2]:.3f}s")
    print(f"refinement update: Δr={cv2.Rodrigues(R)[0].T - cv2.Rodrigues(R0)[0].T}, Δt={(t - t0)*100}(cm)")

    return R, t, R0, t0, queries, losses, time_proc


def main_object_pose_estimation():
    assert torch.cuda.is_available(), "CUDA required."

    device = torch.device('cuda')

    # Setup
    config = load_json("ope_config.json")
    obj_num = config["object"]
    obj_name = config["objects"][obj_num][0]
    model_dir = get_obj_path(obj_name, "model")
    obj_dir = get_obj_path(obj_name, "object")
    obj_out_dir = get_obj_path(obj_name, "output")
    
    # Initialization
    # config["input"] = config["input_femto_bolt"]
    config["input"] = config["input_zed_m"]

    if config["input"]["cam_type"] == "zed_m":
        USE_STEREO = True
    else:
        USE_STEREO = False

    if not USE_STEREO:
        K = load_intrinsics(get_K_path(config["input"]))
    else:
        calib_info = load_json(get_K_path(config["input"]))
        fx = calib_info["left_rect"]["fx"]
        fy = calib_info["left_rect"]["fy"]
        cx = calib_info["left_rect"]["cx"]
        cy = calib_info["left_rect"]["cy"]
        K = params_to_K(fx, fy, cx, cy)

        R_rl = so3_exp_map(torch.tensor(calib_info["R_rodrigues"], dtype=torch.float32, device=device))
        R_lr = R_rl.T
        t_rl = torch.tensor(calib_info["t_meters"], dtype=torch.float32, device=device)
        t_lr = - R_lr @ t_rl

    nets = load_networks(config)
    
    render_options = config["renderer"]["options"]
    gallery_info = construct_galleryInfo(obj_dir)

    gaussian_proxy = GaussianRenderer( 
        init_gaussians(config["renderer"], resolve_ply_path(model_dir), scale=config["objects"][obj_num][1]), 
        K,
        np.array(config["renderer"]["options"]["background"], dtype=np.float32) / 255.0,
        render_options["width"], render_options["height"], False
    )
    gaussian_proxy.set_T_lr(R_lr, t_lr)

    pnp_option = { "reproj_thr": config["pnp"]["reproj_thr"]  }
    
    # Input
    query_paths = get_query_paths(config)
    
    # perform
    performance = []
    pose_results = []
    time_refine = []

    if USE_STEREO:
        assert len(query_paths) % 2 == 0, "For stereo image pairs, the number of query images should be even (left-right pairs)."
        idx = range(0, len(query_paths), 2)
    else:
        idx = range(len(query_paths))
    
    # idx = range(len(query_paths))
    for i in idx:
        # if i == 24:
        #     break
        query = load_rgb(query_paths[i])
        assert render_options["width"] == query.shape[1] and render_options["height"] == query.shape[0]

        if USE_STEREO:
            query_r = load_rgb(query_paths[i+1])

        print("Query file:", query_paths[i])
        # time measurement
        st = time.perf_counter()
        
        if USE_STEREO:
            res = estimate_object_pose_stereo(
                query, query_r, 
                gallery_info, nets, gaussian_proxy, K, R_lr, t_lr, 
                pnp_option, render_options)
            R, t, R0, t0, queries, losses, time_proc = res # , t_loss, losses
            q_bbox = queries[0]["bbox"]
            q_bbox_r = queries[1]["bbox"]
            time_refine.append(time_proc[-1])
        else:
            res = estimate_object_pose(
                query, 
                gallery_info, nets, gaussian_proxy, K, 
                pnp_option, render_options)
            R, t, R0, t0, q_bbox, t_loss, losses = res
        
        et = time.perf_counter()
        # performance.append((et - st, t_loss))
        
        pose_record = {
            "query":            str(query_paths[i].stem),
            "R":                R.tolist(),
            "t":                t.tolist(),
            "R0":               R0.tolist(),
            "t0":               t0.tolist(),
            # "tracking loss":    float(t_loss)            
        }
        pose_results.append(pose_record)

        # Visualization
        fnstem = query_paths[i].stem
        gaussian_proxy.set_T(R, t)
        comp_img = make_comp_image(crop_with_bbox(query, q_bbox), t2np(queries[0]["c_mask"]*255.0).astype(np.uint8), q_bbox, gaussian_proxy, R0, t0)
        vis_path = obj_out_dir / "result" / f"{fnstem}_comp.png"
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_path), cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR))

        gaussian_proxy.set_T(R, t)
        pose_img = make_pose_validation_image(query, gaussian_proxy, q_bbox, 3.0)
        vis_path = obj_out_dir / "result" / f"{fnstem}_pose_valid.png"
        cv2.imwrite(str(vis_path), cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR))

        # np.save( obj_out_dir / "result" / f"{fnstem[:-1]}.npy", losses )

        if USE_STEREO:
            R_r = R_lr.detach().cpu().numpy() @ R
            t_r = t @ R_lr.T.detach().cpu().numpy() + t_lr.detach().cpu().numpy() 

            R_r0 = R_lr.detach().cpu().numpy() @ R0
            t_r0 = t0 @ R_lr.T.detach().cpu().numpy() + t_lr.detach().cpu().numpy() 

            gaussian_proxy.set_T(R_r, t_r)
            comp_img = make_comp_image(crop_with_bbox(query_r, q_bbox_r), t2np(queries[1]["c_mask"]*255.0).astype(np.uint8), q_bbox_r, gaussian_proxy, R_r0, t_r0)
            vis_path = obj_out_dir / "result" / f"{fnstem[:-1]}R_comp.png"
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_path), cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR))

            gaussian_proxy.set_T(R_r, t_r)
            pose_img = make_pose_validation_image(query_r, gaussian_proxy, q_bbox_r, 3.0)
            vis_path = obj_out_dir / "result" / f"{fnstem[:-1]}R_pose_valid.png"
            cv2.imwrite(str(vis_path), cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR))

        # if USE_STEREO:
        #     np.save( obj_out_dir / "result" / f"{fnstem[:-1]}.npy", losses )
        # else:
        #     np.save( obj_out_dir / "result" / f"{fnstem}.npy", losses )

        
    save_json(obj_out_dir / "result" / "refined_poses.json", pose_results)
    
    
    print("\n=== Performance Summary ===")
    print(f"pose refinement estimation time: {np.mean(time_refine[1:]) * 1000:.1f} ms\n\n")
    # print(f"tracking loss: {np.mean([p[1] for p in performance]):.6f}")


if __name__ == "__main__":
    main_object_pose_estimation()



    
    
    

    