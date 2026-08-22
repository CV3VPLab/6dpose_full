import locale
import os
import warnings
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d
from rich import print 
from pathlib import Path
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from modules_6d.yolo_sam       import load_yolo_model, detect_with_yolo, load_sam2_predictor, segment_from_bbox
from modules_6d.retrieval_dino import preprocess_for_dinov2, load_extractor
from modules_6d.retrieval_dino_loftr import compute_loftr_matches
from modules_6d.retrieval_edm import (
    compute_edm_matches,
    compute_edm_trt_matches,
    compute_edm_trt_matches_batch,
    load_edm_model,
    load_edm_trt_session,
    warmup_edm_trt_session
)
from modules_6d.step6_translation import get_initial_pose, get_gallery_pose, solve_pose_pnp

from gaussian_renderer import GaussianModel
from refine_pose import (
    GaussianRenderer, CosineWarmupScheduler, 
    so3_exp_map, crop_chw_with_bbox
)

from utils.general_utils import sync_time
from utils.io_utils import (
    load_json, save_json, 
    load_intrinsics, params_to_K,
    get_obj_path, get_K_path,
    resolve_ply_path ,
    get_named_config
)
from utils.image_utils import (
    load_rgb, render_to_image,
    expand_bbox, compute_bbox, make_same_sized_stereo_bboxes, 
    get_bbox_size, get_bbox_area, square_bbox, 
    crop_with_bbox, square_pad_resize, unmap_to_full_image,     
    make_gallery_square, construct_galleryInfo,
    get_specular_mask, apply_mask, get_mask_inlier_indices,
    erode_binary_tensor,    
    scale_image_draw_maskcontour, draw_contour, imshow_tensor      
)
from utils.image_utils import tensor_to_np as t2np, np_to_tensor as np2t
from utils.geom_utils import (
    depth_tensor_to_xyz_map, depth_tensor_to_xyz_map2, depth_sample_to_xyz, 
    triangulate_stereo, find_nearest_numpy,
    is_valid_point3d,
    T2Rt, Rt_inv_np, Rt_compose_np, mesh_from_depth_grid
)
from utils.loss_utils import (
    ECCLoss,
    dssim_loss, dms_ssim_loss,
    dice_loss, gradient_matching_loss  
)


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

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


######################################################################

FRUSTUM_N = 0.03  # near frustum : 3cm


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
    return load_yolo_model(config["weights"]), config["options"]["conf_thr"], config["options"].get("imgsz", 1280)


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
    ext_config  = get_named_config(config["feat_extractors"])
    extractor   = load_extractor(ext_config)
    matcher     = load_matcher(config["matcher"])

    assert detector    is not None  
    assert segmentator is not None  
    assert extractor   is not None  
    assert matcher     is not None  
    return detector, segmentator, extractor, matcher


def get_query_paths(config):
    config_input = config["input"]
    assert config_input["type"] == "file" or config_input["type"] == "dir"

    obj_name = get_named_config(config["objects"])["name"]
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

    obj_name = get_named_config(config["objects"])["name"]
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
    _, _a, _ = gaussian_proxy.render_no_grad()
    mask_crop = crop_with_bbox(t2np(_a[0].squeeze(2)), q_bbox) > 0.5
    contours, _ = cv2.findContours( mask_crop.astype(np.uint8) * 255, 
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cv2.drawContours(q_crop_t, contours, -1, colors[5], 1)
    cv2.putText(q_crop_t, f"tz {str_t}", (12, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[5], 1)

    return np.hstack((q_crop, q_crop_t))


# processes
def detect_segment(query, nets):
    detector, det_conf_thr,det_imgsz = nets[0]
    segmentator = nets[1]
    
    h, w = query.shape[:2]
    det = detect_with_yolo(detector, cv2.cvtColor(query, cv2.COLOR_RGB2BGR), 
                           conf_thres=det_conf_thr, imgsz=det_imgsz)
    assert det is not None 
    det_conf = det[0]["conf"]
    
    bbox = expand_bbox(det[0]["bbox_xyxy"], 12, w, h)
    mask, seg_score = segment_from_bbox(segmentator, query, bbox)
    
    print(f"detection score: {det_conf:.3f},  segmentation score: {seg_score:.3f}, bounding box: {bbox}") 
    return mask


def detect_segment_stereo(queryList, nets):
    detector, det_conf_thr,det_imgsz = nets[0]
    segmentator = nets[1]
    
    h, w = queryList[0].shape[:2]
    query_bgr_list = [cv2.cvtColor(q, cv2.COLOR_RGB2BGR) for q in queryList]
    det = detect_with_yolo(detector, query_bgr_list, conf_thres=det_conf_thr, imgsz=det_imgsz)
    assert det is not None and len(det) == len(queryList)

    bboxes = [expand_bbox(detEach["bbox_xyxy"], 12, w, h) for detEach in det]    
    cls_ids = [detEach["cls_id"] for detEach in det]

    masks, seg_scores = segment_from_bbox(segmentator, queryList, bboxes)
    
    print(f"detect score: {det[0]['conf']:.3f} {det[1]['conf']:.3f},  segment score: {seg_scores[0]:.3f} {seg_scores[1]:.3f}, bounding box: {bboxes[0]} {bboxes[1]}") 
    return masks, cls_ids


def detect_stereo(queryList, net):
    detector, det_conf_thr,det_imgsz = net
    
    h, w = queryList[0].shape[:2]
    query_bgr_list = [cv2.cvtColor(q, cv2.COLOR_RGB2BGR) for q in queryList]
    det = detect_with_yolo(detector, query_bgr_list, conf_thres=det_conf_thr, imgsz=det_imgsz)

    if det is None or len(det) != len(queryList):
        return None, None

    bboxes = [expand_bbox(detEach["bbox_xyxy"], 12, w, h) for detEach in det]    
    cls_ids = [detEach["cls_id"] for detEach in det]

    print(f"detect score: {det[0]['conf']:.3f} {det[1]['conf']:.3f}, bounding box: {bboxes[0]} {bboxes[1]}")     
    return bboxes, cls_ids


def segment_stereo(queryList, bboxes, segmentator):
    masks, seg_scores = segment_from_bbox(segmentator, queryList, bboxes)
    
    print(f"segment score: {seg_scores[0]:.3f} {seg_scores[1]:.3f}") 
    return masks


# def retrieve_best(queryInfo, galleryInfo, extractor):
#     # KSCHOI TODO: occlusion 상황에서 topk가 의미 있는지 확인 (현재는 DINO feature similarity top-1만 사용함)
#     g_feats = galleryInfo["feats"]
#     g_bbox_size = galleryInfo["bbox_size"]

#     q_bbox = queryInfo["bbox"]
#     # query bounding box can be larger than gallery bbox, so we use the max of both for square cropping
#     q_bbox_size = max( g_bbox_size, q_bbox[2] - q_bbox[0], q_bbox[3] - q_bbox[1] )
#     q_bbox_ext = square_bbox(q_bbox, q_bbox_size)
#     masked_query_crop = crop_with_bbox(queryInfo["masked_query"], q_bbox_ext)    

#     # masked_query_crop = cv2.detailEnhance(masked_query_crop, sigma_s=10, sigma_r=0.15)     

#     # A cropped query image for extracting DINOv2 feature
#     ext_net = extractor[0]
#     ext_opts = extractor[1]
#     dino_size = ext_opts["input_size"]
#     query_dino_in = square_pad_resize(masked_query_crop, dino_size * 2)
#     if g_feats.shape[1] == 384:
#         qfeat = ext_net.encode_rgb(query_dino_in)    
#     elif g_feats.shape[1] == 384 * 4:
#         qfeat = ext_net.encode_4rgb(query_dino_in)
        
#     scores = (g_feats @ qfeat).cpu().numpy()
#     best_item = np.argmax(scores)
    
#     return best_item, scores[best_item], masked_query_crop, q_bbox_ext


def retrieve_topk(queryInfo, galleryInfo, extractor, k=3):    
    g_feats = galleryInfo["feats"]
    g_bbox_size = galleryInfo["bbox_size"]

    # A cropped query image for extracting DINOv2 feature
    ext_net = extractor[0]
    ext_opts = extractor[1]
    dino_size = ext_opts["input_size"]
    query_dino_in = square_pad_resize(queryInfo["m_crop"], dino_size * 2)

    if g_feats.shape[1] == 384:
        qfeat = ext_net.encode_rgb(query_dino_in)    
    elif g_feats.shape[1] == 384 * 4:
        qfeat = ext_net.encode_4rgb(query_dino_in)
        
    scores = (g_feats @ qfeat).cpu().numpy()    
    topk_items = np.argsort(scores)[::-1][:k]    
    
    return topk_items, scores[topk_items]


# Asymmetric Patch-level Chamfer Matching 
def retrieve_APCM_topk(queryInfo, galleryInfo, extractor, k=3):    
    g_feats = galleryInfo["feats"]
    g_bbox_size = galleryInfo["bbox_size"]

    # A cropped query image for extracting DINOv2 feature
    ext_net  = extractor[0]
    ext_opts = extractor[1]
    dino_size = ext_opts["input_size"]
    
    query_dino_in = square_pad_resize(queryInfo["m_crop"], dino_size * 2)
    q_mask        = square_pad_resize(queryInfo["c_mask"], dino_size * 2)

    q_t, qmsk_t = preprocess_for_dinov2(query_dino_in, q_mask)  # Preprocess the image and mask for DINOv2
    q_tokens = ext_net.extract_masked_patch_tokens(q_t.to('cuda'), qmsk_t.to('cuda'))[0]
    q_tokens = F.normalize(q_tokens, p=2, dim=1)

    pca = galleryInfo["pca"]
    query_reduced_np = pca.transform(q_tokens.detach().cpu().numpy())
    query_reduced = torch.from_numpy(query_reduced_np).float().cuda()
    query_reduced = F.normalize(query_reduced, p=2, dim=1)
    
    scores = ext_net.compute_asymmetric_chamfer_similarity(query_reduced, g_feats)        
    topk_values, topk_indices = torch.topk(scores, k=k)

    return t2np(topk_indices), topk_values


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


def compute_best_matches_from_batch(queries, galleries, matcher):
    match_name = matcher["name"]
    match_net = matcher["model"]
    match_opts = matcher["options"]
    in_size = match_opts["input_size"]
    conf_thr = match_opts["conf_thr"]
    
    nPairs = len(queries)
    assert nPairs == len(galleries)

    queries_m   = [square_pad_resize(queries[i],   in_size) for i in range(nPairs)]
    galleries_m = [square_pad_resize(galleries[i], in_size) for i in range(nPairs)]
    
    assert match_name == "EDM_TRT"
    matching_results = compute_edm_trt_matches_batch(
        match_net, 
        queries_m, galleries_m,
        conf_thr=conf_thr
    )    

    # Unmap the matched keypoints back to the original image crop coordinates
    for i in range(nPairs):
        mkpts0, mkpts1 = matching_results[i][0], matching_results[i][1]
        mkpts0 = unmap_square_pad_resize(mkpts0, queries[i].shape[:2],   queries_m[i].shape[:2])
        mkpts1 = unmap_square_pad_resize(mkpts1, galleries[i].shape[:2], galleries_m[i].shape[:2])
        matching_results[i][0], matching_results[i][1] = mkpts0, mkpts1

    return matching_results


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


from utils.image_utils import draw_matches_FHD, draw_matches
def get_T0(query_info, gallery_info, nets, K, reproj_thr):
    best_idx, score, query_crop, q_bbox_ext = retrieve_topk(query_info, gallery_info, nets[2], k=1)
    # prepare the best reference for matcher 
    gallery_crop, g_bbox_ext = make_gallery_square(gallery_info, best_idx, query_crop.shape[0])

    matching_results = compute_matches( query_crop, gallery_crop, nets[3] )
    # matcher input size in the matcher options
    pts0, pts1, conf = unmap_inlier_matches( matching_results, (q_bbox_ext, g_bbox_ext), query_info["mask"])
    # match_img = draw_matches_FHD(gallery_crop, query_crop, g_bbox_ext, q_bbox_ext, pts0, pts1, conf, None)

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


def get_T0_stereo_top1(query_infos, gallery_info, nets, K, reproj_thr):
    best_idx_l, score_l = retrieve_topk(query_infos[0], gallery_info, nets[2], k=1)
    best_idx_r, score_r = retrieve_topk(query_infos[1], gallery_info, nets[2], k=1)
    best_idx_l = best_idx_l[0]
    best_idx_r = best_idx_r[0]
    # prepare the best reference for matcher 
    gallery_crop_l, g_bbox_ext_l = make_gallery_square(gallery_info, best_idx_l, query_infos[0]['m_crop'].shape[0])
    gallery_crop_r, g_bbox_ext_r = make_gallery_square(gallery_info, best_idx_r, query_infos[1]['m_crop'].shape[0])

    matching_results_l = compute_matches( query_infos[0]['m_crop'], gallery_crop_l, nets[3] )
    matching_results_r = compute_matches( query_infos[1]['m_crop'], gallery_crop_r, nets[3] )

    if len(matching_results_l[0]) > len(matching_results_r[0]):
        matching_results = matching_results_l
        q_bbox_ext = query_infos[0]['bbox']
        g_bbox_ext = g_bbox_ext_l
        query_info = query_infos[0]
        best_idx = best_idx_l
        which_query = 0
    else:
        matching_results = matching_results_r
        q_bbox_ext = query_infos[1]['bbox']
        g_bbox_ext = g_bbox_ext_r
        query_info = query_infos[1]
        best_idx = best_idx_r
        which_query = 1

    # matcher input size in the matcher options
    pts0, pts1, conf = unmap_inlier_matches( matching_results, (q_bbox_ext, g_bbox_ext), query_info["mask"])
    # match_img = draw_matches_FHD(gallery_crop, query_crop, g_bbox_ext, q_bbox_ext, pts0, pts1, conf, None)

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


def get_T0_stereo(query_infos, gallery_info, nets, K, Rt_lr, reproj_thr):
    if not hasattr(get_T0_stereo, "R_rl"):
        get_T0_stereo.Rt_rl = Rt_inv_np(Rt_lr)

    best_inds_l, score_l = retrieve_APCM_topk(query_infos[0], gallery_info, nets[2])
    best_inds_r, score_r = retrieve_APCM_topk(query_infos[1], gallery_info, nets[2])
    assert len(best_inds_l) == len(best_inds_r) == 3

    # prepare the references for matcher
    nMatches = 0 
    nMatchList = np.zeros(len(best_inds_l) + len(best_inds_r), dtype=np.int32)
    q_crop_size = query_infos[0]["m_crop"].shape[0]
    best_inds = np.zeros(len(best_inds_l) + len(best_inds_r), dtype=np.int32)
    best_inds[::2] = best_inds_l
    best_inds[1::2] = best_inds_r
    
    gallery_crop, g_bbox = [], []
    query_crop = [query_infos[0]["m_crop"], query_infos[1]["m_crop"],   # best query crops for left and right
                  query_infos[0]["m_crop"], query_infos[1]["m_crop"],   
                  query_infos[0]["m_crop"], query_infos[1]["m_crop"]]

    for i in range(len(best_inds_l)):
        gallery_crop_l, g_bbox_ext_l = make_gallery_square(gallery_info, best_inds_l[i], q_crop_size)
        gallery_crop.append(gallery_crop_l)
        g_bbox.append(g_bbox_ext_l)
        gallery_crop_r, g_bbox_ext_r = make_gallery_square(gallery_info, best_inds_r[i], q_crop_size)    
        gallery_crop.append(gallery_crop_r)
        g_bbox.append(g_bbox_ext_r)

    matching_results = compute_best_matches_from_batch(query_crop[:2], gallery_crop[:2], nets[3])
    nMatches = np.array([len(matching_results[i][0]) for i in range(len(matching_results))])
    
    # 2 galleries : 630 ms, 6 galleries : 690 ms  for USB 4
    if nMatches.max() > 100:
        best_i = nMatches.argmax()
        avg_confs = np.array([matching_results[best_i][2].mean()])
    else:
        matching_results4 = compute_best_matches_from_batch(query_crop[2:], gallery_crop[2:], nets[3])
        matching_results.extend(matching_results4)
        nPairs = len(matching_results)
        nMatches  = np.array([len(matching_results[i][0])   for i in range(nPairs)])
        avg_confs = np.array([matching_results[i][2].mean() for i in range(nPairs)])
        best_i = (nMatches * avg_confs).argmax()

    mkpts0, mkpts1, conf = matching_results[best_i]
    matcher_name = nets[3]['name']
    conf_thr = nets[3]["options"]["conf_thr"]
    idx_clr = 'cyan' if best_i < 2 else 'red'
    print(f"  [{matcher_name}] best matches after conf>={conf_thr}: {len(conf)} points, index [{idx_clr}]{best_i}[/{idx_clr}] in gallery {nMatches}")
    print(f"                 with avg. conf {avg_confs}")

    which_query = 0 if best_i % 2 == 0 else 1
    best_idx = best_inds[best_i]
    
    q_bbox = query_infos[which_query]["bbox"]    

    # matcher input size in the matcher options
    pts0, pts1, conf = unmap_inlier_matches( (mkpts0, mkpts1, conf), 
                                             (q_bbox, g_bbox[best_i]), query_infos[which_query]["mask"] )
    # match_img = draw_matches_FHD(gallery_crop, query_crop, g_bbox_ext, q_bbox_ext, pts0, pts1, conf, None)
    
    # Get initial pose using the matched 2D-3D correspondences and PnP
    R_g, t_g = get_gallery_pose(gallery_info["poses"], best_idx)
    g_bbox = gallery_info["bboxes"][best_idx]

    xyz_dir = gallery_info["path"].parent / "xyz"
    xyz_map = load_xyz_map(xyz_dir, best_idx)
    pts2d_xyz = pts1 - [g_bbox[:2]] # cropped coordinates
    
    # PnP using the 3D model (xyz_map, pts2d_xyz) & 2D points of query (pts0)
    # t0 : a row vector
    R0, t0, pts3d, reproj_err, inlier_idx, match_counts = get_initial_pose(
        xyz_map, pts2d_xyz, pts0, conf, 
        K, R_g, reproj_thr
    )
    if which_query == 1:
        R0, t0 = Rt_compose_np((R0, t0), get_T0_stereo.Rt_rl)

    return R0, t0, which_query


def get_T0_stereo1(query_infos, gallery_info, nets, K, Q, R_lr, t_lr, reproj_thr):
    best_inds_l, score_l, query_crop_l = retrieve_topk(query_infos[0], gallery_info, nets[2])
    best_inds_r, score_r, query_crop_r = retrieve_topk(query_infos[1], gallery_info, nets[2])
    
    # prepare the references for matcher
    nMatches = 0 
    nMatchList = np.zeros(len(best_inds_l) + len(best_inds_r), dtype=np.int32)
    for i in range(len(best_inds_l)):
        gallery_crop_l, g_bbox_ext_l = make_gallery_square(gallery_info, best_inds_l[i], query_crop_l.shape[0])
        matching_results_l = compute_matches( query_crop_l, gallery_crop_l, nets[3] )
        nMatchList[i] = len(matching_results_l[0])
        if nMatches < nMatchList[i]:
            nMatches = nMatchList[i]
            matching_results = matching_results_l
            gallery_crop, g_bbox_ext = gallery_crop_l, g_bbox_ext_l
            best_idx = best_inds_l[i]
            which_query = 0
    
    ioff = len(best_inds_l)
    for i in range(len(best_inds_r)):
        gallery_crop_r, g_bbox_ext_r = make_gallery_square(gallery_info, best_inds_r[i], query_crop_r.shape[0])
        matching_results_r = compute_matches( query_crop_r, gallery_crop_r, nets[3] )
        nMatchList[i + ioff] = len(matching_results_r[0])
        if nMatches < nMatchList[i + ioff]:
            nMatches = nMatchList[i + ioff]
            matching_results = matching_results_r
            gallery_crop, g_bbox_ext = gallery_crop_r, g_bbox_ext_r
            best_idx = best_inds_r[i]
            which_query = 1

    q_bbox_ext = query_infos[which_query]["bbox"]
    
    # matcher input size in the matcher options
    pts0, pts1, conf = unmap_inlier_matches( matching_results, (q_bbox_ext, g_bbox_ext), query_infos[which_query]["mask"])
    # match_img = draw_matches_FHD(gallery_crop, query_crop, g_bbox_ext, q_bbox_ext, pts0, pts1, conf, None)
    
    # Get initial pose using the matched 2D-3D correspondences and PnP
    R_g, t_g = get_gallery_pose(gallery_info["poses"], best_idx)
    g_bbox = gallery_info["bboxes"][best_idx]

    xyz_dir = gallery_info["path"].parent / "xyz"
    xyz_map = load_xyz_map(xyz_dir, best_idx)
    pts2d_xyz = pts1 - [g_bbox[:2]] # cropped coordinates
    
    # PnP using the 3D model (xyz_map, pts2d_xyz) & 2D points of query (pts0)
    R0, t0, pts3d, reproj_err, inlier_idx, match_counts = get_initial_pose(
        xyz_map, pts2d_xyz, pts0, conf, 
        K, R_g, reproj_thr
    )
    if which_query == 1:
        R0 = R_lr.T @ R0
        t0 = (t0 - t_lr) @ R_lr 

    nMatchList.sort()
    if nMatchList[-2] < 100:
        # 3D points triangulated from stereo matching (p3) & 3D model (pts3d) posed by (R0, t0)
        pts3d = pts3d @ R0.T + t0
        pts0 = pts0[inlier_idx]

        md, idx = find_nearest_numpy( pts_l if which_query == 0 else pts_r, pts0 )
        # 2D points commonly detected from both the stereo query(l-r) feature matching and query-render feature matching
        common_idx = md < 1
        idx = idx[common_idx]
        meandist = np.mean( np.linalg.norm( p3[idx] - pts3d[common_idx], axis = 1 ) )

        print( f"[red]  === MEAN distance btw. triangulation & points from xyz[/red] {meandist*1000:.2f} mm, # pts {len(idx)}, from [bold red]{'Left' if which_query==0 else 'Right'}[/bold red]" )

        n_valid = match_counts[-1][1]
        assert n_valid == len(inlier_idx), "Length mismatch after get_initial_pose"
    
    return R0, t0, which_query


def get_T0_stereo0(query_infos, gallery_info, nets, K, reproj_thr):
    best_inds_l, score_l = retrieve_topk(query_infos[0], gallery_info, nets[2])
    best_inds_r, score_r = retrieve_topk(query_infos[1], gallery_info, nets[2])
    # match_res_lr = compute_matches( query_crop_l, query_crop_r, nets[3] )
    # match_img = draw_matches(query_crop_l, query_crop_r, match_res_lr[0][0::5], match_res_lr[1][0::5], match_res_lr[2][0::5], None)

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
    # match_img = draw_matches_FHD(gallery_crop, query_crop, g_bbox_ext, q_bbox_ext, pts0, pts1, conf, None)
    
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

        nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

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


def refine_pose_stereo_GS(query_infos, gProxy:GaussianRenderer, options):
    device = torch.device('cuda')
    
    # query bounding box     
    bbox_l, bbox_r = query_infos[0]["bbox"], query_infos[1]["bbox"]
    query_l,  query_r  = {"bbox": bbox_l}, {"bbox": bbox_r}
    render_l, render_r = {"bbox": bbox_l}, {"bbox": bbox_r}    
    
    # crop the mask and masked query image for the refinement step
    # c_ : cropped, m_ : masked
    query_l["c_mask"] = np2t(query_infos[0]["c_mask"] / 255.0)
    query_r["c_mask"] = np2t(query_infos[1]["c_mask"] / 255.0)
    query_l["m_crop"] = np2t(query_infos[0]["m_crop"] / 255.0)
    query_r["m_crop"] = np2t(query_infos[1]["m_crop"] / 255.0)

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

    losses_hist = []
    best_loss  = 1e9
    best_state = { "R": R0.detach(), "t": t0.detach(), "iter": 0 }

    ecc_loss_fn = ECCLoss()

    qm_tensors = torch.stack((query_l["c_mask"], query_r["c_mask"]))
    q_tensors = torch.stack((query_l["m_crop"], query_r["m_crop"]))

    for it in range(options["iters"]):
        optimizer.zero_grad()

        # T_cur (R_cur, t_cur) : T that estimated ΔR, Δt are composed to T0
        R_cur = so3_exp_map(delta_r) @ R0   # so3_exp_map(delta_r) : ΔR
        # tz clamp: init_t의 ±40% 범위로 제한 (frustum 이탈 방지)
        t0z = float(t0[2].item())
        t_raw = t0 + delta_t
        t_cur = torch.stack( [t_raw[0],  t_raw[1],  torch.clamp(t_raw[2], min=t0z*0.6, max=t0z*1.4)] )

        # full resolution render with identity cam (gsplat)
        gProxy.set_T(R_cur, t_cur)
        _r, _a, _ = gProxy.render()

        render_l_f = _r[0].permute(2,0,1).clamp(0,1)
        render_r_f = _r[1].permute(2,0,1).clamp(0,1)        
        render_l["crop"] = crop_chw_with_bbox(render_l_f, render_l["bbox"])
        render_r["crop"] = crop_chw_with_bbox(render_r_f, render_r["bbox"])
        render_l["c_mask"] = crop_with_bbox((_a[0].squeeze() > 0.5).float(), render_l["bbox"])
        render_r["c_mask"] = crop_with_bbox((_a[1].squeeze() > 0.5).float(), render_r["bbox"])
        rm_tensors = torch.stack((render_l["c_mask"], render_r["c_mask"]))

        losses = {}        
        # 1. Silhouette (Mask) Loss - render_crop:chw, query_mask: hw
        losses["mask"] = dice_loss( rm_tensors, qm_tensors )

        render_ci_mask = erode_binary_tensor(rm_tensors, 3) * qm_tensors
        render_l["mi_crop"] = render_l["crop"] * render_ci_mask[0] 
        render_r["mi_crop"] = render_r["crop"] * render_ci_mask[1] 
        r_tensors = torch.stack((render_l["mi_crop"], render_r["mi_crop"]))

        # 2. RGB Loss (SSIM, L1)
        loss_ssim = (dssim_loss(r_tensors, q_tensors) + dms_ssim_loss(r_tensors, q_tensors))
        loss_l1_rgb = F.l1_loss(r_tensors, q_tensors)

        # 3. Blur Loss
        r_blur = F.avg_pool2d(r_tensors, kernel_size=5, stride=1, padding=2)
        q_blur = F.avg_pool2d(q_tensors, kernel_size=5, stride=1, padding=2)
        losses["blur"] = F.l1_loss(r_blur, q_blur)  
        
        # 4. ECC Loss
        losses["ecc"]  = ecc_loss_fn( r_tensors, q_tensors )
        # 5. Gradient Matching Loss
        losses["grad"] = gradient_matching_loss(r_tensors, q_tensors, 
                                                render_ci_mask.unsqueeze(1))
        
        # 4. [핵심] Dynamic Weighting (Coarse-to-Fine)
        # 초반: Blur 중심 (크게 돌리기) / 후반: SSIM 중심 (칼같이 맞추기)
        progress = it / max(1, options["iters"])    # 현재 학습 진행도 (0.0 ~ 1.0)
        weight_blur = 1.0 * (1.0 - progress)        # 1.0에서 0.0으로 서서히 감소
        weight_ssim = 0.1 + 0.9 * progress          # 0.1에서 1.0으로 서서히 증가
        weight_l1   = 2.5                           # 기본 위치 유지를 위해 고정ge
        weight_mask = 1.0                           # 크기 유지를 위해 고정

        losses["ecc"] *= 2
        losses["grad"] *= 4        
        losses["blur"] *= 5        
        loss = losses["ecc"] + losses["grad"] 
        loss += losses["mask"]
        loss += weight_l1 * loss_l1_rgb + loss_ssim
        loss += weight_blur * losses["blur"]
        loss.backward() # loss-based gradient calculation & backpropagation 

        nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

        optimizer.step()
        scheduler.step()

        tracking_loss = losses["ecc"] + losses["grad"] + loss_ssim
        tracking_loss += losses["mask"] + (weight_l1 * loss_l1_rgb)
        tracking_loss += losses["blur"]
        track_loss_val = float(tracking_loss.item())

        losses_hist.append({"iter": it, "ecc_loss": losses["ecc"].item(), 
                       "grad_loss": losses["grad"].item(), 
                       "blur_loss": losses["blur"].item(),
                       "mask_loss": losses["mask"].item(),
                       "ssim_loss": loss_ssim.item(),
                       "rgb_loss": loss_l1_rgb.item(),
                       "loss": loss.item(),
                       "track_loss": track_loss_val})
        
        # 이제 loss_val이 아닌 tracking_loss 기준으로 최고를 갱신합니다.
        if track_loss_val < best_loss:
            best_loss  = track_loss_val
            best_state = {
                "R": R_cur.detach(),
                "t": t_cur.detach(),
                "iter": it,
                "loss": loss.item(),
                "track_loss": track_loss_val
            }

        # Early stopping
        if it >= options["early_stop_steps"]:
            loss_vals  = torch.tensor([l["track_loss"] for l in losses_hist])
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
    
    return best_R, best_t, best_state["loss"], (query_l, query_r), losses_hist

# loss Initial
# 4. [핵심] Dynamic Weighting (Coarse-to-Fine)
        # # 초반: Blur 중심 (크게 돌리기) / 후반: SSIM 중심 (칼같이 맞추기)
        # progress = it / max(1, options["iters"])    # 현재 학습 진행도 (0.0 ~ 1.0)
        # weight_blur = 1.0 * (1.0 - progress)        # 1.0에서 0.0으로 서서히 감소
        # weight_ssim = 0.1 + 0.9 * progress          # 0.1에서 1.0으로 서서히 증가
        # weight_l1   = 2.5                           # 기본 위치 유지를 위해 고정ge
        # weight_mask = 1.0                           # 크기 유지를 위해 고정

        # losses["ecc"] *= 2
        # losses["grad"] *= 4        
        # losses["blur"] *= 5        
        # loss = losses["ecc"] + losses["grad"] 
        # loss += weight_l1 * loss_l1_rgb + loss_ssim
        # loss += weight_blur * losses["mask"]
        # loss += weight_blur * losses["blur"]

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

        nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

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

        nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

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
    # match_img = draw_matches_FHD(rgb, query_l["m_crop"], render_l["bbox"], query_l["bbox"], pts0, pts1, conf, None)    
    
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

        nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

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
def refine_pose_stereo_PnP_GS_rerender(query_infos, which_query,
                                       gProxy:GaussianRenderer, options, matcher):
    assert gProxy.perturbation == False, "refine_pose_PnP requires gProxy.perturbation=False"

    device = torch.device('cuda')
    
    # query bounding box 
    bbox_l, bbox_r = query_infos[0]["bbox"], query_infos[1]["bbox"]
    query_l,  query_r  = {"bbox": bbox_l}, {"bbox": bbox_r}
    render_l, render_r = {"bbox": bbox_l}, {"bbox": bbox_r}
    
    # ndarray
    query_l["m_crop"] = query_infos[0]["m_crop"]
    query_r["m_crop"] = query_infos[1]["m_crop"]
    # query_l["m_crop"] = cv2.detailEnhance(query_l["m_crop"], sigma_s=10, sigma_r=0.15) 
    # query_r["m_crop"] = cv2.detailEnhance(query_r["m_crop"], sigma_s=10, sigma_r=0.15) 
    # tensor
    query_l["c_mask"] = np2t(query_infos[0]["c_mask"]) / 255.0    
    query_r["c_mask"] = np2t(query_infos[1]["c_mask"]) / 255.0    
    
    # ──────────────────────────────────────────────────────
    # re-PnP 
    # ──────────────────────────────────────────────────────
    K = gProxy.K_mat.squeeze(0)
    fx, fy = t2np(K.diag()[:2])
    cx, cy = t2np(K[:2, 2])

    R_rl = t2np(gProxy.R_lr.T)
    t_rl = -R_rl @ t2np(gProxy.t_lr)
    
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
        nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

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
        R1_r = gProxy.R_lr @ R1
        t1_r = t1 @ gProxy.R_lr.T + gProxy.t_lr 

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
            R_r_cur = gProxy.R_lr @ R_cur
            t_r_cur = t_cur @ gProxy.R_lr.T + gProxy.t_lr 

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

            nn.utils.clip_grad_norm_([delta_r, delta_t], max_norm=0.1)

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


def estimate_object_pose_stereo(query_imgs, gallery_info, nets, 
                                gProxy:GaussianRenderer, K, options):
    if not hasattr(estimate_object_pose_stereo, "Rt_lr_np"):
        estimate_object_pose_stereo.Rt_lr_np = (t2np(gProxy.R_lr), t2np(gProxy.t_lr))

    time_ = list()
    
    # FHD size: 1920 * 1080
    _IMG_SIZE = query_imgs[0].shape[:2]
    _MIN_BBOX_SIZE = gallery_info["bbox_size"]

    opt_preproc, opt_pnp, opt_render, opt_refiner = options
    bSpecularMask = opt_preproc['specular_mask']

    time_.append( sync_time() )
    
    # detect, segment, preprocess query images
    q_bboxes, cls_ids = detect_stereo(query_imgs, nets[0])
    if q_bboxes is None:
        print("[bold red]Stereo Detection FAILED[/bold red]")
        return None

    q_masks = segment_stereo(query_imgs, q_bboxes, nets[1])

    # bounding box size adjustment for stereo images        
    q_bboxes = [compute_bbox(q_masks[0]), compute_bbox(q_masks[1])] # initial bounding boxes    
    q_bboxes = make_same_sized_stereo_bboxes(q_bboxes, _IMG_SIZE, _MIN_BBOX_SIZE)    

    # after getting initial (tight) bounding boxes 
    if bSpecularMask:   
        q_masks[0] = q_masks[0] & ~get_specular_mask(query_imgs[0])
        q_masks[1] = q_masks[1] & ~get_specular_mask(query_imgs[1])

    time_.append( sync_time() )

    query_infos = [{
        "bbox": q_bboxes[0],
        "mask": q_masks[0],                                     # np.uint8        
        "c_mask": crop_with_bbox(q_masks[0], q_bboxes[0]),
        "crop": crop_with_bbox(query_imgs[0], q_bboxes[0])
    }, {
        "bbox": q_bboxes[1],
        "mask": q_masks[1],                                     # np.uint8
        "c_mask": crop_with_bbox(q_masks[1], q_bboxes[1]),
        "crop": crop_with_bbox(query_imgs[1], q_bboxes[1])
    }]
    # add cropped masked query (np.uint8)
    query_infos[0]["m_crop"] = apply_mask(query_infos[0]["crop"], query_infos[0]["c_mask"])
    query_infos[1]["m_crop"] = apply_mask(query_infos[1]["crop"], query_infos[1]["c_mask"])
    
    # background setting for rendering
    bgClr  = np.array(cv2.mean(query_infos[0]["crop"], 255-query_infos[0]["c_mask"]))
    bgClr += np.array(cv2.mean(query_infos[1]["crop"], 255-query_infos[1]["c_mask"]))
    bgClr  = np.round(bgClr[:3] / 2.0)
    gProxy.bg = np2t(bgClr).float() / 255.0

    # Best gallery selection - Feature matching - PnP
    # R0, t0 of left camera
    R0, t0, which_query = get_T0_stereo(query_infos, gallery_info, nets, 
                                        K, estimate_object_pose_stereo.Rt_lr_np, 
                                        opt_pnp["reproj_thr"])
    
    time_.append( sync_time() )

    METHOD = opt_refiner["method"]    
    # Render & Compare
    gProxy.set_T(R0, t0)
    if METHOD == '3DGS':
        R, t, t_loss, queries, losses = refine_pose_stereo_GS(query_infos, 
                                                            gProxy, opt_render)
    elif METHOD == 'RERENDER':
        R, t, t_loss, queries, losses = refine_pose_stereo_rerender(query_infos[0], query_infos[1], 
                                                            gProxy, opt_render, Rt_lr[0], Rt_lr[1])    
    elif METHOD == 'MIXED':
        R, t, t_loss, queries, losses = refine_pose_stereo_PnP_GS_rerender(query_infos, which_query,
                                                            gProxy, opt_render, nets[3])
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
    obj_config = get_named_config(config["objects"])
    obj_name   = obj_config['name']
    obj_params = obj_config['params']

    model_dir   = get_obj_path(obj_name, "model")
    obj_dir     = get_obj_path(obj_name, "object")
    obj_out_dir = get_obj_path(obj_name, "output")
    
    # Initialization
    input_num = config["input"]
    config["input"] = config["inputs"][input_num]

    if config["input"]["cam_type"] == "zed_m" or config["input"]["cam_type"] == "zed":
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
        
    # Loading networks
    nets = load_networks(config)
    
    ext_config = get_named_config(config['feat_extractors'])
    gallery_info = construct_galleryInfo(obj_dir, ext_config['name'])

    opt_render = config["renderer"]["options"]

    gaussian_proxy = GaussianRenderer( 
        init_gaussians(config["renderer"], resolve_ply_path(model_dir), scale=obj_params[0]), 
        K,
        np.array(config["renderer"]["options"]["background"], dtype=np.float32) / 255.0,
        opt_render["width"], opt_render["height"], False
    )
    gaussian_proxy.set_T_lr(R_lr, t_lr)

    opt_preproc = { "specular_mask": True if obj_params[1] == "specular" else False }
    opt_pnp     = { "reproj_thr": config["pnp"]["reproj_thr"]  }
    opt_refiner = { "method": config["refiner"]["method"] }
    
    # Input
    query_paths = get_query_paths(config)
    
    # perform
    performance = []
    pose_results = []
    time_detseg = []
    time_T0 = []
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
        query_img = load_rgb(query_paths[i])
        assert opt_render["width"] == query_img.shape[1] and opt_render["height"] == query_img.shape[0]

        if USE_STEREO:
            query_r_img = load_rgb(query_paths[i+1])

        print(f"\n[bold blue]Query file: {query_paths[i]}[/bold blue]")
        # time measurement
        st = time.perf_counter()
        
        if USE_STEREO:
            res = estimate_object_pose_stereo(
                [query_img, query_r_img], 
                gallery_info, nets, gaussian_proxy, K,  
                (opt_preproc, opt_pnp, opt_render, opt_refiner))
            R, t, R0, t0, queries, losses, time_proc = res # , t_loss, losses
            q_bbox = queries[0]["bbox"]
            q_bbox_r = queries[1]["bbox"]
            
            time_detseg.append(time_proc[0])
            time_T0.append(time_proc[1])
            time_refine.append(time_proc[-1])
        else:
            res = estimate_object_pose(
                query_img, 
                gallery_info, nets, gaussian_proxy, K, 
                opt_pnp, opt_render)
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
        fComparison = config["results"]["comparison"]
        fPerturbation = config["results"]["perturbation"]
        fLoss = config["results"]["loss"]
        res_dir = obj_out_dir / "result"
        res_dir.mkdir(parents=True, exist_ok=True)

        fnstem = query_paths[i].stem
        if fComparison:
            gaussian_proxy.set_T(R, t)
            comp_img = make_comp_image(crop_with_bbox(query_img, q_bbox), t2np(queries[0]["c_mask"]*255.0).astype(np.uint8), q_bbox, gaussian_proxy, R0, t0)
            vis_path = res_dir / f"{fnstem}_comp.png"            
            cv2.imwrite(str(vis_path), cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR))

        if fPerturbation:
            gaussian_proxy.set_T(R, t)
            pose_img = make_pose_validation_image(query_img, gaussian_proxy, q_bbox, 3.0)
            vis_path = res_dir / f"{fnstem}_pose_valid.png"            
            cv2.imwrite(str(vis_path), cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR))

        if fLoss:
            np.save( res_dir / f"{fnstem[:-1]}.npy", losses )

        if USE_STEREO and (fComparison or fPerturbation):
            R_r = R_lr.detach().cpu().numpy() @ R
            t_r = t @ R_lr.T.detach().cpu().numpy() + t_lr.detach().cpu().numpy() 

            R_r0 = R_lr.detach().cpu().numpy() @ R0
            t_r0 = t0 @ R_lr.T.detach().cpu().numpy() + t_lr.detach().cpu().numpy() 

            if fComparison:
                gaussian_proxy.set_T(R_r, t_r)
                comp_img = make_comp_image(crop_with_bbox(query_r_img, q_bbox_r), t2np(queries[1]["c_mask"]*255.0).astype(np.uint8), q_bbox_r, gaussian_proxy, R_r0, t_r0)
                vis_path = res_dir / f"{fnstem[:-1]}R_comp.png"
                cv2.imwrite(str(vis_path), cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR))

            if fPerturbation:
                gaussian_proxy.set_T(R_r, t_r)
                pose_img = make_pose_validation_image(query_r_img, gaussian_proxy, q_bbox_r, 3.0)
                vis_path = res_dir / f"{fnstem[:-1]}R_pose_valid.png"
                cv2.imwrite(str(vis_path), cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR))
   
    save_json(obj_out_dir / "result" / "refined_poses.json", pose_results)
    
    
    print("\n=== Performance Summary ===")
    print(f"detect/segment time: {np.mean(time_detseg[1:]) * 1000:.1f} ms")
    print(f"initial pose estimation time: {np.mean(time_T0[1:]) * 1000:.1f} ms")
    print(f"pose refinement estimation time: {np.mean(time_refine[1:]) * 1000:.1f} ms\n\n")    


if __name__ == "__main__":
    main_object_pose_estimation()



    
    
    

    