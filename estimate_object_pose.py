import locale
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from modules_6d.retrieval_dino_loftr import compute_loftr_matches, draw_loftr_matches
from modules_6d.retrieval_edm import (
    compute_edm_matches,
    compute_edm_trt_matches,
    load_edm_model,
    load_edm_trt_session,
    warmup_edm_trt_session,
)
from modules_6d.step6_translation import get_initial_pose, get_gallery_pose, project_axes_overlay
from refine_pose import (
    RigidPoseGaussianProxy, CosineWarmupScheduler, 
    dssim_loss, dms_ssim_loss, 
    so3_exp_map, crop_chw_with_bbox
)
from tqdm import tqdm
import time

from utils.io_utils import load_json, load_intrinsics, resolve_ply_path 
from utils.image_utils import (
    load_rgb, render_to_image, compute_nonblack_bbox,
    expand_bbox, compute_bbox, get_bbox_size, square_bbox, crop_with_bbox, square_pad_resize, unmap_to_full_image, 
    make_gallery_square, construct_galleryInfo, 
    apply_mask, get_mask_inlier_indices
)
from gaussian_renderer import GaussianModel


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
    

def init_gaussians(config, ply_path):
    options = config["options"]
    gaussians = GaussianModel(options["sh_degree"])
    gaussians.load_ply(ply_path, use_train_test_exp=False)
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

    file_path = Path("data/object")
    file_path = file_path / config["object"] / "query"

    if config_input["type"] == "file":
        return [file_path / config_input["name"]]
    elif config_input["type"] == "dir":
        return [f for f in file_path.iterdir() if f.is_file()]
    else:
        assert False    
    
    
def get_query(config):
    config_input = config["input"]
    assert config_input["type"] == "file" or config_input["type"] == "dir"

    file_path = Path("data/object")
    file_path = file_path / config["object"] / "query"
    
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
    # kind : {"output", "gallery", "xyz", "model"}
    if kind == "model":
        return Path("data/object") / obj_name / "model"
    
    out_path = Path("data/output") / obj_name
    if kind == "output":
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


def make_comp_image(query, gaussian_proxy:RigidPoseGaussianProxy, q_bbox):
    q_crop = crop_with_bbox(query, q_bbox)

    _r, _, _ = gaussian_proxy.render(width = query.shape[1], height = query.shape[0])
    r_crop = crop_with_bbox(render_to_image(_r[0]), q_bbox)
    gray = cv2.cvtColor(r_crop, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours( (gray > 10).astype(np.uint8) * 255, 
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    q_crop_annotation = q_crop.copy()
    R, t = gaussian_proxy.get_T()
    R_np = R.cpu().numpy()
    t_np = t.cpu().numpy()
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
    cv2.drawContours(q_crop_annotation, contours, -1, (0,255,0), 1)
    
    q_w = q_crop.shape[1]
    alpha = 0.6
    overlay_crop = cv2.addWeighted(q_crop, alpha, r_crop, 1.0 - alpha, 0)
    res_img = np.zeros((q_crop.shape[0], q_crop.shape[1] * 4, 3), dtype=np.uint8)
    res_img[:, :q_w] = q_crop
    res_img[:, q_w:q_w*2] = r_crop
    res_img[:, q_w*2:q_w*3] = overlay_crop
    res_img[:, q_w*3:] = q_crop_annotation
    cv2.putText(res_img, "Query",   (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(res_img, "Render",  (q_w + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(res_img, "Overlay", (q_w*2 + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(res_img, "Query+Render contours",   (q_w*3 + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return res_img


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

    # A cropped query image for extracting DINOv2 feature
    ext_net = extractor[0]
    ext_opts = extractor[1]
    dino_size = ext_opts["input_size"]
    query_dino_in = square_pad_resize(masked_query_crop, dino_size * 2)
    qfeat = ext_net.encode_4rgb(query_dino_in)
    # qfeat = extractor.encode_rgb(query_dino_in)    
    
    scores = (g_feats @ qfeat).numpy()
    best_item = np.argmax(scores)
    
    return best_item, scores[best_item], masked_query_crop, q_bbox_ext


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

    # match_img = draw_loftr_matches(query_l, gallery_l, mkpts0, mkpts1, conf, None)

    return mkpts0, mkpts1, conf


def unmap_inlier_matches(matching_results, bboxes, mask, match_in_size):
    # 마스크 필터링: query crop coords → full image coords → mask 체크
    pts0_full = unmap_to_full_image(matching_results[0], bboxes[0], match_in_size)
    mask_keep = get_mask_inlier_indices(pts0_full, mask)
    pts0_full = pts0_full[mask_keep]

    pts1_full = unmap_to_full_image(matching_results[1][mask_keep], bboxes[1], match_in_size)
    conf = matching_results[2][mask_keep]

    return pts0_full, pts1_full, conf


def get_T0(query_info, gallery_info, nets, K, reproj_thr):
    best_idx, score, query_crop, q_bbox_ext = retrieve_best(query_info, gallery_info, nets[2])
    # prepare the best reference for matcher 
    gallery_crop, g_bbox_ext = make_gallery_square(gallery_info, best_idx, query_crop.shape[0])

    matching_results = compute_matches( query_crop, gallery_crop, nets[3] )
    # matcher input size in the matcher options
    match_in_size = nets[3]["options"]["input_size"]
    pts0, pts1, conf = unmap_inlier_matches( matching_results, (q_bbox_ext, g_bbox_ext), query_info["mask"], match_in_size)

    # gimg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # qimg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # gimg[g_bbox_ext[1]:g_bbox_ext[3], g_bbox_ext[0]:g_bbox_ext[2]] = gallery_crop
    # qimg[q_bbox_ext[1]:q_bbox_ext[3], q_bbox_ext[0]:q_bbox_ext[2]] = query_crop
    # match_img = draw_loftr_matches(qimg, gimg, pts0, pts1, conf, None)
    
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
    return R0, t0


def refine_pose(query_info, gProxy:RigidPoseGaussianProxy, options):
    device = 'cuda'
    
    # query bounding box 
    q_bbox = np.array(query_info["bbox"])
    q_side = get_bbox_size(q_bbox)
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
        loss_mask = F.l1_loss(render_alpha, query_mask)

        # 2. RGB Loss (SSIM, L1)
        loss_ssim = dssim_loss(masked_render_crop, query_crop) + dms_ssim_loss(masked_render_crop, query_crop)
        loss_l1_rgb = F.l1_loss(masked_render_crop, query_crop)

        # 3. Blur Loss
        blur_target = F.avg_pool2d(query_crop.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)
        blur_render = F.avg_pool2d(masked_render_crop.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)
        loss_blur = F.l1_loss(blur_render, blur_target)

        ecc_loss = ecc_loss_fn(masked_render_crop.unsqueeze(0), query_crop.unsqueeze(0))

        # render_alpha = (render_crop.sum(dim=0, keepdim=True) > 0.05).float()
        # loss_mask = F.l1_loss(render_alpha, query_mask)

        # # 2. RGB Loss (SSIM, L1)
        # loss_ssim = dssim_loss(render_crop, query_crop) + dms_ssim_loss(render_crop, query_crop)
        # loss_l1_rgb = F.l1_loss(render_crop, query_crop)

        # # 3. Blur Loss
        # blur_target = F.avg_pool2d(query_crop.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)
        # blur_render = F.avg_pool2d(render_crop.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)
        # loss_blur = F.l1_loss(blur_render, blur_target)

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

        tracking_loss = float(loss_ssim.item() + loss_l1_rgb.item() + loss_blur.item() + loss_mask.item())
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
    
    
def estimate_object_pose(query, gallery_info, nets, gProxy:RigidPoseGaussianProxy, K, pnp_option, render_options):
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
    R0, t0 = get_T0(query_info, gallery_info, nets, K, pnp_option["reproj_thr"])
    t_step56 = sync_time()

    # Render & Compare
    gProxy.set_T(R0, t0)
    R, t, t_loss, bbox = refine_pose(query_info, gProxy, render_options)
    t_refine = sync_time()
    print(f"R: {R} \nt: {t*100}(cm) \ntracking loss: {t_loss}")

    return R, t, t_loss, bbox
    

def main_object_pose_estimation():
    assert torch.cuda.is_available(), "CUDA required."

    # Setup
    config = load_json("ope_config.json")
    model_dir = get_obj_path(config["object"], "model")
    obj_out_dir = get_obj_path(config["object"], "output")
    
    # Initialization
    K = load_intrinsics(get_K_path(config["input"]))

    nets = load_networks(config)
    
    gallery_info = construct_galleryInfo(obj_out_dir)

    gaussian_proxy = RigidPoseGaussianProxy( 
        init_gaussians(config["renderer"], resolve_ply_path(model_dir)), 
        np.eye(3, dtype=np.float32), np.zeros((3,), dtype=np.float32), K,
        bg_val = np.array(config["renderer"]["options"]["background"], dtype=np.float32) / 255.0
    )   # (gaussians, R, t, K, bg color)

    pnp_option = { "reproj_thr": config["pnp"]["reproj_thr"]  }
    render_options = config["renderer"]["options"]
    
    # Input
    query_paths = get_query_paths(config)
    
    # perform
    performance = []
    for i in range(len(query_paths)):
        query = load_rgb(query_paths[i])
        assert render_options["width"] == query.shape[1] and render_options["height"] == query.shape[0]

        # time measurement
        st = time.perf_counter()
        
        R, t, t_loss, q_bbox = estimate_object_pose(query, gallery_info, nets, gaussian_proxy, K, pnp_option, render_options)
        
        et = time.perf_counter()
        performance.append((et - st, t_loss))

        # Visualization
        gaussian_proxy.set_T(R, t)
        comp_img = make_comp_image(query, gaussian_proxy, q_bbox)
        vis_path = obj_out_dir / "result" / f"{query_paths[i].stem}_comp.png"
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_path), cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR))

    print("\n=== Performance Summary ===")
    print(f"pose estimation time: {np.mean([p[0] for p in performance]):.6f} seconds")
    print(f"tracking loss: {np.mean([p[1] for p in performance]):.6f}")


if __name__ == "__main__":
    main_object_pose_estimation()



    
    
    

    