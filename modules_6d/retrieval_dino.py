import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_image(path, color=True):
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {path}')
    return img


def list_gallery_images(gallery_dir: Path) -> List[Path]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    files = [p for p in sorted(gallery_dir.iterdir()) if p.suffix.lower() in exts]
    return files


def square_pad_resize(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0+h, x0:x0+w] = img_bgr
    out = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return out


def make_preview_strip(query_img, gallery_imgs, labels, out_path):
    thumbs = []
    q = cv2.resize(query_img, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.putText(q, 'query', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    thumbs.append(q)
    for img, label in zip(gallery_imgs, labels):
        t = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.putText(t, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        thumbs.append(t)
    strip = np.concatenate(thumbs, axis=1)
    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), strip)


def build_contact_sheet(items: List[Tuple[np.ndarray, str]], out_path: Path, cols: int = 3, thumb=(320, 180)):
    if not items:
        return
    tw, th = thumb
    rows = (len(items) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, (img, label) in enumerate(items):
        r = i // cols
        c = i % cols
        t = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(t, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
        sheet[r*th:(r+1)*th, c*tw:(c+1)*tw] = t
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), sheet)


def make_query_vs_best_image(query_crop, best_crop, out_size=320):
    q = square_pad_resize(query_crop, out_size)
    b = square_pad_resize(best_crop, out_size)
    canvas = np.zeros((out_size, out_size * 2, 3), dtype=np.uint8)
    canvas[:, :out_size] = q
    canvas[:, out_size:] = b

    cv2.putText(canvas, "query", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "best", (out_size + 12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


class DinoV2Extractor:
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        try:
            from transformers import AutoImageProcessor, AutoModel
        except Exception as e:
            raise ImportError(
                'transformers is required for step4 DINO retrieval. '\
                'Install with: pip install transformers'
            ) from e

        hf_name = {
            'dinov2_vits14': 'facebook/dinov2-small',
            'dinov2_vitb14': 'facebook/dinov2-base',
            'dinov2_vitl14': 'facebook/dinov2-large',
        }.get(model_name, model_name)

        self.processor = AutoImageProcessor.from_pretrained(hf_name)
        self.model = AutoModel.from_pretrained(hf_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_bgr(self, img_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state[:, 0]
        feat = F.normalize(feat, dim=-1)
        return feat.squeeze(0).detach().cpu()
    
    @torch.no_grad()
    def encode_rgb(self, img_rgb: np.ndarray) -> torch.Tensor:
        inputs = self.processor(images=img_rgb, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state[:, 0]
        feat = F.normalize(feat, dim=-1)
        return feat.squeeze(0).detach().cpu()
    
    @torch.no_grad()
    def encode_4rgb(self, img_rgb: np.ndarray) -> torch.Tensor:
        assert img_rgb.shape[0] == 224 * 2 and img_rgb.shape[1] == 224 * 2, "Input must be 448x448 RGB image containing 4 tiles"

        feat0 = self.encode_rgb(img_rgb[:224, :224])
        feat1 = self.encode_rgb(img_rgb[:224, 224:])
        feat2 = self.encode_rgb(img_rgb[224:, :224])
        feat3 = self.encode_rgb(img_rgb[224:, 224:])
        return torch.row_stack( (feat0, feat1, feat2, feat3) ).reshape(-1)


def load_dino_trt_session(onnx_path, trt_cache_dir=None, require_gpu=True):
    from modules_6d.retrieval_edm import _try_preload_ort_gpu_libs
    import onnxruntime as ort

    _try_preload_ort_gpu_libs()

    providers = []
    trt_options = {}
    if trt_cache_dir:
        trt_cache_dir = Path(trt_cache_dir)
        trt_cache_dir.mkdir(parents=True, exist_ok=True)
        trt_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(trt_cache_dir),
            'trt_profile_min_shapes': 'pixel_values:1x3x224x224',
            'trt_profile_opt_shapes': 'pixel_values:1x3x224x224',
            'trt_profile_max_shapes': 'pixel_values:1x3x224x224',
            'trt_engine_builder_workspace_size': 4294967296
        }
    providers.append(("TensorrtExecutionProvider", trt_options))
    providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f"  [DINO TRT] Loaded: {onnx_path}")
    print(f"  [DINO TRT] Providers: {session.get_providers()}")
    if require_gpu and not any(
        p in session.get_providers()
        for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider")
    ):
        raise RuntimeError(
            "DINO TRT requested, but ONNXRuntime fell back to CPUExecutionProvider."
        )
    return session


class DinoV2ExtractorTRT:
    """DINOv2 inference via ONNXRuntime TensorRT EP."""

    def __init__(self, model_name: str, sess):
        try:
            from transformers import AutoImageProcessor
        except Exception as e:
            raise ImportError(
                'transformers is required for DinoV2ExtractorTRT. '
                'Install with: pip install transformers'
            ) from e

        hf_name = {
            'dinov2_vits14': 'facebook/dinov2-small',
            'dinov2_vitb14': 'facebook/dinov2-base',
            'dinov2_vitl14': 'facebook/dinov2-large',
        }.get(model_name, model_name)

        self.processor = AutoImageProcessor.from_pretrained(hf_name)
        self._sess = sess
        self._input_name = sess.get_inputs()[0].name
        self.device = "cuda"

    def _encode_np(self, pixel_values: np.ndarray) -> torch.Tensor:
        outputs = self._sess.run(None, {self._input_name: pixel_values})
        if len(outputs) > 1 and outputs[1].ndim == 2:
            feat = torch.from_numpy(outputs[1])
        else:
            feat = torch.from_numpy(outputs[0][:, 0, :])
        return F.normalize(feat, dim=-1).squeeze(0)

    def encode_bgr(self, img_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pixel_values = self.processor(images=rgb, return_tensors="np")["pixel_values"].astype(np.float32)
        return self._encode_np(pixel_values)

    def encode_rgb(self, img_rgb: np.ndarray) -> torch.Tensor:
        pixel_values = self.processor(images=img_rgb, return_tensors="np")["pixel_values"].astype(np.float32)
        return self._encode_np(pixel_values)

    def encode_4rgb(self, img_rgb: np.ndarray) -> torch.Tensor:
        assert img_rgb.shape[0] == 224 * 2 and img_rgb.shape[1] == 224 * 2, \
            "Input must be 448x448 RGB image containing 4 tiles"
        feat0 = self.encode_rgb(img_rgb[:224, :224])
        feat1 = self.encode_rgb(img_rgb[:224, 224:])
        feat2 = self.encode_rgb(img_rgb[224:, :224])
        feat3 = self.encode_rgb(img_rgb[224:, 224:])
        return torch.row_stack((feat0, feat1, feat2, feat3)).reshape(-1)


def run_step4_dino_retrieval(args):
    query_path = Path(args.query_masked_path)
    gallery_dir = Path(args.gallery_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    query_img = load_image(query_path)
    qbox = compute_nonblack_bbox(query_img, thresh=args.nonblack_thresh)
    qbox = expand_bbox(qbox, args.crop_margin, query_img.shape[1], query_img.shape[0])
    query_crop = crop_with_bbox(query_img, qbox)
    query_in = square_pad_resize(query_crop, args.dino_input_size)

    gallery_files = list_gallery_images(gallery_dir)
    if not gallery_files:
        raise FileNotFoundError(f'No gallery images found in: {gallery_dir}')

    extractor = DinoV2Extractor(args.dino_model, device=args.device)
    qfeat = extractor.encode_bgr(query_in)

    scores = []
    topk_items = []
    best_img = None
    best_score = -1e9
    best_path = None
    best_bbox = None

    for gp in gallery_files:
        img = load_image(gp)
        gbox = compute_nonblack_bbox(img, thresh=args.nonblack_thresh)
        gbox = expand_bbox(gbox, args.crop_margin, img.shape[1], img.shape[0])
        gcrop = crop_with_bbox(img, gbox)
        gin = square_pad_resize(gcrop, args.dino_input_size)
        gfeat = extractor.encode_bgr(gin)
        score = float(torch.dot(qfeat, gfeat).item())
        record = {
            'file': gp.name,
            'path': str(gp),
            'score_cosine': score,
            'bbox_xyxy': [int(v) for v in gbox],
        }
        scores.append(record)
        if score > best_score:
            best_score = score
            best_img = img
            best_path = gp
            best_bbox = gbox

    scores = sorted(scores, key=lambda x: x['score_cosine'], reverse=True)
    topk = scores[:args.topk]

    topk_gallery_imgs = []
    topk_sheet_items = []
    for i, rec in enumerate(topk):
        img = load_image(rec['path'])
        topk_gallery_imgs.append(img)
        topk_sheet_items.append((img, f"#{i+1} {rec['file']} {rec['score_cosine']:.4f}"))

    best_render_out = out_dir / 'best_render.png'
    cv2.imwrite(str(best_render_out), best_img)

    query_vs_best_out = out_dir / 'query_vs_best.png'
    best_crop = crop_with_bbox(best_img, best_bbox)
    make_preview_strip(
        query_in,
        [square_pad_resize(best_crop, args.dino_input_size)],
        [f"best {best_path.name} {best_score:.4f}"],
        query_vs_best_out,
    )

    topk_preview_out = out_dir / 'topk_preview.png'
    build_contact_sheet(topk_sheet_items, topk_preview_out, cols=min(3, args.topk))

    scores_out = out_dir / 'retrieval_scores.json'
    summary_out = out_dir / 'step4_summary.json'
    save_json(scores_out, {
        'stage': 'step4',
        'sim_method': 'dino',
        'query_masked_path': str(query_path),
        'gallery_dir': str(gallery_dir),
        'dino_model': args.dino_model,
        'dino_input_size': args.dino_input_size,
        'nonblack_thresh': args.nonblack_thresh,
        'crop_margin': args.crop_margin,
        'num_gallery': len(gallery_files),
        'best_render': best_path.name,
        'best_score': best_score,
        'topk': topk,
        'all_scores_sorted': scores,
    })
    save_json(summary_out, {
        'stage': 'step4',
        'status': 'ok',
        'best_render': best_path.name,
        'best_score': best_score,
        'outputs': {
            'retrieval_scores': str(scores_out),
            'best_render': str(best_render_out),
            'query_vs_best': str(query_vs_best_out),
            'topk_preview': str(topk_preview_out),
        }
    })

    print('=' * 60)
    print('[Step 5] DINO retrieval complete')
    print(f'  query        : {query_path}')
    print(f'  gallery_dir   : {gallery_dir}')
    print(f'  num_gallery   : {len(gallery_files)}')
    print(f'  best_render   : {best_path.name}')
    print(f'  best_score    : {best_score:.4f}')
    print(f'  scores_json   : {scores_out}')
    print(f'  topk_preview  : {topk_preview_out}')
    print('=' * 60)
