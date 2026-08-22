import warnings
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from utils.io_utils import ensure_dir
from utils.general_utils import sync_time


warnings.filterwarnings("ignore", message="Can't initialize NVML")


DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=DINO_MEAN, std=DINO_STD)
])


def preprocess_for_dinov2(image_uint8: np.ndarray, mask: np.ndarray = None):
    """
    Args:
        image_uint8: [H, W, 3] 형태의 RGB Numpy 배열 (0~255)
        mask: [H, W] 형태의 마스크 배열 (0 또는 255)
    """
    image_tensor = transform(image_uint8).unsqueeze(0)
    
    # 4. 마스크 처리 (옵션)
    mask_tensor = None
    if mask is not None:
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        if mask.dtype == np.uint8:
            mask_tensor = mask_tensor.float() / 255.0

    return image_tensor, mask_tensor

  
def load_extractor(config):
    options = config['options']
    if config["name"] == 'DINOv2_ONNX':    
        sess = load_dino_trt_session(
            options["onnx"],
            trt_cache_dir=options.get("trt_cache_dir"),
            require_gpu=options.get("require_gpu", True),
        )
        extractor = DinoV2ExtractorTRT(options["model"], sess)
        if options.get("warmup", True):
            t0 = sync_time()
            side_len = int(options.get("input_size", 224))
            dummy_rgb = np.zeros((side_len * 2, side_len * 2, 3), dtype=np.uint8)
            for _ in range(int(options.get("warmup_runs", 1))):
                extractor.encode_4rgb(dummy_rgb)
            t1 = sync_time()
            print(f"  [DINO TRT] Warmup done ({(t1 - t0):.3f}s, runs={options.get('warmup_runs', 1)})")
        return extractor, options
    # elif config["name"] == 'DINOv2_MASK':
    #     return DinoV2Extractor1(options["model"], device='cuda'), options
    
    return DinoV2Extractor(options["model"], device='cuda'), options


class DinoV2Extractor:
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        try:
            from transformers import AutoImageProcessor, AutoModel
        except Exception as e:
            raise ImportError(
                'transformers is required'
            ) from e

        hf_name = {
            'dinov2_vits14': 'facebook/dinov2-small',
            'dinov2_vitb14': 'facebook/dinov2-base',
            'dinov2_vitl14': 'facebook/dinov2-large',
        }.get(model_name, model_name)

        model_path = Path('../../.cache/huggingface/hub/models--facebook--dinov2-small')
        downloaded = True if model_path.exists() else False
        self.processor = AutoImageProcessor.from_pretrained(hf_name, local_files_only=downloaded)
        self.model = AutoModel.from_pretrained(hf_name, local_files_only=downloaded, output_hidden_states=True).to(self.device)
        self.model.eval()
        self.patch_size = 14  # DINOv2 모델의 패치 크기 (14x14)

    @torch.no_grad()
    def encode_bgr(self, img_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.encode_rgb(rgb)        
    
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
        return feat.squeeze(0).detach()
    
    @torch.no_grad()
    def encode_4rgb(self, img_rgb: np.ndarray) -> torch.Tensor:
        assert img_rgb.shape[0] == 224 * 2 and img_rgb.shape[1] == 224 * 2, "Input must be 448x448 RGB image containing 4 tiles"

        feat = self.encode_rgb([img_rgb[:224, :224], img_rgb[:224, 224:], img_rgb[224:, :224], img_rgb[224:, 224:]])
        return feat.reshape(-1)        

    def extract_masked_patch_tokens(
        self, 
        images: torch.Tensor, 
        masks: torch.Tensor, 
        mask_threshold: float = 0.5
    ) -> List[torch.Tensor]:
        """
        Args:
            images: [B, 3, H, W] 정규화된 텐서 (H, W는 14의 배수여야 함)
            masks: [B, 1, H, W] 0~1 사이 마스크 텐서
            mask_threshold: 패치가 마스크 내부로 간주될 비율 임계값
            
        Returns:
            List of Tensors: 배치 내 각 샘플마다 [N_valid_patches, Hidden_Dim] 형태의 텐서 리스트
        """
        B, C, H, W = images.shape
        grid_h, grid_w = H // self.patch_size, W // self.patch_size

        # 1. Hugging Face AutoModel Forward
        # outputs.last_hidden_state: [B, 1 + (grid_h * grid_w), D]
        outputs = self.model(pixel_values=images)
        
        # [CLS] 토큰(인덱스 0) 제거 후 순수 공간 패치 토큰만 선택 -> [B, grid_h * grid_w, D]
        patch_tokens = outputs.hidden_states[9 + 1][:, 1:, :]
        # patch_tokens = outputs.last_hidden_state[:, 1:, :]

        # 2. 마스크를 패치 그리드 크기(grid_h, grid_w)로 Area Downsampling
        downsampled_mask = F.adaptive_avg_pool2d(masks, (grid_h, grid_w))  # [B, 1, grid_h, grid_w]
        downsampled_mask = downsampled_mask.flatten(2).squeeze(1)          # [B, grid_h * grid_w]

        # 3. 마스크 내부 패치들만 추출 및 L2 정규화
        valid_tokens_list = []
        for b in range(B):
            valid_mask = downsampled_mask[b] >= mask_threshold
            
            # 마스크 영역이 매우 작아 threshold를 넘는 패치가 없으면 최댓값을 가진 패치 1개 선택
            if valid_mask.sum() == 0:
                valid_mask = downsampled_mask[b] >= downsampled_mask[b].max()

            tokens = patch_tokens[b, valid_mask]       # [N_valid, D]
            tokens = F.normalize(tokens, p=2, dim=-1)  # 코사인 유사도 연산을 위한 L2 정규화
            valid_tokens_list.append(tokens)

        return valid_tokens_list

    def compute_asymmetric_chamfer_similarity(
        self, 
        query_tokens: torch.Tensor, 
        gallery_tokens_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        쿼리(가려짐 발생)와 각 갤러리 템플릿(전체 물체) 간 비대칭 챔퍼 유사도 계산
        
        S(Q, G) = (1 / |Q|) * sum_{q in Q} max_{g in G} (q · g)
        
        Args:
            query_tokens: [N_q, D] - 쿼리의 유효 패치 토큰
            gallery_tokens_list: List of [N_g_i, D] - 각 갤러리 템플릿의 유효 패치 토큰 리스트
            
        Returns:
            similarities: [len(gallery_tokens_list)]
        """
        scores = []
        for g_tokens in gallery_tokens_list:
            # 코사인 유사도 행렬 계산: [N_q, N_g]
            sim_matrix = torch.matmul(query_tokens, g_tokens.T)
            
            # 각 쿼리 패치에 대해 가장 유사한 갤러리 패치의 유사도(Max over Gallery)
            max_sim_per_query_patch, _ = sim_matrix.max(dim=1)  # [N_q]
            
            # 쿼리 패치들에 대한 평균 유사도 (Asymmetric Mean)
            chamfer_sim = max_sim_per_query_patch.mean()
            scores.append(chamfer_sim)

        return torch.stack(scores)



def load_dino_trt_session(onnx_path, trt_cache_dir=None, require_gpu=True):
    from modules_6d.retrieval_edm import _try_preload_ort_gpu_libs
    import onnxruntime as ort

    _try_preload_ort_gpu_libs()

    so = ort.SessionOptions()
    so.log_severity_level = 0  # 가장 상세한 로그 출력 (Verbose)

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
            'trt_max_workspace_size': 4294967296
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
                'transformers is required'                
            ) from e

        hf_name = {
            'dinov2_vits14': 'facebook/dinov2-small',
            'dinov2_vitb14': 'facebook/dinov2-base',
            'dinov2_vitl14': 'facebook/dinov2-large',
        }.get(model_name, model_name)

        model_path = Path('../../.cache/huggingface/hub/models--facebook--dinov2-small')
        downloaded = True if model_path.exists() else False
        self.processor = AutoImageProcessor.from_pretrained(hf_name, local_files_only=downloaded)
        self._sess = sess
        self._input_name = sess.get_inputs()[0].name
        self.device = "cuda"

    def _encode_np(self, pixel_values: np.ndarray) -> torch.Tensor:
        outputs = self._sess.run(None, {self._input_name: pixel_values})
        if len(outputs) > 1 and outputs[1].ndim == 2:
            feat = torch.from_numpy(outputs[1])
        else:
            feat = torch.from_numpy(outputs[0][:, 0, :])
        return F.normalize(feat.cuda(), dim=-1).squeeze(0)
    
    def encode_bgr(self, img_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.encode_rgb(rgb)        

    def encode_rgb(self, img_rgb: np.ndarray) -> torch.Tensor:
        pixel_values = self.processor(images=img_rgb, do_resize=False, do_center_crop=False, return_tensors="np")["pixel_values"]
        return self._encode_np(pixel_values)

    def encode_4rgb(self, img_rgb: np.ndarray) -> torch.Tensor:
        assert img_rgb.shape[0] == 224 * 2 and img_rgb.shape[1] == 224 * 2, \
            "Input must be 448x448 RGB image containing 4 tiles"
        feat0 = self.encode_rgb(img_rgb[:224, :224])
        feat1 = self.encode_rgb(img_rgb[:224, 224:])
        feat2 = self.encode_rgb(img_rgb[224:, :224])
        feat3 = self.encode_rgb(img_rgb[224:, 224:])
        return torch.row_stack((feat0, feat1, feat2, feat3)).reshape(-1)
    
    