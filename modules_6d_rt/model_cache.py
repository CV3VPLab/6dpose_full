"""
model_cache.py
==============
Holds all inference models and pre-computed gallery features in GPU memory
so they are loaded only once per process and reused across pipeline stages.

Usage:
    cache = ModelCache()
    cache.load_all(args, device="cuda")
    cache.load_gallery_features(args, gallery_dir="...", device="cuda")

Then pass `model_cache=cache` to step1 and step5 entry-points.
"""

import numpy as np
from pathlib import Path


class ModelCache:
    def __init__(self):
        self.yolo             = None   # ultralytics YOLO
        self.sam2             = None   # SAM2ImagePredictor
        self.dino             = None   # DinoV2Extractor
        self.loftr            = None   # KF.LoFTR (eval, on device)
        self.gallery_feats     = None   # dict: {filename: {"feat": np.ndarray, "path": str}}
        self.gallery_files     = None   # list[Path]: sorted gallery image paths (step5 dir scan cache)
        self.gallery_feat_matrix = None # np.ndarray (N, D): stacked features for vectorized scoring
        self.gaussians         = None   # GaussianModel (3DGS, for steps 6/7)
        self.ply_med_norm      = None   # float: median ||xyz|| of canonical PLY (step6 scale correction)
        self.gallery_pose_dict = None   # dict: {index(int) -> pose_record} (step6 pose lookup)
        self.xyz_scale_factor  = None   # float: pre-computed XYZ map scale correction (constant per gallery)

    # ─────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────

    def load_all(self, args, device: str = "cuda") -> None:
        """Load YOLO, SAM2, DINOv2, and LoFTR into GPU memory."""
        import kornia.feature as KF
        from modules_6d.yolo_sam import load_yolo_model, load_sam2_predictor
        from modules_6d.retrieval_dino import DinoV2Extractor

        print("=" * 60)
        print("[ModelCache] Loading all models into GPU memory ...")
        print("=" * 60)

        print("  [1/4] YOLO ...")
        self.yolo = load_yolo_model(args.yolo_weights)

        print("  [2/4] SAM2 ...")
        self.sam2 = load_sam2_predictor(
            sam2_repo=args.sam2_repo,
            checkpoint_path=args.sam2_checkpoint,
            config_path=args.sam2_config,
            device=device,
        )

        print("  [3/4] DINOv2 ...")
        self.dino = DinoV2Extractor(args.dino_model, device=device)

        print("  [4/4] LoFTR ...")
        self.loftr = (
            KF.LoFTR(pretrained=args.loftr_pretrained)
            .to(device)
            .eval()
        )

        print("[ModelCache] All models loaded.\n")

    def warmup_cuda(self, device: str = "cuda") -> None:
        """
        Run tiny dummy forward passes through each model to pre-compile CUDA
        kernels.  Call this once after load_all() so the *first real frame*
        does not pay the JIT overhead.
        """
        import time
        import numpy as np
        import torch

        print("[ModelCache] CUDA warmup ...")
        t0 = time.perf_counter()

        # --- YOLO warmup (ultralytics auto-warms, but an explicit call seals it)
        if self.yolo is not None:
            dummy_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
            self.yolo.predict(source=dummy_bgr, conf=0.25, verbose=False)

        # --- SAM2 warmup: set_image triggers the full ViT encoder on a tiny image
        if self.sam2 is not None:
            dummy_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            self.sam2.set_image(dummy_rgb)
            self.sam2.predict(
                point_coords=None,
                point_labels=None,
                box=np.array([[0, 0, 32, 32]], dtype=np.float32),
                multimask_output=False,
            )

        # --- DINOv2 warmup
        if self.dino is not None:
            dummy_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
            self.dino.encode_bgr(dummy_bgr)

        # --- LoFTR warmup
        if self.loftr is not None:
            import kornia.feature as KF
            tiny = torch.zeros(1, 1, 64, 64, device=device)
            with torch.no_grad():
                self.loftr({"image0": tiny, "image1": tiny})

        print(f"[ModelCache] CUDA warmup done ({time.perf_counter() - t0:.3f}s)\n")

    # ─────────────────────────────────────────────────────────────────────
    # Gallery feature cache (DINOv2)
    # ─────────────────────────────────────────────────────────────────────

    def load_gallery_features(
        self,
        args,
        gallery_dir: str,
        device: str = "cuda",
        feat_cache_dir: str | None = None,
    ) -> None:
        """
        Compute (or load from disk cache) DINOv2 features for every gallery
        image and keep them in memory.  After this call, step5 skips the
        per-frame feature extraction for gallery images entirely.

        Parameters
        ----------
        feat_cache_dir : directory for .npy file-level cache.
                         Defaults to <gallery_dir>/../dino_cache_ds.
        """
        from modules_6d_rt.retrieval_dino_loftr_rt import (
            load_or_compute_gallery_features,
        )

        gallery_dir = Path(gallery_dir)
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        gallery_files = sorted(
            p for p in gallery_dir.iterdir() if p.suffix.lower() in exts
        )
        if not gallery_files:
            raise FileNotFoundError(f"No gallery images found in: {gallery_dir}")

        if feat_cache_dir is None:
            feat_cache_dir = gallery_dir.parent / "dino_cache_ds"
        else:
            feat_cache_dir = Path(feat_cache_dir)

        print("=" * 60)
        print(f"[ModelCache] Pre-loading DINOv2 gallery features ...")
        print(f"  gallery   : {gallery_dir}  ({len(gallery_files)} images)")
        print(f"  feat cache: {feat_cache_dir}")
        print("=" * 60)

        self.gallery_feats = load_or_compute_gallery_features(
            gallery_files=gallery_files,
            extractor=self.dino,
            cache_dir=feat_cache_dir,
            model_name=args.dino_model,
            nonblack_thresh=args.nonblack_thresh,
            crop_margin=args.crop_margin,
        )
        # Cache the sorted file list (avoids per-frame iterdir scan)
        self.gallery_files = gallery_files
        # Build stacked feature matrix (N, D) for vectorized cosine scoring
        self.gallery_feat_matrix = np.stack(
            [self.gallery_feats[gp.name]["feat"] for gp in gallery_files], axis=0
        )
        print(f"[ModelCache] Gallery features ready: {len(self.gallery_feats)} images  "
              f"matrix={self.gallery_feat_matrix.shape}\n")

    # ─────────────────────────────────────────────────────────────────────
    # GaussianModel (3DGS) preload for steps 6 / 7
    # ─────────────────────────────────────────────────────────────────────

    # def load_gs_model(self, args) -> None:
    #     """
    #     Load the canonical GaussianModel into GPU memory once.
    #     Requires args.gs_model_dir and args.gs_iter.
    #     After this call, steps 6 and 7 can use self.gaussians directly
    #     instead of re-loading the PLY on every frame.
    #     """
    #     import sys, os
    #     gs_repo = getattr(args, "gs_repo", None)
    #     if gs_repo and str(gs_repo) not in sys.path:
    #         sys.path.insert(0, str(gs_repo))

    #     from render_gallery import load_gaussians

    #     gs_iter = int(args.gs_iter) if getattr(args, "gs_iter", None) is not None else -1

    #     print("=" * 60)
    #     print("[ModelCache] Loading GaussianModel into GPU memory ...")
    #     print(f"  model_dir : {args.gs_model_dir}")
    #     print(f"  iteration : {gs_iter}")
    #     print("=" * 60)

    #     self.gaussians, ply_path, resolved_iter = load_gaussians(
    #         model_dir=args.gs_model_dir,
    #         iteration=gs_iter,
    #         sh_degree=3,
    #     )
    #     print(f"[ModelCache] GaussianModel loaded (iter={resolved_iter}).\n")

    def load_gs_model(self, args) -> None:
        import sys, os
        gs_repo = getattr(args, "gs_repo", None)
        if gs_repo and str(gs_repo) not in sys.path:
            sys.path.insert(0, str(gs_repo))

        from render_gallery import load_gaussians

        gs_iter = int(args.gs_iter) if getattr(args, "gs_iter", None) is not None else -1

        print("=" * 60)
        print("[ModelCache] Loading GaussianModel into GPU memory ...")
        
        self.gaussians, ply_path, resolved_iter = load_gaussians(
            model_dir=args.gs_model_dir,
            iteration=gs_iter,
            sh_degree=3,
        )

        # ====== [필수 추가] Gradient 동결 (속도 향상 및 발산 방지) ======
        import torch
        # dir() 대신 vars().values()를 사용하여 @property 접근 시 발생하는 에러 방지
        for attr in vars(self.gaussians).values():
            if isinstance(attr, torch.Tensor) and attr.requires_grad:
                attr.requires_grad_(False)
                
        # nn.Module을 상속받은 구조일 경우를 대비한 추가 안전장치
        if hasattr(self.gaussians, "parameters"):
            for p in self.gaussians.parameters():
                p.requires_grad_(False)
        # ================================================================

        # Pre-compute canonical PLY median ||xyz|| for step6 XYZ scale correction.
        # Done once here so step6 never re-loads the PLY from disk.
        xyz_np = self.gaussians.get_xyz.detach().cpu().float().numpy()
        self.ply_med_norm = float(np.median(np.linalg.norm(xyz_np, axis=1)))
        print(f"[ModelCache] ply_med_norm cached: {self.ply_med_norm:.6f}")

        print(f"[ModelCache] GaussianModel loaded & frozen (iter={resolved_iter}).\n")

    def load_gallery_poses(self, args) -> None:
        """
        Load gallery_poses.json once and index it by pose index (int) for O(1)
        lookup in step6, instead of loading + linear-scanning every frame.
        """
        import json
        pose_json_path = getattr(args, "gallery_pose_json", None)
        if not pose_json_path:
            return
        with open(pose_json_path, "r", encoding="utf-8") as f:
            gallery = json.load(f)
        self.gallery_pose_dict = {int(p["index"]): p for p in gallery["poses"]}
        print(f"[ModelCache] gallery_pose_dict cached: {len(self.gallery_pose_dict)} poses")

    def compute_xyz_scale_factor(self, args) -> None:
        """
        Compute the XYZ map scale correction factor once at startup using one
        sample gallery XYZ map.  Since the canonical PLY and gallery are fixed,
        this factor is constant across all frames — no need to recompute per frame.
        Requires load_gs_model() to have been called first (needs ply_med_norm).
        """
        from pathlib import Path

        xyz_dir = getattr(args, "gallery_xyz_dir", None)
        if not xyz_dir or self.ply_med_norm is None:
            print("[ModelCache] xyz_scale_factor: skipped (no xyz_dir or ply_med_norm)")
            return

        xyz_dir = Path(xyz_dir)
        xyz_files = sorted(p for p in xyz_dir.iterdir() if p.suffix.lower() == ".npy")
        if not xyz_files:
            print(f"[ModelCache] xyz_scale_factor: no .npy files found in {xyz_dir}")
            return

        # Use the first file — scale factor is constant across the gallery
        xyz_map = np.load(str(xyz_files[0]))
        _valid_mask = np.abs(xyz_map).sum(axis=-1) > 1e-6
        if not _valid_mask.any():
            print("[ModelCache] xyz_scale_factor: no valid pixels in sample XYZ map")
            return

        _xyz_valid = xyz_map[_valid_mask]
        _N_SAMPLE = 50_000
        if len(_xyz_valid) > _N_SAMPLE:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(_xyz_valid), _N_SAMPLE, replace=False)
            _xyz_valid = _xyz_valid[idx]

        _norms = np.linalg.norm(_xyz_valid, axis=1)
        _p5, _p95 = np.percentile(_norms, [5, 95])
        _core = _norms[(_norms >= _p5) & (_norms <= _p95)]
        if len(_core) < 100:
            print("[ModelCache] xyz_scale_factor: not enough core points")
            return

        _scale = self.ply_med_norm / float(np.median(_core))
        if 0.8 < _scale < 1.25:
            self.xyz_scale_factor = _scale
            print(f"[ModelCache] xyz_scale_factor cached: {_scale:.6f}")
        else:
            print(f"[ModelCache] xyz_scale_factor out of range ({_scale:.6f}), will compute per-frame")