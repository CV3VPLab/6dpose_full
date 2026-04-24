"""
main_6d_rt_all.py
=================
Single-process real-time pipeline: loads all models ONCE, then runs
step1 → step5 → step6 → step7 → result_visualize in sequence.

Models kept in GPU memory throughout:
  • YOLO               (step1)
  • SAM2               (step1)
  • DINOv2             (step5)
  • LoFTR              (step5)
  • DINOv2 gallery features — in RAM, no per-frame disk reads
  • GaussianModel      (steps 6 / 7 — t-refine renders + pose refinement)

Usage:
    python main_6d_rt_all.py [same args as main_6d_rt.py]

Required extra args vs. main_6d_rt.py:
    --gallery_dir   path to the pre-rendered gallery images
    (already present in run_6d_ds_rt.sh)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── gsplat warmup ──────────────────────────────────────────────────────────
# Force gsplat CUDA extension to compile/load NOW, before any large models
# are loaded into RAM.  Compilation of the backward kernels can OOM-segfault
# if done later when YOLO/SAM2/GaussianModel already occupy GPU/RAM.
def _warmup_gsplat():
    import torch
    from gsplat import rasterization
    N = 1
    _m = torch.zeros(N, 3, device="cuda")
    _q = torch.tensor([[1., 0., 0., 0.]], device="cuda")
    _s = torch.full((N, 3), 1e-4, device="cuda")
    _o = torch.ones(N, device="cuda")
    _c = torch.zeros(N, 3, device="cuda")
    _v = torch.eye(4, device="cuda").unsqueeze(0)
    _K = torch.tensor([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]], device="cuda").unsqueeze(0)
    with torch.no_grad():
        rasterization(_m, _q, _s, _o, _c, _v, _K, 100, 100, packed=False)
    print("[gsplat warmup] CUDA extension loaded OK")

_warmup_gsplat()
# ──────────────────────────────────────────────────────────────────────────


def _hms(seconds: float) -> str:
    """Format seconds as  X.XXXs  or  Xm XX.Xs  for long waits."""
    if seconds < 60:
        return f"{seconds:.3f}s"
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}m {s:.1f}s"


def build_parser():
    p = argparse.ArgumentParser(
        description="6D pose RT pipeline — all stages, single process"
    )

    # Common
    p.add_argument("--out_dir",    type=str, required=True)
    p.add_argument("--device",     type=str, default="cuda")

    # Step 1
    p.add_argument("--query_img",        type=str, default=None)
    p.add_argument("--yolo_weights",     type=str, default=None)
    p.add_argument("--sam2_checkpoint",  type=str, default=None)
    p.add_argument("--sam2_repo",        type=str, default=None)
    p.add_argument("--sam2_config",      type=str, default=None)
    p.add_argument("--yolo_conf",        type=float, default=0.25)
    p.add_argument("--bbox_margin",      type=int,   default=10)
    p.add_argument("--use_manual_fallback", action="store_true")

    # Step 5
    p.add_argument("--query_masked_path",  type=str,   default=None)
    p.add_argument("--gallery_dir",        type=str,   default=None)
    p.add_argument("--dino_model",         type=str,   default="dinov2_vits14")
    p.add_argument("--dino_input_size",    type=int,   default=224)
    p.add_argument("--topk",              type=int,   default=5)
    p.add_argument("--crop_margin",       type=int,   default=12)
    p.add_argument("--nonblack_thresh",   type=int,   default=8)
    p.add_argument("--loftr_pretrained",  type=str,   default="outdoor",
                   choices=["outdoor", "indoor"])
    p.add_argument("--loftr_conf_thresh", type=float, default=0.4)
    p.add_argument("--loftr_ransac_thresh", type=float, default=200.0)
    p.add_argument("--dino_scores_json",  type=str,   default=None)
    p.add_argument("--dino_cache_dir",   type=str,   default=None,
                   help="Directory for DINOv2 gallery feature cache (.npy). "
                        "Defaults to <gallery_dir>/../dino_cache_ds")

    # Step 6
    p.add_argument("--gallery_pose_json",   type=str,   default=None)
    p.add_argument("--gallery_xyz_dir",     type=str,   default=None)
    p.add_argument("--intrinsics_path",     type=str,   default=None)
    p.add_argument("--query_mask_path",     type=str,   default=None)
    p.add_argument("--pnp_reproj_error",    type=float, default=4.0)
    p.add_argument("--object_height_m",     type=float, default=0.125)
    p.add_argument("--axis_len_m",          type=float, default=0.04)
    p.add_argument("--canonical_ply_path",  type=str,   default=None)
    p.add_argument("--skip_t_refine",       action="store_true")
    p.add_argument("--t_refine_iou_thresh", type=float, default=0.30)
    p.add_argument("--no_pnp_ransac",       action="store_true")
    p.add_argument("--no_pnp",              action="store_true")

    # GS shared (step6 t-refine render + step7)
    p.add_argument("--gs_model_dir",   type=str,   default=None)
    p.add_argument("--gs_iter",        type=int,   default=None)
    p.add_argument("--gs_repo",        type=str,   default=None)
    p.add_argument("--gs_python",      type=str,   default=sys.executable)
    p.add_argument("--gs_output_dir",  type=str,   default=None)
    p.add_argument("--gs_mode",        type=str,   default="2dgs", choices=["2dgs", "3dgs"])
    p.add_argument("--render_width",   type=int,   default=1920)
    p.add_argument("--render_height",  type=int,   default=1080)
    p.add_argument("--bg_color",       type=str,   default="0,0,0")

    # Step 7
    p.add_argument("--initial_pose_json",        type=str,   default=None)
    p.add_argument("--refine_iters",             type=int,   default=1000)
    p.add_argument("--lr_rot",                   type=float, default=1e-2)
    p.add_argument("--lr_trans",                 type=float, default=5e-3)
    p.add_argument("--refine_crop_size",         type=int,   default=320)
    p.add_argument("--refine_crop_margin_scale", type=float, default=1.3)
    p.add_argument("--refine_warmup_steps",      type=int,   default=10)
    p.add_argument("--refine_early_stop_steps",  type=int,   default=100)
    p.add_argument("--refine_early_stop_thresh", type=float, default=1e-7)

    # Result visualize
    p.add_argument("--obj_width",  type=float, default=0.095)
    p.add_argument("--obj_height", type=float, default=0.155)
    p.add_argument("--obj_depth",  type=float, default=0.095)

    return p


def main():
    args = build_parser().parse_args()

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"

    t_total_start = time.perf_counter()
    timings = {}

    # ── 0. Load all models once ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[RT-ALL] Loading models into GPU memory")
    print("=" * 60)
    t0 = time.perf_counter()
    from modules_6d_rt.model_cache import ModelCache
    cache = ModelCache()
    cache.load_all(args, device=device)
    cache.load_gallery_features(args, gallery_dir=args.gallery_dir, device=device,
                               feat_cache_dir=args.dino_cache_dir or None)
    if args.gs_model_dir:
        cache.load_gs_model(args)
        cache.compute_xyz_scale_factor(args)
    cache.load_gallery_poses(args)
    cache.warmup_cuda(device=device)
    timings["model_load"] = time.perf_counter() - t0
    print(f"[RT-ALL] Model load time : {_hms(timings['model_load'])}")

    # ── 1. YOLO + SAM2 segmentation ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[RT-ALL] Step 1: YOLO + SAM2")
    print("=" * 60)
    from modules_6d_rt.step1_query_extraction_rt import run_step1_query_extraction_rt
    t0 = time.perf_counter()
    run_step1_query_extraction_rt(args, model_cache=cache)
    timings["step1"] = time.perf_counter() - t0
    print(f"[RT-ALL] Step 1 time : {_hms(timings['step1'])}")

    # ── 5. DINOv2 + LoFTR retrieval ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[RT-ALL] Step 5: DINOv2 + LoFTR retrieval")
    print("=" * 60)
    from modules_6d_rt.retrieval_dino_loftr_rt import run_step4_dino_loftr_rerank_rt
    t0 = time.perf_counter()
    run_step4_dino_loftr_rerank_rt(args, model_cache=cache)
    timings["step5"] = time.perf_counter() - t0
    print(f"[RT-ALL] Step 5 time : {_hms(timings['step5'])}")

    # ── 6. PnP + translation refinement ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("[RT-ALL] Step 6: PnP + translation refinement")
    print("=" * 60)
    from modules_6d_rt.step6_translation_rt import run_step6_translation_rt
    t0 = time.perf_counter()
    run_step6_translation_rt(args, model_cache=cache)
    timings["step6"] = time.perf_counter() - t0
    print(f"[RT-ALL] Step 6 time : {_hms(timings['step6'])}")

    # ── 7. GS photometric pose refinement ────────────────────────────────────
    print("\n" + "=" * 60)
    print("[RT-ALL] Step 7: GS pose refinement")
    print("=" * 60)
    # Flush all pending CUDA kernels from step 6 renders so the gsplat
    # rasterizer in step 7 starts from a clean GPU execution state.
    # Without this, tile-sorting in alpha compositing is non-deterministic
    # relative to a fresh subprocess, causing different optimization trajectories.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    from modules_6d_rt.refine_pose_gs_rt import run_step7_refine_pose_gs_rt
    t0 = time.perf_counter()
    run_step7_refine_pose_gs_rt(args, model_cache=cache)
    timings["step7"] = time.perf_counter() - t0
    print(f"[RT-ALL] Step 7 time : {_hms(timings['step7'])}")

    # ── Result visualize ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[RT-ALL] Result visualize")
    print("=" * 60)
    from modules_6d_rt.result_visualize_rt import run_result_visualize_rt
    t0 = time.perf_counter()
    run_result_visualize_rt(args, model_cache=cache)
    timings["result_visualize"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_total_start

    # ── Timing summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[RT-ALL] Timing summary")
    print("=" * 60)
    labels = [
        ("model_load",       "GPU model load"),
        ("step1",            "Step 1  YOLO + SAM2"),
        ("step5",            "Step 5  DINOv2 + LoFTR"),
        ("step6",            "Step 6  PnP + t-refine"),
        ("step7",            "Step 7  GS refinement"),
        ("result_visualize", "Result visualize"),
        ("total",            "TOTAL"),
    ]
    for key, label in labels:
        print(f"  {label:<28} {_hms(timings[key]):>10}")
    print("=" * 60)

    # Save timing JSON next to the output
    timing_path = Path(args.out_dir) / "timing.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    with open(timing_path, "w") as f:
        json.dump(timings, f, indent=2)
    print(f"[RT-ALL] Timings saved : {timing_path}")
    print(f"[RT-ALL] Result        : {args.gs_output_dir}/bbox_3d_result_rt.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
