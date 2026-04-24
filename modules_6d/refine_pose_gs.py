import json
import subprocess
import sys
from pathlib import Path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_step6_refine_pose_gs(args):
    required = {
        "gs_repo":            args.gs_repo,

        "gs_model_dir":       args.gs_model_dir,
        "initial_pose_json":  args.initial_pose_json,
        "query_masked_path":  args.query_masked_path,
        "query_mask_path":    args.query_mask_path,
        "intrinsics_path":    args.intrinsics_path,
        "gs_output_dir":      args.gs_output_dir,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing required args for step6: {missing}")

    gs_repo           = Path(args.gs_repo).resolve()
    gs_python         = Path(args.gs_python if args.gs_python else sys.executable).resolve()
    gs_model_dir      = Path(args.gs_model_dir).resolve()
    initial_pose_json = Path(args.initial_pose_json).resolve()
    query_masked_path = Path(args.query_masked_path).resolve()
    query_mask_path   = Path(args.query_mask_path).resolve()
    intrinsics_path   = Path(args.intrinsics_path).resolve()
    gs_output_dir     = Path(args.gs_output_dir).resolve()

    ensure_dir(gs_output_dir)

    refine_script = gs_repo / "refine_pose.py"
    if not refine_script.exists():
        raise FileNotFoundError(f"refine_pose.py not found in GS repo: {refine_script}")

    cmd = [
        str(gs_python),
        str(refine_script),
        "--model_dir",          str(gs_model_dir),
        "--initial_pose_json",  str(initial_pose_json),
        "--query_masked_path",  str(query_masked_path),
        "--query_mask_path",    str(query_mask_path),
        "--intrinsics_path",    str(intrinsics_path),
        "--output_dir",         str(gs_output_dir),
        "--width",              str(args.render_width),
        "--height",             str(args.render_height),
        "--background",         str(args.bg_color),
        "--iteration",          str(args.gs_iter if args.gs_iter is not None else -1),
        "--sh_degree",          "3",
        "--iters",              str(args.refine_iters),
        "--lr_rot",             str(args.lr_rot),
        "--lr_trans",           str(args.lr_trans),
        "--crop_size",          str(args.refine_crop_size),
        "--crop_margin_scale",  str(args.refine_crop_margin_scale),
        "--warmup_steps",       str(args.refine_warmup_steps),
        "--early_stop_steps",   str(args.refine_early_stop_steps),
        "--early_stop_thresh",  str(args.refine_early_stop_thresh),
    ]

    print("=" * 60)
    print("[Step 7] Running GS pose refinement (v5 crop-based)")
    print("Command:")
    print(" ".join(cmd))
    print("=" * 60)

    subprocess.run(cmd, check=True, cwd=str(gs_repo))

    summary = {
        "stage":              "step6",
        "gs_repo":            str(gs_repo),
        "gs_python":          str(gs_python),
        "gs_model_dir":       str(gs_model_dir),
        "initial_pose_json":  str(initial_pose_json),
        "query_masked_path":  str(query_masked_path),
        "query_mask_path":    str(query_mask_path),
        "intrinsics_path":    str(intrinsics_path),
        "output_dir":         str(gs_output_dir),
        "iters":              args.refine_iters,
        "lr_rot":             args.lr_rot,
        "lr_trans":           args.lr_trans,
        "crop_size":          args.refine_crop_size,
        "crop_margin_scale":  args.refine_crop_margin_scale,
        "warmup_steps":       args.refine_warmup_steps,
        "early_stop_steps":   args.refine_early_stop_steps,
        "early_stop_thresh":  args.refine_early_stop_thresh,
    }
    save_json(Path(args.out_dir) / "step6_summary.json", summary)

    print("=" * 60)
    print("[Step 7] GS pose refinement complete")
    print(f"  output_dir : {gs_output_dir}")
    print("=" * 60)