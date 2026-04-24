import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(description="6D pose pipeline — real-time version")

    p.add_argument(
        "--stage",
        required=True,
        choices=["step1", "step5", "step6", "step7", "result_visualize"],
    )

    # -------------------------
    # Common
    # -------------------------
    p.add_argument("--out_dir", type=str, required=True)

    # -------------------------
    # Step 1
    # -------------------------
    p.add_argument("--query_img", type=str, default=None)
    p.add_argument("--yolo_weights", type=str, default=None)
    p.add_argument("--sam2_checkpoint", type=str, default=None)
    p.add_argument("--sam2_repo", type=str, default=None)
    p.add_argument("--sam2_config", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--yolo_conf", type=float, default=0.25)
    p.add_argument("--bbox_margin", type=int, default=10)
    p.add_argument("--use_manual_fallback", action="store_true")

    # -------------------------
    # Step 5 (DINOv2 + LoFTR retrieval)
    # -------------------------
    p.add_argument("--query_masked_path", type=str, default=None)
    p.add_argument("--gallery_dir", type=str, default=None)
    p.add_argument("--dino_model", type=str, default="dinov2_vits14")
    p.add_argument("--dino_input_size", type=int, default=224)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--crop_margin", type=int, default=12)
    p.add_argument("--nonblack_thresh", type=int, default=8)
    p.add_argument("--loftr_pretrained", type=str, default="outdoor", choices=["outdoor", "indoor"])
    p.add_argument("--loftr_conf_thresh", type=float, default=0.4)
    p.add_argument("--loftr_ransac_thresh", type=float, default=200.0)
    p.add_argument("--dino_scores_json", type=str, default=None)
    p.add_argument("--dino_cache_dir",  type=str, default=None,
                   help="Directory for DINOv2 gallery feature cache (.npy). "
                        "Defaults to <gallery_dir>/../dino_cache_ds")

    # -------------------------
    # Step 6 (PnP + translation refinement)
    # -------------------------
    p.add_argument("--gallery_pose_json", type=str, default=None)
    p.add_argument("--gallery_xyz_dir", type=str, default=None)
    p.add_argument("--intrinsics_path", type=str, default=None)
    p.add_argument("--query_mask_path", type=str, default=None)
    p.add_argument("--pnp_reproj_error", type=float, default=4.0)
    p.add_argument("--object_height_m", type=float, default=0.125)
    p.add_argument("--axis_len_m", type=float, default=0.04)
    p.add_argument("--canonical_ply_path", type=str, default=None)
    p.add_argument("--skip_t_refine", action="store_true")
    p.add_argument("--t_refine_iou_thresh", type=float, default=0.30)
    p.add_argument("--no_pnp_ransac", action="store_true")
    p.add_argument("--no_pnp", action="store_true")

    # -------------------------
    # GS (shared by step 6 t-refine and step 7)
    # -------------------------
    p.add_argument("--gs_model_dir", type=str, default=None)
    p.add_argument("--gs_iter", type=int, default=None)
    p.add_argument("--gs_repo", type=str, default=None)
    p.add_argument("--gs_python", type=str, default=sys.executable)
    p.add_argument("--gs_output_dir", type=str, default=None)
    p.add_argument("--gs_mode", type=str, default="2dgs", choices=["2dgs", "3dgs"])
    p.add_argument("--render_width", type=int, default=3840)
    p.add_argument("--render_height", type=int, default=2160)
    p.add_argument("--bg_color", type=str, default="0,0,0")

    # -------------------------
    # Step 7 (GS pose refinement)
    # -------------------------
    p.add_argument("--initial_pose_json", type=str, default=None)
    p.add_argument("--refine_iters", type=int, default=1000)
    p.add_argument("--lr_rot", type=float, default=1e-2)
    p.add_argument("--lr_trans", type=float, default=5e-3)
    p.add_argument("--refine_crop_size", type=int, default=320)
    p.add_argument("--refine_crop_margin_scale", type=float, default=1.3)
    p.add_argument("--refine_warmup_steps", type=int, default=10)
    p.add_argument("--refine_early_stop_steps", type=int, default=100)
    p.add_argument("--refine_early_stop_thresh", type=float, default=1e-7)

    # -------------------------
    # Result visualize
    # -------------------------
    p.add_argument("--obj_width", type=float, default=0.095,
                   help="Object bounding box width in metres")
    p.add_argument("--obj_height", type=float, default=0.155,
                   help="Object bounding box height in metres")
    p.add_argument("--obj_depth", type=float, default=0.095,
                   help="Object bounding box depth in metres")

    return p


def main():
    args = build_parser().parse_args()

    if args.stage == "step1":
        from modules_6d_rt.step1_query_extraction_rt import run_step1_query_extraction_rt
        run_step1_query_extraction_rt(args)

    elif args.stage == "step5":
        from modules_6d_rt.retrieval_dino_loftr_rt import run_step4_dino_loftr_rerank_rt
        run_step4_dino_loftr_rerank_rt(args)

    elif args.stage == "step6":
        from modules_6d_rt.step6_translation_rt import run_step6_translation_rt
        run_step6_translation_rt(args)

    elif args.stage == "step7":
        from modules_6d_rt.refine_pose_gs_rt import run_step6_refine_pose_gs_rt
        run_step6_refine_pose_gs_rt(args)

    elif args.stage == "result_visualize":
        from modules_6d_rt.result_visualize_rt import run_result_visualize_rt
        run_result_visualize_rt(args)

    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
