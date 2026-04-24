import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(description="6D pose pipeline")

    p.add_argument(
        "--stage",
        required=True,
        choices=["step1", "step2", "step3", "step4", "step5", "step6", "step7"],
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
    # Step 2
    # -------------------------
    p.add_argument("--gs_model_dir", type=str, default=None,
                   help="Original GS model directory, e.g. pepsi_painted")
    p.add_argument("--canonical_model_dir", type=str, default=None,
                   help="Output canonicalized GS model directory")
    p.add_argument("--gs_iter", type=int, default=None,
                   help="Target iteration number. If omitted, use latest iteration")
    p.add_argument("--axis_method", type=str, default="pca", choices=["pca"])
    p.add_argument("--up_axis", type=str, default="z", choices=["x", "y", "z"])
    p.add_argument("--flip_axis", type=str, default="auto", choices=["auto", "keep", "flip"])
    p.add_argument("--apply_scale", type=int, default=0)
    p.add_argument("--target_height_m", type=float, default=0.125)
    p.add_argument("--target_radius_m", type=float, default=0.0325)
    p.add_argument("--preview_before_after_path", type=str, default=None)
    p.add_argument("--max_preview_points", type=int, default=25000)
    p.add_argument("--elevations_deg", type=str, default="20,45,70")
    p.add_argument("--azimuth_step_deg", type=float, default=45.0)
    p.add_argument("--radius", type=str, default="1.0",
                   help="Single float or comma-separated list, e.g. '0.25,0.35,0.45'")
    p.add_argument("--roll_angles_deg", type=str, default="0",
                   help="In-plane roll angles, e.g. '-15,0,15'")
    p.add_argument("--look_at", type=str, default="0,0,0")
    p.add_argument("--up_hint", type=str, default="0,0,1")
    p.add_argument("--preview_size", type=int, default=900)

    # -------------------------
    # Step 3
    # -------------------------
    p.add_argument("--gallery_pose_json", type=str, default=None)
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--intrinsics_path", type=str, default=None)
    p.add_argument("--render_width", type=int, default=512)
    p.add_argument("--render_height", type=int, default=512)
    p.add_argument("--render_backend", type=str, default="preview_ply", choices=["preview_ply"])
    p.add_argument("--point_size", type=int, default=2)
    p.add_argument("--max_points", type=int, default=120000)
    p.add_argument("--bg_color", type=str, default="0,0,0")

    # -------------------------
    # Step 4 (GS rendering)
    # -------------------------
    p.add_argument("--gs_mode", type=str, default="3dgs", choices=["3dgs", "2dgs"])
    p.add_argument("--gs_repo", type=str, default=None)
    p.add_argument("--gs_python", type=str, default=sys.executable)
    p.add_argument("--gs_output_dir", type=str, default=None)
    p.add_argument("--save_depth", action="store_true")
    p.add_argument("--save_xyz", action="store_true")
    p.add_argument("--depth_dir", type=str, default=None)
    p.add_argument("--depth_vis_dir", type=str, default=None)
    p.add_argument("--xyz_dir", type=str, default=None)

    # -------------------------
    # Step 5 (similarity)
    # -------------------------
    p.add_argument("--query_masked_path", type=str, default=None)
    p.add_argument("--gallery_dir", type=str, default=None)
    p.add_argument("--sim_method", type=str, default="dino", choices=["dino", "loftr", "dino_loftr"])
    p.add_argument("--dino_model", type=str, default="dinov2_vits14")
    p.add_argument("--dino_input_size", type=int, default=224)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--crop_margin", type=int, default=12)
    p.add_argument("--nonblack_thresh", type=int, default=8)
    p.add_argument("--loftr_pretrained", type=str, default="outdoor", choices=["outdoor", "indoor"])
    p.add_argument("--loftr_conf_thresh", type=float, default=0.5)
    p.add_argument("--loftr_ransac_thresh", type=float, default=3.0)
    p.add_argument("--dino_scores_json", type=str, default=None) 
    p.add_argument("--step1_json", type=str, default=None)
    p.add_argument("--object_height_m", type=float, default=0.125)
    p.add_argument("--axis_len_m", type=float, default=0.04)
    p.add_argument("--canonical_ply_path", type=str, default=None,
               help="Path to canonical point_cloud.ply for initial pose visualization")

    # -------------------------
    # Step 6 (translation refinement)
    # -------------------------
    p.add_argument("--skip_t_refine", action="store_true",
                   help="Skip post-PnP translation refinement entirely")
    p.add_argument("--t_refine_iou_thresh", type=float, default=0.30,
                   help="Min IoU for accepting refined t (below → keep PnP t)")
    p.add_argument("--no_pnp_ransac", action="store_true",
                   help="Disable RANSAC in PnP: use all correspondences with ITERATIVE solvePnP")
    p.add_argument("--no_pnp", action="store_true",
                   help="Skip PnP entirely: use R_gallery + bbox-based t_init → t_refine only")
    p.add_argument("--query_mask_path", type=str, default=None)
    p.add_argument("--step5_json", type=str, default=None)
    p.add_argument("--gallery_rgb_dir", type=str, default=None)
    p.add_argument("--gallery_xyz_dir", type=str, default=None)
    p.add_argument("--gallery_meta_json", type=str, default=None)
    p.add_argument("--pnp_reproj_error", type=float, default=4.0)
    p.add_argument("--query_top_exclude_ratio", type=float, default=0.18)
    p.add_argument("--render_top_exclude_ratio", type=float, default=0.18)
    p.add_argument("--pre_pnp_ransac_mode", type=str, default="homography",
                   choices=["homography", "fundamental", "none"])
    p.add_argument("--pre_pnp_ransac_thresh", type=float, default=3.0)

    # -------------------------
    # Step 7 (pose refinement)
    # ------------------------- 
    p.add_argument("--initial_pose_json", type=str, default=None)
    p.add_argument("--refine_iters", type=int, default=100)
    p.add_argument("--lr_rot", type=float, default=1e-2)
    p.add_argument("--lr_trans", type=float, default=5e-3)
    p.add_argument("--l1_weight", type=float, default=0.7)
    p.add_argument("--ssim_weight", type=float, default=0.3)
    p.add_argument("--refine_crop_size", type=int, default=320)
    p.add_argument("--refine_crop_margin_scale", type=float, default=1.3)
    p.add_argument("--refine_warmup_steps", type=int, default=10)
    p.add_argument("--refine_early_stop_steps", type=int, default=20)
    p.add_argument("--refine_early_stop_thresh", type=float, default=1e-5)
    return p


def main():
    args = build_parser().parse_args()

    if args.stage == "step1":
        from modules_6d.step1_query_extraction import run_step1_query_extraction
        run_step1_query_extraction(args)

    elif args.stage == "step2":
        from modules_6d.canonicalize_gs_model import run_step2_canonicalize_gs_model
        run_step2_canonicalize_gs_model(args)

    elif args.stage == "step3":
        from modules_6d.gallery_pose import run_step3_gallery_pose
        run_step3_gallery_pose(args)

    elif args.stage == "step4":
        from modules_6d.render_gallery_gs import run_step4_render_gallery_gs
        run_step4_render_gallery_gs(args)

    elif args.stage == "step5":
        if args.sim_method == "dino":
            from modules_6d.retrieval_dino import run_step5_dino_retrieval
            run_step5_dino_retrieval(args)
        elif args.sim_method == "dino_loftr":
            from modules_6d.retrieval_dino_loftr import run_step5_dino_loftr_rerank
            run_step5_dino_loftr_rerank(args)
        else:
            raise NotImplementedError("Use dino first, then dino_loftr. Pure loftr-only can be added later.")
        
    elif args.stage == "step6":
        from modules_6d.step6_translation import run_step6_translation
        run_step6_translation(args)
        

    elif args.stage == "step7":
        from modules_6d.refine_pose_gs import run_step6_refine_pose_gs
        run_step6_refine_pose_gs(args)

    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()