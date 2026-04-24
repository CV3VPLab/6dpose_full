#!/bin/bash
set -e

# gsplat JIT: pin CUDA_HOME to the canonical path so build.ninja is
# identical every run (avoids .so invalidation), limit parallel nvcc
# jobs to 1 to prevent OOM segfaults during compilation.
export CUDA_HOME=/usr/local/cuda-11.8
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.6"
export FAST_COMPILE=1

# =========================================================
# run_6d_ds_rt.sh  –  Real-time 6D pose with downsampled inputs
#                     3840x2160 → 1920x1080 (scale 0.5)
#
# Per-frame stages:
#   step1            : YOLO + SAM2 query extraction (on DS image)
#   step5            : DINOv2 + LoFTR gallery retrieval & matching
#   step6            : PnP + translation refinement
#   step7            : GS photometric pose refinement (1920x1080)
#   result_visualize : draw 3D bounding box on query image
#
# Set STAGE="all" to run the full per-frame pipeline in sequence.
#
# Steps 2/3/4 (canonicalize, gallery pose generation, gallery rendering)
# are OFFLINE preprocessing — run them once via run_6d_ds.sh.
# The DS gallery must be pre-built with:
#   STAGE=step3  bash run_6d_ds.sh
#   STAGE=step4  bash run_6d_ds.sh
# =========================================================
STAGE="all"

# ---------------------
# Downsampling factor
# ---------------------
DS_SCALE="0.5"
DS_RENDER_WIDTH="1920"
DS_RENDER_HEIGHT="1080"

# ---------------------
# Common
# ---------------------
GS_MODE="3dgs"   # "3dgs" or "2dgs" — must match the gallery and canonical model below

OUT_DIR="data/outputs/rt_output_ds_0422/q2_00044_${GS_MODE}_scale_ds"

DEVICE="cuda"

# ---------------------
# Step 0 – DS prep settings
# ---------------------
QUERY_IMG_ORIG="data/query_0408/q2_00044.jpg"
INTRINSICS_PATH_ORIG="data/can_data/intrinsics_4k.txt"

# Paths for the downsampled query image and scaled intrinsics
QUERY_IMG_DS="data/query_0408/q2_00044_ds.jpg"
INTRINSICS_PATH_DS="data/can_data/intrinsics_ds.txt"

# Downsampled paths used throughout
QUERY_IMG="${QUERY_IMG_DS}"
INTRINSICS_PATH="${INTRINSICS_PATH_DS}"

# ---------------------
# Auto-run ds_prep if downsampled files are missing
# ---------------------
if [ ! -f "${QUERY_IMG_DS}" ] || [ ! -f "${INTRINSICS_PATH_DS}" ]; then
  echo "=== [auto ds_prep] Downsampled files not found – generating now ==="
  python -m modules_6d.downsample_inputs \
    --query_img      "${QUERY_IMG_ORIG}" \
    --out_img        "${QUERY_IMG_DS}" \
    --intrinsics     "${INTRINSICS_PATH_ORIG}" \
    --out_intrinsics "${INTRINSICS_PATH_DS}" \
    --scale          "${DS_SCALE}"
  echo "=== [auto ds_prep] Done ==="
fi

# ---------------------
# Gallery (pre-computed offline with run_6d_ds.sh step3+step4)
# ---------------------
GALLERY_SHARE_DIR="data/outputs/gallery_shared_pepsi_ds_580k_${GS_MODE}"
GALLERY_POSE_JSON="${GALLERY_SHARE_DIR}/gallery_poses.json"
GALLERY_DIR="${GALLERY_SHARE_DIR}/gallery_renders_gs_ds"
GS_XYZ_DIR="${GALLERY_SHARE_DIR}/gallery_xyz_gs_ds"
DINO_CACHE_DIR="${GALLERY_SHARE_DIR}/dino_cache_ds"

# ---------------------
# GS model (canonical)
# ---------------------
CANONICAL_MODEL_DIR="data/can_data/${GS_MODE}_pepsi_pinset/pepsi_painted_canonical"
GS_ITER="30000"
CANONICAL_PLY_PATH="${CANONICAL_MODEL_DIR}/point_cloud/iteration_${GS_ITER}/point_cloud.ply"

GS_REPO="$HOME/6dpose_full"
GS_RENDER_WIDTH="${DS_RENDER_WIDTH}"
GS_RENDER_HEIGHT="${DS_RENDER_HEIGHT}"
GS_BG_COLOR="0,0,0"

# ---------------------
# Step 1 settings
# ---------------------
YOLO_W="weights/best_3cls_v2.pt"
SAM2_CKPT="weights/sam2.1_hiera_tiny.pt"
SAM2_REPO="sam2_repo"
SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_t.yaml"

# ---------------------
# Step 5 settings
# ---------------------
QUERY_MASKED_PATH="${OUT_DIR}/query_masked_full.png"
DINO_MODEL="dinov2_vits14"
DINO_INPUT_SIZE="224"
TOPK="1"
CROP_MARGIN="12"
NONBLACK_THRESH="8"
DINO_SCORES_JSON="${OUT_DIR}/retrieval_scores.json"
LOFTR_PRETRAINED="outdoor"
LOFTR_CONF_THRESH="0.5"
LOFTR_RANSAC_THRESH="200.0"

# ---------------------
# Step 6 settings
# ---------------------
QUERY_MASK_PATH="${OUT_DIR}/query_mask.png"
OBJECT_HEIGHT_M="0.125"
AXIS_LEN_M="0.04"
SKIP_T_REFINE="1"
T_REFINE_IOU_THRESH="0.60"
NO_PNP_RANSAC="0"
NO_PNP="0"
PNP_REPROJ_ERROR="15.0"

# ---------------------
# Step 7 settings
# ---------------------
INITIAL_POSE_JSON="${OUT_DIR}/initial_pose.json"
GS_REFINE_OUTPUT_DIR="${OUT_DIR}/refine_pose_gs_ds"
REFINE_ITERS="50"
LR_ROT="0.02"
LR_TRANS="0.001"
REFINE_CROP_SIZE="640"
REFINE_CROP_MARGIN_SCALE="1.3"
REFINE_WARMUP_STEPS="10"
REFINE_EARLY_STOP_STEPS="100"
REFINE_EARLY_STOP_THRESH="1e-7"

# ---------------------
# Result visualize settings
# ---------------------
OBJ_WIDTH="0.095"
OBJ_HEIGHT="0.165"
OBJ_DEPTH="0.095"


# =========================================================
# Stage runners
# =========================================================

run_step1() {
  python main_6d_rt.py \
    --stage step1 \
    --query_img "${QUERY_IMG}" \
    --out_dir "${OUT_DIR}" \
    --yolo_weights "${YOLO_W}" \
    --sam2_checkpoint "${SAM2_CKPT}" \
    --sam2_repo "${SAM2_REPO}" \
    --sam2_config "${SAM2_CONFIG}" \
    --device "${DEVICE}" \
    --yolo_conf 0.25 \
    --bbox_margin 10 \
    --use_manual_fallback
}

run_step5() {
  python main_6d_rt.py \
    --stage step5 \
    --out_dir "${OUT_DIR}" \
    --query_masked_path "${QUERY_MASKED_PATH}" \
    --gallery_dir "${GALLERY_DIR}" \
    --dino_model "${DINO_MODEL}" \
    --dino_input_size "${DINO_INPUT_SIZE}" \
    --topk "${TOPK}" \
    --crop_margin "${CROP_MARGIN}" \
    --nonblack_thresh "${NONBLACK_THRESH}" \
    --device "${DEVICE}" \
    --dino_scores_json "${DINO_SCORES_JSON}" \
    --loftr_pretrained "${LOFTR_PRETRAINED}" \
    --loftr_conf_thresh "${LOFTR_CONF_THRESH}" \
    --loftr_ransac_thresh "${LOFTR_RANSAC_THRESH}" \
    --dino_cache_dir "${DINO_CACHE_DIR}"
}

run_step6() {
  python main_6d_rt.py \
    --stage step6 \
    --out_dir "${OUT_DIR}" \
    --gallery_pose_json "${GALLERY_POSE_JSON}" \
    --gallery_xyz_dir "${GS_XYZ_DIR}" \
    --intrinsics_path "${INTRINSICS_PATH}" \
    --query_img "${QUERY_IMG}" \
    --query_masked_path "${QUERY_MASKED_PATH}" \
    --query_mask_path "${QUERY_MASK_PATH}" \
    --pnp_reproj_error "${PNP_REPROJ_ERROR}" \
    --axis_len_m "${AXIS_LEN_M}" \
    --object_height_m "${OBJECT_HEIGHT_M}" \
    --canonical_ply_path "${CANONICAL_PLY_PATH}" \
    --gs_model_dir "${CANONICAL_MODEL_DIR}" \
    --gs_iter "${GS_ITER}" \
    --gs_repo "${GS_REPO}" \
    --gs_mode "${GS_MODE}" \
    --render_width "${GS_RENDER_WIDTH}" \
    --render_height "${GS_RENDER_HEIGHT}" \
    --bg_color "${GS_BG_COLOR}" \
    --t_refine_iou_thresh "${T_REFINE_IOU_THRESH}" \
    $( [ "${SKIP_T_REFINE}" = "1" ] && echo --skip_t_refine ) \
    $( [ "${NO_PNP_RANSAC}" = "1" ] && echo --no_pnp_ransac ) \
    $( [ "${NO_PNP}" = "1" ] && echo --no_pnp )
}

run_step7() {
  python main_6d_rt.py \
    --stage step7 \
    --out_dir "${OUT_DIR}" \
    --gs_model_dir "${CANONICAL_MODEL_DIR}" \
    --gs_iter "${GS_ITER}" \
    --initial_pose_json "${INITIAL_POSE_JSON}" \
    --query_masked_path "${QUERY_MASKED_PATH}" \
    --query_mask_path "${QUERY_MASK_PATH}" \
    --intrinsics_path "${INTRINSICS_PATH}" \
    --gs_repo "${GS_REPO}" \
    --gs_output_dir "${GS_REFINE_OUTPUT_DIR}" \
    --render_width "${GS_RENDER_WIDTH}" \
    --render_height "${GS_RENDER_HEIGHT}" \
    --bg_color "${GS_BG_COLOR}" \
    --refine_iters "${REFINE_ITERS}" \
    --lr_rot "${LR_ROT}" \
    --lr_trans "${LR_TRANS}" \
    --refine_crop_size "${REFINE_CROP_SIZE}" \
    --refine_crop_margin_scale "${REFINE_CROP_MARGIN_SCALE}" \
    --refine_warmup_steps "${REFINE_WARMUP_STEPS}" \
    --refine_early_stop_steps "${REFINE_EARLY_STOP_STEPS}" \
    --refine_early_stop_thresh "${REFINE_EARLY_STOP_THRESH}"
}

run_result_visualize() {
  python main_6d_rt.py \
    --stage result_visualize \
    --out_dir "${OUT_DIR}" \
    --query_img "${QUERY_IMG}" \
    --gs_output_dir "${GS_REFINE_OUTPUT_DIR}" \
    --intrinsics_path "${INTRINSICS_PATH}" \
    --obj_width "${OBJ_WIDTH}" \
    --obj_height "${OBJ_HEIGHT}" \
    --obj_depth "${OBJ_DEPTH}"
}


# =========================================================
# Dispatch
# =========================================================

if [ "${STAGE}" = "all" ]; then
  # Single-process run: all models loaded once, gallery features stay in RAM.
  # Steps 6/7 still launch GS subprocesses (different conda env), but
  # YOLO, SAM2, DINOv2, LoFTR are never reloaded between frames.
  echo "=== [RT-DS] Running full per-frame pipeline (1920x1080, preloaded models) ==="
  python main_6d_rt_all.py \
    --out_dir "${OUT_DIR}" \
    --query_masked_path "${QUERY_MASKED_PATH}" \
    --device "${DEVICE}" \
    --query_img "${QUERY_IMG}" \
    --yolo_weights "${YOLO_W}" \
    --sam2_checkpoint "${SAM2_CKPT}" \
    --sam2_repo "${SAM2_REPO}" \
    --sam2_config "${SAM2_CONFIG}" \
    --yolo_conf 0.25 \
    --bbox_margin 10 \
    --use_manual_fallback \
    --gallery_dir "${GALLERY_DIR}" \
    --dino_model "${DINO_MODEL}" \
    --dino_input_size "${DINO_INPUT_SIZE}" \
    --topk "${TOPK}" \
    --crop_margin "${CROP_MARGIN}" \
    --nonblack_thresh "${NONBLACK_THRESH}" \
    --dino_scores_json "${DINO_SCORES_JSON}" \
    --loftr_pretrained "${LOFTR_PRETRAINED}" \
    --loftr_conf_thresh "${LOFTR_CONF_THRESH}" \
    --loftr_ransac_thresh "${LOFTR_RANSAC_THRESH}" \
    --dino_cache_dir "${DINO_CACHE_DIR}" \
    --gallery_pose_json "${GALLERY_POSE_JSON}" \
    --gallery_xyz_dir "${GS_XYZ_DIR}" \
    --intrinsics_path "${INTRINSICS_PATH}" \
    --query_mask_path "${QUERY_MASK_PATH}" \
    --pnp_reproj_error "${PNP_REPROJ_ERROR}" \
    --axis_len_m "${AXIS_LEN_M}" \
    --object_height_m "${OBJECT_HEIGHT_M}" \
    --canonical_ply_path "${CANONICAL_PLY_PATH}" \
    --gs_model_dir "${CANONICAL_MODEL_DIR}" \
    --gs_iter "${GS_ITER}" \
    --gs_repo "${GS_REPO}" \
    --gs_mode "${GS_MODE}" \
    --gs_output_dir "${GS_REFINE_OUTPUT_DIR}" \
    --render_width "${GS_RENDER_WIDTH}" \
    --render_height "${GS_RENDER_HEIGHT}" \
    --bg_color "${GS_BG_COLOR}" \
    --t_refine_iou_thresh "${T_REFINE_IOU_THRESH}" \
    $( [ "${SKIP_T_REFINE}" = "1" ] && echo --skip_t_refine ) \
    $( [ "${NO_PNP_RANSAC}" = "1" ] && echo --no_pnp_ransac ) \
    $( [ "${NO_PNP}" = "1" ] && echo --no_pnp ) \
    --initial_pose_json "${INITIAL_POSE_JSON}" \
    --refine_iters "${REFINE_ITERS}" \
    --lr_rot "${LR_ROT}" \
    --lr_trans "${LR_TRANS}" \
    --refine_crop_size "${REFINE_CROP_SIZE}" \
    --refine_crop_margin_scale "${REFINE_CROP_MARGIN_SCALE}" \
    --refine_warmup_steps "${REFINE_WARMUP_STEPS}" \
    --refine_early_stop_steps "${REFINE_EARLY_STOP_STEPS}" \
    --refine_early_stop_thresh "${REFINE_EARLY_STOP_THRESH}" \
    --obj_width "${OBJ_WIDTH}" \
    --obj_height "${OBJ_HEIGHT}" \
    --obj_depth "${OBJ_DEPTH}"
  echo "=== [RT-DS] Done. Result: ${GS_REFINE_OUTPUT_DIR}/bbox_3d_result_rt.png ==="

elif [ "${STAGE}" = "step1" ]; then
  run_step1

elif [ "${STAGE}" = "step5" ]; then
  run_step5

elif [ "${STAGE}" = "step6" ]; then
  run_step6

elif [ "${STAGE}" = "step7" ]; then
  run_step7

elif [ "${STAGE}" = "result_visualize" ]; then
  run_result_visualize

else
  echo "Unsupported STAGE: ${STAGE}"
  echo "Valid values: all, step1, step5, step6, step7, result_visualize"
  exit 1
fi
