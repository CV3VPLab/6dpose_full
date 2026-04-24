#!/bin/bash
set -e

export TORCH_CUDA_ARCH_LIST="8.6"
# =========================================================
# run_6d_ds.sh  –  same pipeline as run_6d.sh but with
# downsampled (DS) inputs: 3840x2160 → 1920x1080 (scale 0.5)
#
# Step selection:
#   ds_prep : downsample query image + generate scaled intrinsics
#   step1   : YOLO + SAM2 query extraction (on DS image)
#   step2   : GS-aware canonicalization
#   step3   : ring/elevation gallery pose generation
#   step4   : GS gallery render  (1920x1080)
#   step5   : DINO+LoFTR retrieval
#   step6   : translation refinement + PnP
#   step7   : pose refinement (GS, 1920x1080)
# =========================================================
STAGE="step5"

# ---------------------
# Downsampling factor
# ---------------------
DS_SCALE="0.5"
DS_RENDER_WIDTH="1920"
DS_RENDER_HEIGHT="1080"

# ---------------------
# Common
# ---------------------
GS_MODE="3dgs"
OUT_DIR="data/outputs_0424/q2_00044_${GS_MODE}_scale_ds_100iter"
DEVICE="cuda"

# ---------------------
# Step 0 – DS prep settings
# ---------------------
QUERY_IMG_ORIG="data/query_0408/q2_00044.jpg"
INTRINSICS_PATH_ORIG="data/can_data/intrinsics_4k.txt"

# Paths for the downsampled query image and scaled intrinsics
QUERY_IMG_DS="data/query_0408/q2_00044_ds.jpg"
INTRINSICS_PATH_DS="data/can_data/intrinsics_ds.txt"

# # ---------------------
# # Step 0 – DS prep settings
# # ---------------------
# QUERY_IMG_ORIG="data/query_frames_0415/frame_00005.png"   # only used for ds_prep check
# INTRINSICS_PATH_ORIG="data/can_data/intrinsics_4k.txt"

# # Paths for the downsampled query image and scaled intrinsics
# QUERY_IMG_DS="data/query_0415/frame_00005_ds.png"
# INTRINSICS_PATH_DS="data/can_data/intrinsics_ds.txt"

# Use downsampled paths from step1 onwards
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
# Step 1 settings
# ---------------------
YOLO_W="weights/best_3cls_v2.pt"
SAM2_CKPT="weights/sam2.1_hiera_tiny.pt"
SAM2_REPO="sam2_repo"
SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_t.yaml"

# ---------------------
# Step 2 settings
# ---------------------
GS_MODEL_DIR="data/can_data/${GS_MODE}_pepsi_pinset/pepsi_painted"
CANONICAL_MODEL_DIR="data/can_data/${GS_MODE}_pepsi_pinset/pepsi_painted_canonical"
GS_ITER="30000"
AXIS_METHOD="pca"
UP_AXIS="z"
FLIP_AXIS="auto"
APPLY_SCALE="1"
TARGET_HEIGHT_M="0.125"
TARGET_RADIUS_M="0.0325"
PREVIEW_BEFORE_AFTER_PATH="${OUT_DIR}/canonical_preview_before_after.png"
MAX_PREVIEW_POINTS="25000"

# ---------------------
# Step 3 settings
# ---------------------
ELEVATIONS_DEG="0,15,30,45,60,75"
AZIMUTH_STEP_DEG="20"
# 여러 radius: 가깝고 중간 거리 (실제 촬영 거리에 맞게 조정)
RADIUS="0.3,0.35,0.4,0.45"
# in-plane roll: 캔이 기울어진 경우를 커버
ROLL_ANGLES_DEG="-30,-20,-10,0,10,20,30"
LOOK_AT="0,0,0"
UP_HINT="0,0,1"
PREVIEW_SIZE="900"

# ---------------------
# Step 4 settings – GS gallery render (downsampled)
# ---------------------
GALLERY_SHARE_DIR="data/can_data"
GALLERY_POSE_JSON="${GALLERY_SHARE_DIR}/gallery_poses.json"
CANONICAL_PLY_PATH="${CANONICAL_MODEL_DIR}/point_cloud/iteration_${GS_ITER}/point_cloud.ply"
GS_REPO="$HOME/6dpose_pipeline"

GS_GALLERY_OUTPUT_DIR="${GALLERY_SHARE_DIR}/gallery_renders_gs_ds"
GS_RENDER_WIDTH="${DS_RENDER_WIDTH}"
GS_RENDER_HEIGHT="${DS_RENDER_HEIGHT}"
GS_BG_COLOR="0,0,0"

SAVE_DEPTH="1"
SAVE_XYZ="1"
GS_DEPTH_DIR="${GALLERY_SHARE_DIR}/gallery_depth_gs_ds"
GS_DEPTH_VIS_DIR="${GALLERY_SHARE_DIR}/gallery_depth_vis_gs_ds"
GS_XYZ_DIR="${GALLERY_SHARE_DIR}/gallery_xyz_gs_ds"

# ---------------------
# Step 5 settings
# ---------------------
QUERY_MASKED_PATH="${OUT_DIR}/query_masked_full.png"
GALLERY_DIR="${GALLERY_SHARE_DIR}/gallery_renders_gs_ds"
DINO_MODEL="dinov2_vits14"
DINO_INPUT_SIZE="224"
TOPK="3"
CROP_MARGIN="12"
NONBLACK_THRESH="8"
SIM_METHOD="dino_loftr"
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
T_REFINE_IOU_THRESH="0.30"
NO_PNP_RANSAC="0"
NO_PNP="0"
PNP_REPROJ_ERROR="10.0"

# ---------------------
# Step 7 settings
# ---------------------
INITIAL_POSE_JSON="${OUT_DIR}/initial_pose.json"
GS_REFINE_OUTPUT_DIR="${OUT_DIR}/refine_pose_gs_ds"
REFINE_ITERS="100"
LR_ROT="0.02"
LR_TRANS="0.001"
REFINE_CROP_SIZE="640"
REFINE_CROP_MARGIN_SCALE="1.3"
REFINE_WARMUP_STEPS="5"
REFINE_EARLY_STOP_STEPS="100"
REFINE_EARLY_STOP_THRESH="1e-7"

# =========================================================
# Stage dispatch
# =========================================================

if [ "${STAGE}" = "ds_prep" ]; then
  echo "=== [ds_prep] Downsampling query image and intrinsics ==="
  python -m modules_6d.downsample_inputs \
    --query_img      "${QUERY_IMG_ORIG}" \
    --out_img        "${QUERY_IMG_DS}" \
    --intrinsics     "${INTRINSICS_PATH_ORIG}" \
    --out_intrinsics "${INTRINSICS_PATH_DS}" \
    --scale          "${DS_SCALE}"
  echo "[ds_prep] Done."

elif [ "${STAGE}" = "step1" ]; then
  python main_6d.py \
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

elif [ "${STAGE}" = "step2" ]; then
  python main_6d.py \
    --stage step2 \
    --out_dir "${OUT_DIR}" \
    --gs_model_dir "${GS_MODEL_DIR}" \
    --canonical_model_dir "${CANONICAL_MODEL_DIR}" \
    --gs_iter "${GS_ITER}" \
    --axis_method "${AXIS_METHOD}" \
    --up_axis "${UP_AXIS}" \
    --flip_axis "${FLIP_AXIS}" \
    --apply_scale "${APPLY_SCALE}" \
    --target_height_m "${TARGET_HEIGHT_M}" \
    --target_radius_m "${TARGET_RADIUS_M}" \
    --preview_before_after_path "${PREVIEW_BEFORE_AFTER_PATH}" \
    --max_preview_points "${MAX_PREVIEW_POINTS}"

elif [ "${STAGE}" = "step3" ]; then
  python main_6d.py \
    --stage step3 \
    --out_dir "${GALLERY_SHARE_DIR}" \
    --elevations_deg "${ELEVATIONS_DEG}" \
    --azimuth_step_deg "${AZIMUTH_STEP_DEG}" \
    --radius "${RADIUS}" \
    --roll_angles_deg="${ROLL_ANGLES_DEG}" \
    --look_at "${LOOK_AT}" \
    --up_hint "${UP_HINT}" \
    --preview_size "${PREVIEW_SIZE}"

elif [ "${STAGE}" = "step4" ]; then
  python main_6d.py \
    --stage step4 \
    --out_dir "${OUT_DIR}" \
    --gs_model_dir "${CANONICAL_MODEL_DIR}" \
    --gs_iter "${GS_ITER}" \
    --gs_mode "${GS_MODE}" \
    --gallery_pose_json "${GALLERY_POSE_JSON}" \
    --intrinsics_path "${INTRINSICS_PATH}" \
    --gs_repo "${GS_REPO}" \
    --gs_output_dir "${GS_GALLERY_OUTPUT_DIR}" \
    --render_width "${GS_RENDER_WIDTH}" \
    --render_height "${GS_RENDER_HEIGHT}" \
    --bg_color "${GS_BG_COLOR}" \
    $( [ "${SAVE_DEPTH}" = "1" ] && echo --save_depth ) \
    $( [ "${SAVE_XYZ}" = "1" ] && echo --save_xyz ) \
    --depth_dir "${GS_DEPTH_DIR}" \
    --depth_vis_dir "${GS_DEPTH_VIS_DIR}" \
    --xyz_dir "${GS_XYZ_DIR}"

elif [ "${STAGE}" = "step5" ]; then
  python main_6d.py \
    --stage step5 \
    --out_dir "${OUT_DIR}" \
    --query_masked_path "${QUERY_MASKED_PATH}" \
    --gallery_dir "${GALLERY_DIR}" \
    --sim_method "${SIM_METHOD}" \
    --dino_model "${DINO_MODEL}" \
    --dino_input_size "${DINO_INPUT_SIZE}" \
    --topk "${TOPK}" \
    --crop_margin "${CROP_MARGIN}" \
    --nonblack_thresh "${NONBLACK_THRESH}" \
    --device "${DEVICE}" \
    --dino_scores_json "${DINO_SCORES_JSON}" \
    --loftr_pretrained "${LOFTR_PRETRAINED}" \
    --loftr_conf_thresh "${LOFTR_CONF_THRESH}" \
    --loftr_ransac_thresh "${LOFTR_RANSAC_THRESH}"

elif [ "${STAGE}" = "step6" ]; then
  python main_6d.py \
    --stage step6 \
    --out_dir "${OUT_DIR}" \
    --gs_mode "${GS_MODE}" \
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
    --render_width "${GS_RENDER_WIDTH}" \
    --render_height "${GS_RENDER_HEIGHT}" \
    --bg_color "${GS_BG_COLOR}" \
    --t_refine_iou_thresh "${T_REFINE_IOU_THRESH}" \
    $( [ "${SKIP_T_REFINE}" = "1" ] && echo --skip_t_refine ) \
    $( [ "${NO_PNP_RANSAC}" = "1" ] && echo --no_pnp_ransac ) \
    $( [ "${NO_PNP}" = "1" ] && echo --no_pnp )

elif [ "${STAGE}" = "step7" ]; then
  python main_6d.py \
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

else
  echo "Unsupported STAGE: ${STAGE}"
  echo "Valid stages: ds_prep  step1  step2  step3  step4  step5  step6  step7"
  exit 1
fi
