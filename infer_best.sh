#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Best params from SeeSR sweep (Total Score 2.639)
# ablation_dt_only_static_s42_ckpt-30000_g6p0_c0p9_s50_adain
# ----------------------------
GPU="6"
TEST_DIR="/home/jiyang/jiyang/Projects/SeeSR/preset/datasets/contest/test_data_contest"
SAVE_DIR="./output/best_ablation_dt_static_30000_g6_c0p9_s50_adain_2"

# Best model: ablation_dt_only_static (static degradation token)
SEESR_MODEL_PATH="./model_zoo/team06_AIT/ablation_dt_only_static_s42_30000"
SEESR_USE_DEGRADATION_TOKEN="true"
SEESR_USE_DYNAMIC_DEGRADATION_TOKEN="false"

# Best inference params
SEESR_GUIDANCE_SCALE="6.0"
SEESR_CONDITIONING_SCALE="0.9"
SEESR_NUM_INFERENCE_STEPS="50"
SEESR_ALIGN_METHOD="adain"

export SEESR_MODEL_PATH
export SEESR_USE_DEGRADATION_TOKEN
export SEESR_USE_DYNAMIC_DEGRADATION_TOKEN
export SEESR_GUIDANCE_SCALE
export SEESR_CONDITIONING_SCALE
export SEESR_NUM_INFERENCE_STEPS
export SEESR_ALIGN_METHOD

cd "$(dirname "$0")"
mkdir -p "${SAVE_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

CUDA_VISIBLE_DEVICES="${GPU}" python test.py \
  --test_dir "${TEST_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --model_id 6
