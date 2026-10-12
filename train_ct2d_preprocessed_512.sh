#!/bin/bash
# Train on preprocessed 2D CT slices.

# Prepare a patient-level split first:
#   python scripts/data_prepare/split_ct2d_preprocessed_8%3A2.py \
#     --plain_dir /path/to/plain_2d \
#     --enhance_dir /path/to/enhance_2d \
#     --out_dir /path/to/ct2d_splits_512 \
#     --verify_slices
SPLIT_ROOT="${SPLIT_ROOT:-ct2d_splits_512}"

NON_CONTRAST_DIR="${NON_CONTRAST_DIR:-${SPLIT_ROOT}/train/plain_2d}"
CONTRAST_DIR="${CONTRAST_DIR:-${SPLIT_ROOT}/train/enhance_2d}"
MASK_DIR="${MASK_DIR:-${SPLIT_ROOT}/train/mask}"

EVAL_NON_CONTRAST_DIR="${EVAL_NON_CONTRAST_DIR:-${SPLIT_ROOT}/val/plain_2d}"
EVAL_CONTRAST_DIR="${EVAL_CONTRAST_DIR:-${SPLIT_ROOT}/val/enhance_2d}"
EVAL_MASK_DIR="${EVAL_MASK_DIR:-${SPLIT_ROOT}/val/mask}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/ct2d}"

MODEL="JiT-CT2D-B/16"
IMG_SIZE="${IMG_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-1000}"
BLR="${BLR:-1e-4}"

echo "=========================================="
echo "Training on preprocessed 2D CT images"
echo "=========================================="

RESUME_ARGS=()
if [ -f "${OUTPUT_DIR}/checkpoint-last.pth" ]; then
  echo "Resuming from ${OUTPUT_DIR}/checkpoint-last.pth"
  RESUME_ARGS=(--resume "${OUTPUT_DIR}")
fi

export MASTER_PORT="${MASTER_PORT:-29504}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
torchrun --nproc_per_node="${NPROC_PER_NODE}" --nnodes=1 --node_rank=0 --master_port="${MASTER_PORT}" \
main_ct2d.py \
--model "${MODEL}" \
--img_size "${IMG_SIZE}" \
--non_contrast_dir "${NON_CONTRAST_DIR}" \
--contrast_dir "${CONTRAST_DIR}" \
--eval_non_contrast_dir "${EVAL_NON_CONTRAST_DIR}" \
--eval_contrast_dir "${EVAL_CONTRAST_DIR}" \
--batch_size "${BATCH_SIZE}" \
--blr "${BLR}" \
--epochs "${EPOCHS}" \
--warmup_epochs 5 \
--P_mean -0.8 --P_std 0.8 \
--noise_scale 1.0 \
--output_dir "${OUTPUT_DIR}" \
"${RESUME_ARGS[@]}" \
--online_eval \
--eval_freq 20 \
--condition_fusion add \
--log_freq 50 \
--use_preprocessed \
--mask_dir "${MASK_DIR}" \
--eval_mask_dir "${EVAL_MASK_DIR}"

echo "Training complete."
