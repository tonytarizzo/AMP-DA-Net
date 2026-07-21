#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

PAPER_ROOT="${PAPER_ROOT:-runs/paper}"
GLOBAL_MODEL="${GLOBAL_MODEL:-resnet}"
RUN_LABEL="${RUN_LABEL:-$GLOBAL_MODEL}"
DATASET_PATH="${DATASET_PATH:-}"
DATASET_ARGS=(--dataset-dir "$PAPER_ROOT/datasets/iFed_datasets")
if [[ -n "$DATASET_PATH" ]]; then
  DATASET_ARGS+=(--dataset-path "$DATASET_PATH")
fi

"$PYTHON_BIN" -u -m scripts.2_pretraining \
  --pt-epochs 500 \
  --pt-batch-size 64 \
  --pt-patience 20 \
  --pt-delta 1e-6 \
  --sched-patience 10 \
  --sched-factor 0.5 \
  "${DATASET_ARGS[@]}" \
  --code-order pop \
  --model "$GLOBAL_MODEL" \
  --within-round random \
  --K-t 40 \
  --min-p 7 \
  --max-p 13 \
  --message-split 20 \
  --n 128 \
  --dim 64 \
  --snr-mode range \
  --snr-min-db 0.0 \
  --snr-max-db 20.0 \
  --test-snrs '20,15,10,5,3,0' \
  --num-layers 10 \
  --num-filters 32 \
  --kernel-size 3 \
  --use-poisson \
  --update-alpha \
  --learn-sigma2 \
  --finetune-K \
  --finetune-pi \
  --finetune-sigma2 \
  --per-sample-sigma \
  --no-per-sample-alpha \
  --K-max -1 \
  --blend-init 0.85 \
  --codebook-trainable \
  --codebook-init gaussian \
  --quant-type kmeans_plusplus \
  --kmeans-max-iter 0 \
  --curv-mode none \
  --amp-lr 0.0001 \
  --lambda-sparse 0.01 \
  --lambda-w 0.001 \
  --lambda-k 0.01 \
  --lambda-q 0.0 \
  --max-hours "${MAX_HOURS:-17}" \
  --seed 42 \
  --ckpt-dir "$PAPER_ROOT/checkpoints/pretrain_$RUN_LABEL" \
  --save-dir "$PAPER_ROOT/results/pretrain_$RUN_LABEL"
