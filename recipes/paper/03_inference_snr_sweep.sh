#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

: "${DECODER_PATH:?Set DECODER_PATH to the trained .pth decoder}"
: "${CODEBOOK_PATH:?Set CODEBOOK_PATH to the paired _cb.pt file}"

PAPER_ROOT="${PAPER_ROOT:-runs/paper}"
GLOBAL_MODEL="${GLOBAL_MODEL:-resnet}"
RUN_LABEL="${RUN_LABEL:-resnet_comm_resnet}"
SNR_LIST="${SNR_LIST:-0 3 5 10 15 20}"

for SNR in $SNR_LIST; do
  "$PYTHON_BIN" -u -m scripts.3_feel_inference \
    --data-dir "$PAPER_ROOT/datasets" \
    --save-dir "$PAPER_ROOT/results/inference/$RUN_LABEL/snr_${SNR}db" \
    --model "$GLOBAL_MODEL" \
    --batch-size-train 16 \
    --batch-size-test 16 \
    --frac-random 0.2 \
    --K-t 40 \
    --min-p 7 \
    --max-p 13 \
    --total-rounds 1000 \
    --local-epochs 3 \
    --local-batch-size 20 \
    --local-lr 0.01 \
    --optimizer fedavg \
    --global-lr 1.0 \
    --message-split 20 \
    --n 128 \
    --dim 64 \
    --snr-db "$SNR" \
    --snr-mode range \
    --snr-min-db 0.0 \
    --snr-max-db 20.0 \
    --post-rounding \
    --post-processing paper \
    --grad-accumulation-size 512 \
    --code-order pop \
    --agg-method average \
    --corruption-frac 0.0 \
    --use-error-feedback \
    --kmeans-max-iter 0 \
    --curv-mode none \
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
    --decoder-path "$DECODER_PATH" \
    --codebook-path "$CODEBOOK_PATH" \
    --codebook-init gaussian \
    --codebook-trainable \
    --within-round random \
    --quant-type kmeans_plusplus \
    --lambda-q 0.0 \
    --early-stopping-patience 50 \
    --max-hours "${MAX_HOURS:-17}" \
    --seed 42
done
