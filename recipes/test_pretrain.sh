#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

"$PYTHON_BIN" -u -m scripts.2_pretraining \
  --dataset-dir           "runs/datasets/iFed_datasets" \
  --save-dir              "runs/results/test/pretrain" \
  --ckpt-dir              "runs/checkpoints/test" \
  \
  --pt-epochs             2 \
  --pt-batch-size         32 \
  --pt-patience           10 \
  \
  --model                 cifarcnn \
  --code-order            pop \
  --within-round          random \
  \
  --K-t                   4 \
  --min-p                 2 \
  --max-p                 2 \
  \
  --message-split         10 \
  --n                     64 \
  --dim                   32 \
  --snr-mode              range \
  --snr-min-db            0.0 \
  --snr-max-db            20.0 \
  --test-snrs             '20' \
  \
  --num-layers            5 \
  --num-filters           16 \
  --kernel-size           3 \
  \
  --codebook-init         gaussian \
  --codebook-trainable    \
  --quant-type            kmeans_plusplus \
  --kmeans-max-iter       0 \
  --curv-mode             none \
  \
  --amp-lr                0.001 \
  --lambda-sparse         0.001 \
  --lambda-w              0.001 \
  --lambda-k              0.01 \
  --lambda-q              0.0 \
  \
  --seed                  42
