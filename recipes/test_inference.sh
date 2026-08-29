#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

"$PYTHON_BIN" -u -m scripts.3_feel_inference \
  --data-dir              "runs/datasets" \
  --save-dir              "runs/results/test/inference" \
  --ckpt-dir              "runs/checkpoints/test" \
  \
  --model                 cifarcnn \
  --K-t                   4 \
  --min-p                 2 \
  --max-p                 2 \
  --total-rounds          2 \
  --local-epochs          1 \
  --local-batch-size      32 \
  --local-lr              0.01 \
  --max-local-samples     128 \
  \
  --optimizer             fedavg \
  --global-lr             1.0 \
  \
  --message-split         10 \
  --n                     64 \
  --dim                   32 \
  --snr-db                20.0 \
  --snr-mode              range \
  --snr-min-db            0.0 \
  --snr-max-db            20.0 \
  \
  --post-rounding         \
  --post-processing       paper \
  --grad-accumulation-size 32 \
  --code-order            pop \
  \
  --agg-method            average \
  --trim-threshold        0.8 \
  --corruption-frac       0.0 \
  --use-error-feedback    \
  \
  --kmeans-max-iter       0 \
  --curv-mode             none \
  \
  --num-layers            5 \
  --num-filters           16 \
  --kernel-size           3 \
  \
  --codebook-init         gaussian \
  --codebook-trainable    \
  --within-round          random \
  --quant-type            kmeans_plusplus \
  --lambda-q              0.0 \
  \
  --batch-size-train      32 \
  --batch-size-test       32 \
  --frac-random           0.2 \
  \
  --early-stopping-patience 10 \
  --seed                  42
