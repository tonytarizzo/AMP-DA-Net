#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

"$PYTHON_BIN" -u -m scripts.1_dataset_collection \
  --data-dir              "runs/datasets" \
  --save-dir              "runs/results/test/collection" \
  \
  --train-size            200 \
  --valid-size            50 \
  --test-size             50 \
  --num-test-rounds       1 \
  --within-round          random \
  \
  --model                 cifarcnn \
  --K-t                   4 \
  --min-p                 2 \
  --max-p                 2 \
  --frac-random           0.2 \
  \
  --message-split         10 \
  \
  --total-rounds          3 \
  --local-epochs          1 \
  --local-lr              0.01 \
  --max-local-samples     128 \
  \
  --optimizer             fedavg \
  --global-lr             1.0 \
  \
  --batch-size-train      32 \
  --batch-size-test       32 \
  \
  --seed                  42
