#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

PAPER_ROOT="${PAPER_ROOT:-runs/paper}"
GLOBAL_MODEL="${GLOBAL_MODEL:-resnet}"
RUN_LABEL="${RUN_LABEL:-$GLOBAL_MODEL}"

"$PYTHON_BIN" -u -m scripts.1_dataset_collection \
  --data-dir "$PAPER_ROOT/datasets" \
  --save-dir "$PAPER_ROOT/results/collection_$RUN_LABEL" \
  --train-size 256000 \
  --valid-size 64000 \
  --test-size 20000 \
  --num-test-rounds 5 \
  --within-round random \
  --model "$GLOBAL_MODEL" \
  --K-t 40 \
  --min-p 7 \
  --max-p 13 \
  --frac-random 0.2 \
  --message-split 20 \
  --total-rounds 1000 \
  --local-epochs 3 \
  --local-lr 0.01 \
  --optimizer fedavg \
  --global-lr 1.0 \
  --momentum 0.0 \
  --beta1 0.9 \
  --beta2 0.99 \
  --eps 0 \
  --batch-size-train 16 \
  --batch-size-test 16 \
  --max-hours "${MAX_HOURS:-17}" \
  --seed 42
