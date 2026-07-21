#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

: "${RESNET_DECODER_PATH:?Set RESNET_DECODER_PATH}"
: "${RESNET_CODEBOOK_PATH:?Set RESNET_CODEBOOK_PATH}"
: "${VGG_DECODER_PATH:?Set VGG_DECODER_PATH}"
: "${VGG_CODEBOOK_PATH:?Set VGG_CODEBOOK_PATH}"

PAPER_ROOT="${PAPER_ROOT:-runs/paper}"

PAPER_ROOT="$PAPER_ROOT" GLOBAL_MODEL=smallvgg RUN_LABEL=resnet_comm_vgg SNR_LIST=10 \
  DECODER_PATH="$RESNET_DECODER_PATH" CODEBOOK_PATH="$RESNET_CODEBOOK_PATH" \
  bash recipes/paper/03_inference_snr_sweep.sh

PAPER_ROOT="$PAPER_ROOT" GLOBAL_MODEL=smallvgg RUN_LABEL=vgg_comm_vgg SNR_LIST=10 \
  DECODER_PATH="$VGG_DECODER_PATH" CODEBOOK_PATH="$VGG_CODEBOOK_PATH" \
  bash recipes/paper/03_inference_snr_sweep.sh
