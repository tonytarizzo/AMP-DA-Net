#!/usr/bin/env bash
set -euo pipefail

GLOBAL_MODEL=smallvgg RUN_LABEL=vgg \
  bash "$(dirname "$0")/02_pretrain_resnet.sh"
