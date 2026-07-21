#!/usr/bin/env bash
set -euo pipefail

GLOBAL_MODEL=smallvgg RUN_LABEL=vgg \
  bash "$(dirname "$0")/01_collect_resnet.sh"
