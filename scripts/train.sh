#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   dataset=voc2012 ./scripts/train.sh
# Optional overrides via env vars:
#   COMMENT, VERSION, EPOCHS, DEV

DATASET="${dataset:-voc2012}"
COMMENT="${COMMENT:-interpolation-main}"
VERSION="${VERSION:-1.9.2}"
EPOCHS="${EPOCHS:-30}"
DEV="${DEV:-cuda}"

export dataset="${DATASET}"

echo "[train] dataset=${DATASET}, comment=${COMMENT}, version=${VERSION}, epochs=${EPOCHS}, dev=${DEV}"
python recipe_interpolation.py \
  --comment "${COMMENT}" \
  --version "${VERSION}" \
  --num_epoch "${EPOCHS}" \
  --dev "${DEV}" \
  "$@"
