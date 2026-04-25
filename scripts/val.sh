#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   dataset=voc2012 ./scripts/val.sh
# Optional overrides via env vars:
#   COMMENT, VERSION, DEV

DATASET="${dataset:-voc2012}"
COMMENT="${COMMENT:-interpolation-main}"
VERSION="${VERSION:-1.9.2}"
DEV="${DEV:-cuda}"

export dataset="${DATASET}"

echo "[val] dataset=${DATASET}, comment=${COMMENT}, version=${VERSION}, dev=${DEV}"
python recipe_validate.py \
  --comment "${COMMENT}" \
  --version "${VERSION}" \
  --num_epoch 1 \
  --dev "${DEV}" \
  "$@"
