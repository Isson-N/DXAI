#!/usr/bin/env bash
# Пакетная обработка в контейнере без доступа к сети.
#   bash docker/run.sh <папка или zip с DICOM> <папка для результатов> [csv|xlsx] [тег]
set -euo pipefail
INPUT="$(realpath "$1")"
OUTDIR="$(realpath -m "$2")"
FORMAT="${3:-xlsx}"
TAG="${4:-dxaqc:0.1.0}"
mkdir -p "$OUTDIR"

if [ -d "$INPUT" ]; then
  MOUNT=(-v "$INPUT:/input:ro"); IN=/input
else
  MOUNT=(-v "$(dirname "$INPUT"):/input:ro"); IN="/input/$(basename "$INPUT")"
fi

docker run --rm --network none \
  --user "$(id -u):$(id -g)" \
  "${MOUNT[@]}" -v "$OUTDIR:/output" \
  "$TAG" predict --input "$IN" --output "/output/results.$FORMAT"
echo "результат: $OUTDIR/results.$FORMAT, ошибки: $OUTDIR/errors.csv"
