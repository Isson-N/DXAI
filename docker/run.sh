#!/usr/bin/env bash
# Пакетная обработка в контейнере без доступа к сети.
#   bash docker/run.sh <папка или zip с DICOM> <папка для результатов> [csv|xlsx] [тег]
set -euo pipefail
INPUT="$(realpath "$1")"
OUTDIR="$(realpath -m "$2")"
FORMAT="${3:-xlsx}"
TAG="${4:-dxaqc:0.1.0}"
ENGINE="${CONTAINER_ENGINE:-docker}"
mkdir -p "$OUTDIR"

if [ -d "$INPUT" ]; then
  MOUNT=(-v "$INPUT:/input:ro"); IN=/input
else
  MOUNT=(-v "$(dirname "$INPUT"):/input:ro"); IN="/input/$(basename "$INPUT")"
fi

# DEVICE=cuda enables GPU; GPU_MODE=manual passes devices on hosts without toolkit integration.
DEVICE="${DEVICE:-cpu}"
source "$(dirname "$0")/gpu_args.sh"
USER_ARGS=(--user "$(id -u):$(id -g)")
if [ "$(basename "$ENGINE")" = podman ] && [ "$("$ENGINE" info --format '{{.Host.Security.Rootless}}')" = true ]; then
  USER_ARGS=(--user 0:0)
fi
"$ENGINE" run --rm --network none "${GPU[@]}" \
  "${USER_ARGS[@]}" \
  "${MOUNT[@]}" -v "$OUTDIR:/output" \
  "$TAG" predict --input "$IN" --output "/output/results.$FORMAT" --device "$DEVICE"
echo "результат: $OUTDIR/results.$FORMAT, ошибки: $OUTDIR/errors.csv"
