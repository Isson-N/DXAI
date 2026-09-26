#!/usr/bin/env bash
# Measure full-model CUDA inference in the built image without network access.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INPUT="$(realpath "$1")"
TAG="${2:-dxaqc:0.1.0}"
ENGINE="${CONTAINER_ENGINE:-docker}"
DEVICE=cuda
source "$(dirname "$0")/gpu_args.sh"

if [ -d "$INPUT" ]; then
  MOUNT=(-v "$INPUT:/input:ro"); IN=/input
elif [ -f "$INPUT" ]; then
  MOUNT=(-v "$(dirname "$INPUT"):/input:ro"); IN="/input/$(basename "$INPUT")"
else
  echo "Нет входного набора: $INPUT" >&2
  exit 1
fi

USER_ARGS=(--user "$(id -u):$(id -g)")
if [ "$(basename "$ENGINE")" = podman ] && [ "$("$ENGINE" info --format '{{.Host.Security.Rootless}}')" = true ]; then
  USER_ARGS=(--user 0:0)
fi

"$ENGINE" run --rm --network none "${GPU[@]}" "${USER_ARGS[@]}" \
  "${MOUNT[@]}" -v "$ROOT/tools/benchmark_gpu.py:/app/benchmark_gpu.py:ro" \
  --entrypoint python "$TAG" /app/benchmark_gpu.py --input "$IN" --models /app/models
