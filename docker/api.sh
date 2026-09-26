#!/usr/bin/env bash
# Local-only batch API over HTTP. Input is read-only; outputs persist on the host.
set -euo pipefail
INPUT="$(realpath "$1")"
OUTDIR="$(realpath -m "$2")"
PORT="${3:-8000}"
TAG="${4:-dxaqc:0.1.0}"
ENGINE="${CONTAINER_ENGINE:-docker}"
DEVICE="${DEVICE:-cpu}"

if [ ! -d "$INPUT" ]; then
  echo "Нет входного каталога: $INPUT" >&2
  exit 1
fi
mkdir -p "$OUTDIR"
GPU=(); [ "$DEVICE" = cuda ] && GPU=(--gpus all)
USER_ARGS=(--user "$(id -u):$(id -g)")
if [ "$(basename "$ENGINE")" = podman ] && [ "$("$ENGINE" info --format '{{.Host.Security.Rootless}}')" = true ]; then
  USER_ARGS=(--user 0:0)
fi

"$ENGINE" run --rm -p "127.0.0.1:$PORT:8000" "${GPU[@]}" \
  "${USER_ARGS[@]}" -v "$INPUT:/input:ro" -v "$OUTDIR:/output" \
  "$TAG" serve --host 0.0.0.0 --port 8000 --input-root /input \
  --output-root /output --device "$DEVICE"
