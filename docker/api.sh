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
source "$(dirname "$0")/gpu_args.sh"
NETWORK=(-p "127.0.0.1:$PORT:8000")
HOST=0.0.0.0
CONTAINER_PORT=8000
if [ "${API_NETWORK:-bridge}" = host ]; then
  NETWORK=(--network host)
  HOST=127.0.0.1
  CONTAINER_PORT="$PORT"
elif [ "${API_NETWORK:-bridge}" != bridge ]; then
  echo "Unknown API_NETWORK: $API_NETWORK (expected bridge or host)" >&2
  exit 1
fi
USER_ARGS=(--user "$(id -u):$(id -g)")
if [ "$(basename "$ENGINE")" = podman ] && [ "$("$ENGINE" info --format '{{.Host.Security.Rootless}}')" = true ]; then
  USER_ARGS=(--user 0:0)
fi

"$ENGINE" run --rm "${NETWORK[@]}" "${GPU[@]}" \
  "${USER_ARGS[@]}" -v "$INPUT:/input:ro" -v "$OUTDIR:/output" \
  "$TAG" serve --host "$HOST" --port "$CONTAINER_PORT" --input-root /input \
  --output-root /output --device "$DEVICE"
