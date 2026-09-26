#!/usr/bin/env bash
# Generate non-clinical DICOM fixtures and run the real local API.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:-dxaqc:0.1.0}"
PORT="${2:-8000}"
ENGINE="${CONTAINER_ENGINE:-docker}"
DEMO_DIR="${DEMO_DIR:-$ROOT/artifacts/demo_local}"
mkdir -p "$DEMO_DIR/input" "$DEMO_DIR/output"

USER_ARGS=(--user "$(id -u):$(id -g)")
if [ "$(basename "$ENGINE")" = podman ] && [ "$("$ENGINE" info --format '{{.Host.Security.Rootless}}')" = true ]; then
  USER_ARGS=(--user 0:0)
fi

"$ENGINE" run --rm --network none "${USER_ARGS[@]}" \
  -v "$ROOT/tools/make_demo_dicom.py:/app/make_demo_dicom.py:ro" \
  -v "$DEMO_DIR/input:/demo" --entrypoint python "$TAG" \
  /app/make_demo_dicom.py /demo

echo "Synthetic DICOMs ready. In another terminal, call:"
echo "  curl http://127.0.0.1:$PORT/health"
echo "  curl -X POST http://127.0.0.1:$PORT/predict -H 'Content-Type: application/json' -d '{\"input\":\".\",\"format\":\"csv\"}'"
echo "Predictions for these artificial images are not clinically meaningful."

bash "$ROOT/docker/api.sh" "$DEMO_DIR/input" "$DEMO_DIR/output" "$PORT" "$TAG"
