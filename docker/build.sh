#!/usr/bin/env bash
# Сборка образа. Запуск из любой папки: bash docker/build.sh [тег]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:-dxaqc:$(grep -m1 '^version' "$ROOT/pyproject.toml" | cut -d'"' -f2)}"
# --network host: на части хостов у docker build нет DNS; на запуск не влияет (run.sh — без сети)
docker build --network host -f "$ROOT/docker/Dockerfile" -t "$TAG" "$ROOT"
echo "собран образ: $TAG"
