#!/usr/bin/env bash
# Сборка, запуск без сети и проверка формата на тестовом наборе.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="dxaqc:smoke"
INPUT="${1:-$ROOT/data/test}"
if [ ! -d "$INPUT" ] && [ ! -f "$INPUT" ]; then
  echo "Нет входного набора: $INPUT" >&2
  exit 1
fi
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

bash "$ROOT/docker/build.sh" "$TAG"
"${CONTAINER_ENGINE:-docker}" run --rm --network none --entrypoint python "$TAG" -c '
from dxaqc.service_model import load
model = load("/app/models")
assert model.keypoints is not None, model.notes
assert model.cnn is not None, model.notes
assert model.foreign is not None, model.notes
assert model.hip_rotation is not None, model.notes
print("Все четыре модели загружены")
'
bash "$ROOT/docker/run.sh" "$INPUT" "$OUT" csv "$TAG"

python3 - "$OUT/results.csv" <<'EOF'
import csv, sys
expected = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
            "quality_prob", "violation_type", "processing_status", "time_of_processing"]
regions = {"Поясничный отдел позвоночника", "Проксимальный отдел бедра"}
with open(sys.argv[1], encoding="utf-8") as f:
    reader = csv.DictReader(f)
    assert reader.fieldnames == expected, reader.fieldnames
    rows = list(reader)
assert rows, "пустой результат"
for r in rows:
    assert r["processing_status"] in {"Success", "Failure"}, r
    if r["processing_status"] == "Success":
        assert r["anatomical_region"] in regions, r
        assert r["quality_class"] in {"0", "1"}, r
        assert 0.0 <= float(r["quality_prob"]) <= 1.0, r
        assert float(r["time_of_processing"]) >= 0, r
# Раньше тест принимал сервис без моделей и строки с Failure: проверялся только формат.
failed = [r["path_to_study"] for r in rows if r["processing_status"] != "Success"]
assert not failed, f"снимки с ошибкой обработки: {failed}"
print(f"smoke-тест пройден: {len(rows)} строк, все Success")
EOF
