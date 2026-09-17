#!/usr/bin/env bash
# Smoke-тест контейнера (план v2, этап 1): сборка → запуск без сети на data/test → проверка формата.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="dxaqc:smoke"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

bash "$ROOT/docker/build.sh" "$TAG"
bash "$ROOT/docker/run.sh" "$ROOT/data/test" "$OUT" csv "$TAG"

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
print(f"smoke-тест пройден: {len(rows)} строк")
EOF
