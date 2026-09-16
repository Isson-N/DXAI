#!/usr/bin/env bash
# Скачивание открытых наборов без регистрации. Запуск: bash extra-data/download_open.sh
set -u
cd "$(dirname "$0")"
log(){ echo "[$(date '+%F %T')] $*"; }

log "KAUH scoliosis/spondylolisthesis spine X-ray (Mendeley xkt857dsxk, 69 MB)"
curl -sSL --retry 3 -o spine_xray_kauh_scoliosis/ImagesOriginalSize.zip \
  "https://data.mendeley.com/public-files/datasets/xkt857dsxk/files/22606bca-6b93-4e2b-a4be-9d16d6747ea1/file_downloaded" && log ok || log FAIL

log "FracAtlas (figshare 22363012, 338 MB)"
url=$(curl -s https://api.figshare.com/v2/articles/22363012 | python3 -c 'import sys,json;print(json.load(sys.stdin)["files"][0]["download_url"])')
curl -sSL --retry 3 -o fracatlas/FracAtlas.zip "$url" && log ok || log FAIL

log "TotalSegmentator small subset v2.0.1 (Zenodo 10047263, 3.24 GB)"
curl -sSL --retry 3 -o totalsegmentator_small/Totalsegmentator_dataset_small_v201.zip \
  "https://zenodo.org/records/10047263/files/Totalsegmentator_dataset_small_v201.zip?download=1" && log ok || log FAIL

log "checks"; ls -la spine_xray_kauh_scoliosis fracatlas totalsegmentator_small
for z in spine_xray_kauh_scoliosis/*.zip fracatlas/*.zip totalsegmentator_small/*.zip; do
  python3 -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); print(sys.argv[1], len(z.namelist()), 'files, testzip:', z.testzip())" "$z" || log "BAD ZIP $z"
done
log done
