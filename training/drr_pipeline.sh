#!/bin/bash
# генерация -> склейка -> предобучение -> b2 с --init-encoder на 3 seed (параллельно на одной GPU)
set -e
cd ~/dxa; export PYTHONPATH=src:training OMP_NUM_THREADS=1
mkdir -p ~/dxa_work/drr
if [ ! -f ~/dxa_work/drr/all.npz ]; then
  for k in 0 1 2 3 4 5 6; do
    .venv/bin/python -W ignore training/build_drr_hip.py --dir ~/dxa_work/ts_all --out ~/dxa_work/drr/part$k.npz --shard $k/7 > ~/dxa_work/drr/gen_$k.log 2>&1 &
  done; wait
  .venv/bin/python - <<'PY'
import numpy as np, glob, os
p=sorted(glob.glob(os.path.expanduser("~/dxa_work/drr/part*.npz"))); d=[np.load(x) for x in p]
out={k:np.concatenate([x[k] for x in d]) for k in ("images","theta","side","ct_id")}
np.savez(os.path.expanduser("~/dxa_work/drr/all.npz"),**out); print("merged",out["images"].shape,len(set(out["ct_id"])))
PY
fi
echo "=== pretrain $(date)"
[ -f models/drr_hip_pretrain.pt ] || OMP_NUM_THREADS=4 .venv/bin/python training/pretrain_drr_hip.py --data ~/dxa_work/drr/all.npz --out models/drr_hip_pretrain.pt --epochs 15 --workers 4
.venv/bin/python -c "import torch,sys;r=torch.load('models/drr_hip_pretrain.pt',weights_only=False)['val_r'];print('val_r',r);sys.exit(0 if r>=0.5 else 1)"
echo "=== b2 with init $(date)"
for s in 42 1 2; do
  OMP_NUM_THREADS=2 .venv/bin/python training/train_baselines.py --models b2 --seed $s --min-specificity 0.8 --epochs 30 \
     --init-encoder models/drr_hip_pretrain.pt --out experiments/results_drr_seed$s > ~/dxa_work/drr/b2_$s.log 2>&1 &
done; wait
echo "=== compare $(date)"
for s in 42:results_final 1:results_seed1 2:results_seed2; do
  n=${s%%:*}; base=${s##*:}
  echo "--- seed $n"; .venv/bin/python training/ensemble_cnn_heads.py experiments/$base/b2/oof.csv experiments/results_drr_seed$n/b2/oof.csv
done
echo "=== done $(date)"
