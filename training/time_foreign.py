#!/usr/bin/env python3
"""Замер времени инференса патч-модели на снимках поясницы."""
import argparse, time
from pathlib import Path
import pandas as pd
from dxaqc.foreign_patch import ForeignPatchModel, read_image

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",default="experiments/results/foreign_patch/foreign_patch.pt"); p.add_argument("--index",default="data/index/images.csv"); args=p.parse_args()
    model=ForeignPatchModel.load(args.model,"cpu"); index=pd.read_csv(args.index)
    rows=index[index.region.astype(str).str.lower()=="spine"].head(10); times=[]
    for row in rows.itertuples():
        image=read_image(Path(row.path)); t=time.perf_counter(); model.probability(image); times.append(time.perf_counter()-t)
    print(f"среднее: {sum(times)/len(times):.4f} с/снимок; максимум: {max(times):.4f} с/снимок")
if __name__ == "__main__": main()
