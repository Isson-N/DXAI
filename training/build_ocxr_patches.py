#!/usr/bin/env python3
"""Построение размеченных патчей из object-CXR."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from dxaqc.foreign_patch import normalize, grid_starts, extract_patch

def parse_boxes(value: str):
    boxes = []
    for item in str(value or "").split(";"):
        tok = item.split()
        if len(tok) < 5:
            continue
        typ, nums = int(tok[0]), list(map(float, tok[1:]))
        if typ in (0, 1) and len(nums) >= 4:
            xs, ys = nums[0::2], nums[1::2]
        elif typ == 2 and len(nums) >= 6:
            xs, ys = nums[0::2], nums[1::2]
        else:
            continue
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=.25); ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--stride", type=int, default=32); ap.add_argument("--neg-per-image", type=int, default=8)
    ap.add_argument("--max-images", type=int, default=None); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); root = Path(a.dir)
    import pandas as pd
    df = pd.read_csv(root / "train.csv"); patches=[]; labels=[]; names=[]; nimg=0; pos=neg=0
    for _, row in df.iterrows():
        if a.max_images is not None and nimg >= a.max_images: break
        path = root / "train" / str(row.image_name)
        if not path.exists(): continue
        orig = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        h,w = orig.shape; sh,sw = max(1, round(h*a.scale)), max(1, round(w*a.scale))
        arr = np.asarray(Image.fromarray(orig).resize((sw,sh), Image.Resampling.BILINEAR), dtype=np.float32)
        arr = normalize(arr); boxes=[(x*a.scale,y*a.scale,X*a.scale,Y*a.scale) for x,y,X,Y in parse_boxes(row.annotation)]
        candidates=[]; positives=[]
        for y in grid_starts(sh,a.size,a.stride):
            for x in grid_starts(sw,a.size,a.stride):
                x2,y2=x+a.size,y+a.size; area=a.size*a.size; inter=[]
                for bx,by,bX,bY in boxes:
                    ix=max(0,min(x2,bX)-max(x,bx)); iy=max(0,min(y2,bY)-max(y,by)); inter.append(ix*iy)
                if inter and max(inter)/area >= .5: positives.append((x,y,1))
                elif not inter or max(inter)==0: candidates.append((x,y,0))
        chosen = candidates if len(candidates)<=a.neg_per_image else [candidates[i] for i in rng.choice(len(candidates),a.neg_per_image,replace=False)]
        for x,y,l in positives+chosen:
            patches.append(extract_patch(arr,x+a.size/2,y+a.size/2,a.size)); labels.append(l); names.append(str(row.image_name))
        pos += len(positives); neg += len(chosen); nimg += 1
    np.savez_compressed(a.out, patches=np.asarray(patches,np.float32), labels=np.asarray(labels,np.int8), image=np.asarray(names))
    print(f"картинок: {nimg}, положительных окон: {pos}, отрицательных: {neg}")
if __name__ == "__main__": main()
