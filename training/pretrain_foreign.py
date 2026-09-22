#!/usr/bin/env python3
"""Предобучение PatchNet на object-CXR патчах."""
from __future__ import annotations
import argparse
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from train_foreign_patches import train_fold, window_scores

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--patches",required=True); ap.add_argument("--out",required=True); ap.add_argument("--epochs",type=int,default=15); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--val-fraction",type=float,default=.1); a=ap.parse_args()
    d=np.load(a.patches,allow_pickle=True); images=np.asarray(d["image"]).astype(str); rng=np.random.default_rng(a.seed); uniq=rng.permutation(np.unique(images)); nv=max(1,int(len(uniq)*a.val_fraction)); valset=set(uniq[:nv]); val=np.array([x in valset for x in images]);
    x=torch.from_numpy(d["patches"]).float().unsqueeze(1); y=torch.from_numpy(d["labels"]).float(); w=torch.ones(len(y)); device="cuda" if torch.cuda.is_available() else "cpu"
    model=train_fold(x[~val],y[~val],w[~val],device,a.epochs,a.seed); scores=window_scores(model,x[val],device)
    print(f"val AUC: {roc_auc_score(y[val].numpy(),scores):.4f}"); torch.save(model.state_dict(),a.out)
if __name__ == "__main__": main()
