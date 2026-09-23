#!/usr/bin/env python3
"""Nested OOF classifier for hip-rotation crops centred on the lesser trochanter."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dxaqc.foreign_patch import read_image, extract_patch, PatchNet  # noqa: F401,E402
from train_foreign_patches import train_fold, ensemble_scores, f1_of  # noqa: E402

POINTS = ["H", "D"]

def threshold(scores, truth):
    best, val = .5, -1.
    for t in np.unique(np.round(scores, 4)):
        v = f1_of(truth, (scores >= t).astype(int))
        if v > val: best, val = float(t), v
    return best

def auc(y, s):
    try: return float(roc_auc_score(y, s))
    except ValueError: return float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="experiments/results/hip_keypoints_full/oof_points.csv")
    ap.add_argument("--index", default="data/index/images.csv")
    ap.add_argument("--folds", default="experiments/folds.csv")
    ap.add_argument("--cnn", default="experiments/results_final/b2/oof.csv")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default="experiments/results/hip_crop")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(); seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    device = "cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device)
    idx = pd.read_csv(a.index); pts = pd.read_csv(a.points); pts = pts.drop(columns=[c for c in ("study", "study_n", "fold") if c in pts.columns]); folds = pd.read_csv(a.folds); cnn = pd.read_csv(a.cnn)[["sop_uid", "hip_pos_prob"]]
    df = idx[idx.region.eq("hip") & idx.y_pos.notna()].copy(); df = df.merge(pts, on="sop_uid", how="left").merge(cnn, on="sop_uid", how="left").merge(folds[["study","fold"]], on="study", how="left", suffixes=("", "_fold"))
    df["fold"] = df["fold_fold"].fillna(df.get("fold", np.nan)).astype(int)
    xs, ys, labels, studies, uids, cnn_scores = [], [], [], [], [], []
    for r in df.itertuples():
        im = read_image(ROOT / str(r.path)); h, w = im.shape
        tx, ty = getattr(r, "T_x_pred", np.nan), getattr(r, "T_y_pred", np.nan)
        if not (np.isfinite(tx) and np.isfinite(ty)):
            pts2 = [(getattr(r, f"{p}_x_pred", np.nan), getattr(r, f"{p}_y_pred", np.nan)) for p in POINTS]
            valid = [(x,y) for x,y in pts2 if np.isfinite(x) and np.isfinite(y)]
            tx, ty = (np.mean([x for x,_ in valid]), np.mean([y for _,y in valid])) if valid else (w/2, h/2)
        xs.append(extract_patch(im, tx, ty, a.size)); ys.append(float(r.y_pos)); labels.append(int(r.fold)); studies.append(str(r.study)); uids.append(str(r.sop_uid)); cnn_scores.append(float(r.hip_pos_prob) if np.isfinite(r.hip_pos_prob) else .5)
    X = torch.from_numpy(np.stack(xs)).float().unsqueeze(1); y = np.asarray(ys); ff = np.asarray(labels); cnn_scores = np.asarray(cnn_scores); all_folds = sorted(np.unique(ff))
    rows, choices = [], []
    def fit(mask, tag): return [train_fold(X[mask], torch.from_numpy(y[mask]).float(), torch.ones(mask.sum()), device, a.epochs, s*1000+tag) for s in seeds]
    for outer in all_folds:
        pooled = {"crop": {}, "combo": {}, "cnn": {}}
        for inner in all_folds:
            if inner == outer: continue
            models = fit((ff != outer) & (ff != inner), outer*100+inner); sel = ff == inner
            sc = ensemble_scores(models, X[sel], device); ids = np.asarray(uids)[sel]
            for uid, crop, cp in zip(ids, sc, cnn_scores[sel]): pooled["crop"][uid]=float(crop); pooled["combo"][uid]=float((crop+cp)/2); pooled["cnn"][uid]=float(cp)
        th = {k: threshold(np.array(list(v.values())), np.array([y[uids.index(uid)] for uid in v])) for k,v in pooled.items()}; choices.append({"fold":int(outer), "thresholds":th})
        models = fit(ff != outer, outer*100+99); sel = ff == outer; sc = ensemble_scores(models, X[sel], device)
        for uid, crop, cp, yy, study in zip(np.asarray(uids)[sel], sc, cnn_scores[sel], y[sel], np.asarray(studies)[sel]): rows.append({"sop_uid":uid,"study":study,"fold":int(outer),"y_true":int(yy),"score_crop":float(crop),"pred_crop":int(crop>=th["crop"]),"score_combo":float((crop+cp)/2),"pred_combo":int((crop+cp)/2>=th["combo"]),"cnn_prob":float(cp),"_pred_cnn":int(cp>=th["cnn"])})
    out=Path(a.out); out.mkdir(parents=True, exist_ok=True); frame=pd.DataFrame(rows); frame.to_csv(out/"oof.csv",index=False)
    metrics={"thresholds":choices,"crop":{"f1":f1_of(frame.y_true,frame.pred_crop),"auc":auc(frame.y_true,frame.score_crop)},"combo":{"f1":f1_of(frame.y_true,frame.pred_combo),"auc":auc(frame.y_true,frame.score_combo)}}
    cnn_th=threshold(np.asarray(cnn_scores), y); metrics["cnn"]={"threshold":cnn_th,"f1":f1_of(frame.y_true,frame._pred_cnn),"auc":auc(y,cnn_scores)}
    frame.drop(columns=["_pred_cnn"]).to_csv(out/"oof.csv",index=False)
    (out/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=True),encoding="utf-8")
if __name__ == "__main__": main()
