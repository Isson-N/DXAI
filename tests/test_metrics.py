import math

import numpy as np
import pandas as pd

from dxaqc import cv
from dxaqc.metrics import binary_metrics, cluster_bootstrap_ci, macro_f1


def test_binary_metrics_counts():
    m = binary_metrics([1, 1, 0, 0], [1, 0, 0, 1], [0.9, 0.4, 0.2, 0.6])
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 1, 1, 1)
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5 and m["f1"] == 0.5
    assert m["roc_auc"] == 0.75


def test_auc_undefined_for_single_class():
    m = binary_metrics([0, 0, 0], [0, 1, 0], [0.1, 0.9, 0.2])
    assert math.isnan(m["roc_auc"]) and math.isnan(m["sensitivity"])


def test_macro_f1_is_plain_mean_over_violations():
    per_violation = {
        "spine_positioning": ([1, 0], [1, 0]),   # 1.0
        "spine_axis": ([1, 0], [0, 0]),          # 0.0
        "hip_roi": ([1, 1], [1, 1]),             # 1.0
    }
    assert math.isclose(macro_f1(per_violation), 2 / 3)


def test_bootstrap_skips_degenerate_replicates():
    groups = np.array([0, 1, 2, 3])
    y = np.array([1, 0, 0, 0])
    s = np.array([0.9, 0.1, 0.2, 0.3])

    def auc(idx):
        return binary_metrics(y[idx], (s[idx] > 0.5).astype(int), s[idx])["roc_auc"]

    ci = cluster_bootstrap_ci(groups, auc, n_boot=300, seed=1)
    assert ci["estimate"] == 1.0
    assert 0 < ci["skipped_share"] < 1


def _toy_index(n_studies=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_studies):
        for region in ("spine", "hip"):
            y = {"y_pos": int(rng.random() < 0.15), "y_axis": int(rng.random() < 0.15),
                 "y_foreign": int(rng.random() < 0.15), "y_roi": int(rng.random() < 0.1)}
            if region == "spine":
                y["y_roi"] = np.nan
            else:
                y["y_axis"] = y["y_foreign"] = np.nan
            q = int(any(v == 1 for v in y.values()))
            rows.append({"study_n": i, "study": f"s{i}", "region": region, "side": "R" if region == "hip" else np.nan,
                         "labeled": True, "quality_class": q, **y})
    return pd.DataFrame(rows)


def test_folds_keep_studies_together_and_cover_all():
    df = _toy_index()
    folds = cv.assign_folds(df, seed=3)
    assert set(folds) == set(range(cv.N_FOLDS))
    assert (df.assign(f=folds).groupby("study").f.nunique() == 1).all()


def test_feasibility_inner_parts_exclude_outer_and_inner_fold():
    df = _toy_index()
    folds = cv.assign_folds(df, seed=3)
    feas = cv.feasibility(df, folds, pd.DataFrame(columns=["target", "study", "region", "side"]))
    row = feas[(feas.target == "spine_any") & (feas.part == "inner_train")
               & (feas.outer_fold == 0) & (feas.inner_fold == 1)].iloc[0]
    expected = df[(df.region == "spine") & ~folds.isin([0, 1])]
    assert row.pos_images + row.neg_images == len(expected)


def test_policy_masks_only_listed_image():
    df = _toy_index()
    policy = pd.DataFrame([{"study": "s0", "region": "spine", "side": np.nan, "target": "spine_axis", "train_mask": 1},
                           {"study": "s1", "region": "spine", "side": np.nan, "target": "spine_axis", "train_mask": 0}])
    masked = cv._masked(df, policy, "spine_axis")
    assert masked.sum() == 1 and df.loc[masked, "study"].item() == "s0"
    assert not cv._masked(df, policy, "spine_any").any()
