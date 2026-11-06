# TransferLens - honest re-evaluation of the transfer classifier
# Reuses the feature engineering in train_model.py (everything before the
# TRAIN / TEST SPLIT section), then compares:
#   A) the original protocol: random 80/20 split, threshold tuned on test
#   B) a time split: train <= 2022, validate 2023, test 2024
#   C) the same time split without the season-number feature
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score,
                             average_precision_score)

SRC = "train_model.py"
TRAIN_MAX, VAL_SEASON, TEST_SEASON = 2022, 2023, 2024
SEED = 42

lines = open(SRC).read().splitlines()
cut = next(i for i, l in enumerate(lines) if "TRAIN / TEST SPLIT" in l)
while cut > 0 and lines[cut - 1].strip().startswith("#"):
    cut -= 1
print("Running feature engineering from", SRC, "lines 1 to", cut, "...")
ns = {"__name__": "feature_export", "__file__": os.path.abspath(SRC)}
exec(compile("\n".join(lines[:cut]), SRC, "exec"), ns)
X, y, df = ns["X"].copy(), pd.Series(ns["y"]).astype(int), ns["df"]
assert X.index.isin(df.index).all(), "X index does not line up with df"
season = df.loc[X.index, "season_year_numeric"].astype(float)
pid = df.loc[X.index, "player_id"]

cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
for c in cat_cols:
    X[c] = X[c].astype(str).fillna("missing")

print("\n" + "=" * 60)
print("FEATURES:", X.shape[1], "| ROWS:", len(X), "| categorical:", cat_cols)
print("season feature in X:", [c for c in X.columns if "season" in c.lower()])
print("rows / positive rate by season:")
print(pd.DataFrame({"rows": y.groupby(season).size(),
                    "pos_rate": y.groupby(season).mean().round(3)}).to_string())


def fit(Xtr, ytr, Xva, yva, cols):
    m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6,
                           eval_metric="AUC", random_seed=SEED, verbose=0,
                           od_type="Iter", od_wait=100,
                           cat_features=[c for c in cat_cols if c in cols])
    m.fit(Xtr[cols], ytr, eval_set=(Xva[cols], yva), use_best_model=True)
    return m


def best_thr(yt, p):
    ths = np.round(np.arange(0.20, 0.81, 0.01), 2)
    return float(max(ths, key=lambda t: accuracy_score(yt, (p >= t).astype(int))))


def report(name, yt, p, thr, thr_from):
    pred = (p >= thr).astype(int)
    br = float(np.mean(yt))
    return {"setup": name, "threshold": thr, "thr_tuned_on": thr_from,
            "accuracy": accuracy_score(yt, pred),
            "majority_baseline_acc": max(br, 1 - br),
            "roc_auc": roc_auc_score(yt, p),
            "pr_auc": average_precision_score(yt, p), "base_rate": br,
            "f1": f1_score(yt, pred, zero_division=0),
            "precision": precision_score(yt, pred, zero_division=0),
            "recall": recall_score(yt, pred, zero_division=0),
            "n_test": len(yt)}


rows = []
all_cols = list(X.columns)

print("\n[A] random split (original protocol) ... fitting")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20,
                                      random_state=SEED, stratify=y)
Xtr2, Xva, ytr2, yva = train_test_split(Xtr, ytr, test_size=0.10,
                                        random_state=SEED, stratify=ytr)
m = fit(Xtr2, ytr2, Xva, yva, all_cols)
p_te, p_va = m.predict_proba(Xte[all_cols])[:, 1], m.predict_proba(Xva[all_cols])[:, 1]
rows.append(report("A random split", yte, p_te, best_thr(yte, p_te), "TEST (leaky)"))
rows.append(report("A random split", yte, p_te, best_thr(yva, p_va), "validation"))
overlap = pid.loc[Xte.index].isin(set(pid.loc[Xtr.index].dropna())).mean()
print("    test rows whose player also appears in train: %.1f%%" % (100 * overlap))

tr, va, te = season <= TRAIN_MAX, season == VAL_SEASON, season == TEST_SEASON
print("\nTime split rows: train %d | val %d | test %d | dropped (2025-26 etc.) %d"
      % (tr.sum(), va.sum(), te.sum(), (~(tr | va | te)).sum()))
season_cols = [c for c in all_cols if "season_year" in c.lower()]
setups = [("B time split", all_cols)]
if season_cols:
    setups.append(("C time split, no season feature",
                   [c for c in all_cols if c not in season_cols]))
for name, cols in setups:
    print("\n[%s] fitting ..." % name)
    m = fit(X[tr], y[tr], X[va], y[va], cols)
    p_va = m.predict_proba(X[va][cols])[:, 1]
    p_te = m.predict_proba(X[te][cols])[:, 1]
    rows.append(report(name, y[te], p_te, best_thr(y[va], p_va), "validation (2023)"))
    rows.append(report(name, y[te], p_te, 0.50, "fixed 0.50"))
    if name.startswith("C"):
        imp = pd.Series(m.get_feature_importance(), index=cols).sort_values(ascending=False)
        print("    top 10 features:")
        print(imp.head(10).round(2).to_string())

res = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
res.to_csv("results/time_split_results.csv", index=False)
print("\n" + "=" * 60)
print("RESULTS (test set metrics)")
print("=" * 60)
show = ["setup", "thr_tuned_on", "threshold", "accuracy", "majority_baseline_acc",
        "roc_auc", "pr_auc", "base_rate", "f1", "precision", "recall"]
pd.set_option("display.width", 250)
print(res[show].round(4).to_string(index=False))
print("\nSaved: results/time_split_results.csv")
