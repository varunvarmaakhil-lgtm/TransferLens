import os
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

D = "/Users/varunvarma/Transferlens/data/"
OLD, LAB, NEW = D + "transfer_training_updated.csv", D + "processed/labels_from_appearances.csv", D + "transfer_training_v2.csv"
MODEL_DIR = "/Users/varunvarma/Transferlens/models/"
SEED = 42

old = pd.read_csv(OLD, low_memory=False)
old = old[old["player_id"].notna()].copy()
old["_s"] = pd.to_numeric(old["season"].astype(str).str[:4], errors="coerce")
lab = pd.read_csv(LAB, usecols=["player_id", "season", "moved"]).rename(columns={"season": "_s"})
v2 = old.drop(columns=["transferred_next_season"]).merge(lab, on=["player_id", "_s"], how="left")
v2 = v2.rename(columns={"moved": "transferred_next_season"})
v2 = v2[v2["transferred_next_season"].notna()].drop(columns=["_s"])
v2["transferred_next_season"] = v2["transferred_next_season"].astype(int)
v2.to_csv(NEW, index=False)

SRC = "train_model.py"
lines = open(SRC).read().splitlines()
lines = ['DATA_PATH = "%s"' % NEW if l.startswith("DATA_PATH") else l for l in lines]
cut = next(i for i, l in enumerate(lines) if "TRAIN / TEST SPLIT" in l)
while cut > 0 and lines[cut - 1].strip().startswith("#"):
    cut -= 1
print("Running feature engineering ...")
ns = {"__name__": "feature_export", "__file__": os.path.abspath(SRC)}
exec(compile("\n".join(lines[:cut]), SRC, "exec"), ns)
X, y, df = ns["X"].copy(), pd.Series(ns["y"]).astype(int), ns["df"]
season = df.loc[X.index, "season_year_numeric"].astype(float)

fv = pd.read_csv(D + "processed/features_v2.csv").rename(columns={"season": "_s"})
key = pd.DataFrame({"_row": X.index, "player_id": df.loc[X.index, "player_id"].values,
                    "_s": season.values})
extra = key.merge(fv, on=["player_id", "_s"], how="left").set_index("_row")
extra = extra.drop(columns=["player_id", "_s"]).loc[X.index]
X = pd.concat([X, extra.add_prefix("v2_")], axis=1)

time_cols = [c for c in X.columns if c in ("season", "name", "date_of_birth", "player_id") or "season_year" in c or "birth_year" in c]
X = X.drop(columns=time_cols)
cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
for c in cat_cols:
    X[c] = X[c].astype(str)
print("features used: %d | dropped: %s" % (X.shape[1], time_cols))


def best_f1_thr(yt, p):
    ths = np.round(np.arange(0.10, 0.81, 0.01), 2)
    return float(max(ths, key=lambda t: f1_score(yt, (p >= t).astype(int), zero_division=0)))


os.makedirs(MODEL_DIR, exist_ok=True)
tr, va = season <= 2023, season == 2024
print("\nstep 1: train <=2023 (%d rows), early-stop + threshold on 2024 (%d rows)" % (tr.sum(), va.sum()))
m1 = CatBoostClassifier(iterations=1500, learning_rate=0.06, depth=6, eval_metric="AUC",
                        random_seed=SEED, verbose=0, od_type="Iter", od_wait=60, cat_features=cat_cols)
m1.fit(X[tr], y[tr], eval_set=(X[va], y[va]), use_best_model=True)
best_it = int(m1.get_best_iteration()) + 1
thr = best_f1_thr(y[va], m1.predict_proba(X[va])[:, 1])
print("   best iterations %d | F1-optimal threshold %.2f" % (best_it, thr))

allrows = season <= 2024
print("step 2: refit on all labelled seasons 2012-2024 (%d rows) with %d iterations" % (allrows.sum(), best_it))
final = CatBoostClassifier(iterations=best_it, learning_rate=0.06, depth=6, random_seed=SEED,
                           verbose=0, cat_features=cat_cols)
final.fit(X[allrows], y[allrows])
final.save_model(MODEL_DIR + "transfer_model_v2.cbm")

ev = pd.read_csv("results/rolling_eval_v2.csv")
meta = {
    "model": "CatBoostClassifier",
    "label": "main club (most minutes) in season t+1 differs from season t; from appearances table",
    "season_definition": "season t = July t to June t+1",
    "trained_on_seasons": "2012-2024",
    "n_train_rows": int(allrows.sum()),
    "iterations": best_it,
    "threshold_f1": thr,
    "features": list(X.columns),
    "cat_features": cat_cols,
    "dropped_columns": time_cols,
    "rolling_eval_2019_2024": {
        "roc_auc_mean": round(float(ev.cb_roc_auc.mean()), 4),
        "roc_auc_sd": round(float(ev.cb_roc_auc.std()), 4),
        "pr_auc_mean": round(float(ev.cb_pr_auc.mean()), 4),
        "logreg_roc_auc_mean": round(float(ev.lr_roc_auc.mean()), 4)},
}
with open(MODEL_DIR + "transfer_model_v2.json", "w") as fh:
    json.dump(meta, fh, indent=2)
imp = pd.Series(final.get_feature_importance(), index=X.columns).sort_values(ascending=False)
imp.round(3).to_csv("results/feature_importance_v2.csv", header=["importance"])

check = CatBoostClassifier()
check.load_model(MODEL_DIR + "transfer_model_v2.cbm")
p = check.predict_proba(X[va])[:, 1]
print("reload check (in-sample, file integrity only): %d predictions, mean prob %.3f, true rate %.3f"
      % (len(p), p.mean(), y[va].mean()))
print("\nSaved:\n  %stransfer_model_v2.cbm\n  %stransfer_model_v2.json\n  results/feature_importance_v2.csv" % (MODEL_DIR, MODEL_DIR))
print("top 10 features:")
print(imp.head(10).round(2).to_string())
