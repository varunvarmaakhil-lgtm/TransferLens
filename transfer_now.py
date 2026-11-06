import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

D = "/Users/varunvarma/Transferlens/data/processed/"
MODEL_DIR = "/Users/varunvarma/Transferlens/models/"
SEED = 42

b = pd.read_csv(D + "player_seasons.csv", low_memory=False)
lab = pd.read_csv(D + "labels_from_appearances.csv", usecols=["player_id", "season", "moved"])
b = b.merge(lab, on=["player_id", "season"], how="left").sort_values(["player_id", "season"]).reset_index(drop=True)
g = b.groupby("player_id")["moved"]
b["prev_moved_rate"] = g.transform(lambda s: s.shift(1).expanding().mean())
b["prev_moves"] = g.transform(lambda s: s.shift(1).fillna(0).cumsum())

drop = {"player_id", "season", "name", "date_of_birth", "club_id", "club_name", "moved"}
cat_cols = ["position", "sub_position", "foot", "country_of_citizenship", "main_comp"]
feats = [c for c in b.columns if c not in drop]
X = b[feats].copy()
for c in cat_cols:
    X[c] = X[c].fillna("missing").astype(str)
y, season, has = b["moved"], b["season"], b["moved"].notna()
print("rows %d (labelled %d) | features %d" % (len(b), has.sum(), len(feats)))


def fit(tr, va, iters=1500, early=True):
    m = CatBoostClassifier(iterations=iters, learning_rate=0.06, depth=6, eval_metric="AUC",
                           random_seed=SEED, verbose=0, cat_features=cat_cols,
                           od_type="Iter" if early else None, od_wait=60 if early else None)
    if early:
        m.fit(X[tr], y[tr].astype(int), eval_set=(X[va], y[va].astype(int)), use_best_model=True)
    else:
        m.fit(X[tr], y[tr].astype(int))
    return m


rows = []
for T in tqdm([2022, 2023, 2024], desc="Transfer eval", unit="season", ncols=90):
    tr, va, te = has & (season <= T - 2), has & (season == T - 1), has & (season == T)
    p = fit(tr, va).predict_proba(X[te])[:, 1]
    rows.append({"test_season": T, "roc_auc": roc_auc_score(y[te], p),
                 "pr_auc": average_precision_score(y[te], p), "base_rate": y[te].mean()})
    tqdm.write("[test %d] ROC-AUC %.3f | PR-AUC %.3f | base rate %.3f"
               % (T, rows[-1]["roc_auc"], rows[-1]["pr_auc"], rows[-1]["base_rate"]))
res = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
res.to_csv("results/transfer_now_eval.csv", index=False)
print("mean ROC-AUC %.3f on 2022-2024 (Phase 1 model on the same seasons: 0.742)" % res.roc_auc.mean())

print("final model: tune on 2024, refit on 2012-2024, score 2025/26 ...")
m1 = fit(has & (season <= 2023), has & (season == 2024))
best_it = int(m1.get_best_iteration()) + 1
final = fit(has & (season <= 2024), None, iters=best_it, early=False)
now = season == 2025
out = b.loc[now, ["player_id"]].copy()
out["p_move"] = final.predict_proba(X[now])[:, 1]
out.to_csv(D + "transfer_prob_2025.csv", index=False)
os.makedirs(MODEL_DIR, exist_ok=True)
final.save_model(MODEL_DIR + "transfer_now.cbm")
with open(MODEL_DIR + "transfer_now.json", "w") as fh:
    json.dump({"label": "main club changes between season t and t+1 (appearances)",
               "trained_on": "2012-2024", "iterations": best_it, "features": feats,
               "cat_features": cat_cols, "base_rate_2024": round(float(y[season == 2024].mean()), 4),
               "eval_2022_2024": {"roc_auc_mean": round(float(res.roc_auc.mean()), 4),
                                  "pr_auc_mean": round(float(res.pr_auc.mean()), 4)}}, fh, indent=2)
imp = pd.Series(final.get_feature_importance(), index=feats).sort_values(ascending=False)
imp.round(3).to_csv("results/feature_importance_transfer_now.csv", header=["importance"])
print("scored %d players for 2025/26 | mean probability %.3f" % (len(out), out.p_move.mean()))
print("Saved models/transfer_now.cbm and data/processed/transfer_prob_2025.csv")
