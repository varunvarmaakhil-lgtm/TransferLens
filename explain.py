"""SHAP explanations for the 2025/26 transfer and fair-price models (CatBoost built-in SHAP)."""
import json
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

D = "/Users/varunvarma/Transferlens/data/processed/"
M = "/Users/varunvarma/Transferlens/models/"
SEASON, TOP = 2025, 8

b = pd.read_csv(D + "player_seasons.csv", low_memory=False)
lab = pd.read_csv(D + "labels_from_appearances.csv", usecols=["player_id", "season", "moved"])
b = b.merge(lab, on=["player_id", "season"], how="left").sort_values(["player_id", "season"]).reset_index(drop=True)
g = b.groupby("player_id")["moved"]
b["prev_moved_rate"] = g.transform(lambda s: s.shift(1).expanding().mean())
b["prev_moves"] = g.transform(lambda s: s.shift(1).fillna(0).cumsum())
now = b[b["season"] == SEASON].reset_index(drop=True)
print("players in %d/%d: %d" % (SEASON, SEASON + 1, len(now)))


def matrix(meta):
    X = now[meta["features"]].copy()
    for c in meta["cat_features"]:
        X[c] = X[c].fillna("missing").astype(str)
    return X


def run(name, model, meta, dim=None):
    X = matrix(meta)
    pool = Pool(X, cat_features=meta["cat_features"])
    sv = np.asarray(model.get_feature_importance(pool, type="ShapValues"))
    raw = np.asarray(model.predict(pool, prediction_type="RawFormulaVal"))
    if sv.ndim == 3:
        sv, raw = sv[:, dim, :], raw[:, dim]
    contrib, base = sv[:, :-1], sv[:, -1]
    err = float(np.max(np.abs(contrib.sum(axis=1) + base - raw)))
    print("[%s] SHAP additivity check: max |sum(SHAP) + base - prediction| = %.2e" % (name, err))
    glob = pd.Series(np.abs(contrib).mean(axis=0), index=X.columns).sort_values(ascending=False)
    glob.round(4).to_csv("results/shap_global_%s.csv" % name, header=["mean_abs_shap"])
    print("[%s] top global drivers: %s" % (name, ", ".join(glob.index[:8])))
    cut = np.quantile(np.abs(contrib), [0.75, 0.95])
    rows = []
    for i in range(len(X)):
        order = np.argsort(-np.abs(contrib[i]))[:TOP]
        for rank, j in enumerate(order, 1):
            v, a = X.iat[i, j], abs(contrib[i, j])
            rows.append({"player_id": now.at[i, "player_id"], "model": name, "rank": rank,
                         "feature": X.columns[j],
                         "value": v if isinstance(v, str) else (None if pd.isna(v) else round(float(v), 4)),
                         "shap": round(float(contrib[i, j]), 4),
                         "strength": "strong" if a >= cut[1] else ("moderate" if a >= cut[0] else "weak")})
    return pd.DataFrame(rows), err


os.makedirs("results", exist_ok=True)
mt = json.load(open(M + "transfer_now.json"))
clf = CatBoostClassifier()
clf.load_model(M + "transfer_now.cbm")
ex_t, err_t = run("transfer", clf, mt)
mf = json.load(open(M + "fair_price_v1.json"))
reg = CatBoostRegressor()
reg.load_model(M + "fair_price_v1.cbm")
ex_f, err_f = run("fairprice", reg, mf, dim=1)
ex = pd.concat([ex_t, ex_f], ignore_index=True)
ex.to_csv(D + "explanations_%d.csv" % SEASON, index=False)
print("saved %d explanation rows to %sexplanations_%d.csv" % (len(ex), D, SEASON))
