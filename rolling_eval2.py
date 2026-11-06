import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

D = "/Users/varunvarma/Transferlens/data/"
OLD, LAB, NEW = D + "transfer_training_updated.csv", D + "processed/labels_from_appearances.csv", D + "transfer_training_v2.csv"
TEST_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
SEED = 42

old = pd.read_csv(OLD, low_memory=False)
old = old[old["player_id"].notna()].copy()
old["_s"] = pd.to_numeric(old["season"].astype(str).str[:4], errors="coerce")
lab = pd.read_csv(LAB, usecols=["player_id", "season", "moved"]).rename(columns={"season": "_s"})
v2 = old.drop(columns=["transferred_next_season"]).merge(lab, on=["player_id", "_s"], how="left")
v2 = v2.rename(columns={"moved": "transferred_next_season"})
n_before = len(v2)
v2 = v2[v2["transferred_next_season"].notna()].drop(columns=["_s"])
v2["transferred_next_season"] = v2["transferred_next_season"].astype(int)
v2.to_csv(NEW, index=False)
print("v2 training file: %d rows kept, %d dropped (no appearances next season)"
      % (len(v2), n_before - len(v2)))

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

FV2 = D + "processed/features_v2.csv"
fv = pd.read_csv(FV2).rename(columns={"season": "_s"})
key = pd.DataFrame({"_row": X.index, "player_id": df.loc[X.index, "player_id"].values,
                    "_s": season.values})
extra = key.merge(fv, on=["player_id", "_s"], how="left").set_index("_row")
extra = extra.drop(columns=["player_id", "_s"]).loc[X.index]
X = pd.concat([X, extra.add_prefix("v2_")], axis=1)
print("added %d v2 features, matched %.1f%% of rows" % (extra.shape[1], 100 * extra.notna().any(axis=1).mean()))

time_cols = [c for c in X.columns if c in ("season", "name", "date_of_birth", "player_id") or "season_year" in c or "birth_year" in c]
X = X.drop(columns=time_cols)
cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
for c in cat_cols:
    X[c] = X[c].astype(str)
num_cols = [c for c in X.columns if c not in cat_cols]
print("features used: %d (dropped time proxies: %s)" % (X.shape[1], time_cols))


def fit_cb(Xa, ya, Xv, yv):
    m = CatBoostClassifier(iterations=1500, learning_rate=0.06, depth=6, eval_metric="AUC",
                           random_seed=SEED, verbose=0, od_type="Iter", od_wait=60,
                           cat_features=cat_cols)
    m.fit(Xa, ya, eval_set=(Xv, yv), use_best_model=True)
    return m


def fit_lr(Xa, ya):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=2000))
    return m.fit(Xa[num_cols], ya)


def best_f1_thr(yt, p):
    ths = np.round(np.arange(0.10, 0.81, 0.01), 2)
    return float(max(ths, key=lambda t: f1_score(yt, (p >= t).astype(int), zero_division=0)))


rows = []
for T in TEST_SEASONS:
    tr, va, te = season <= T - 2, season == T - 1, season == T
    print("\n[test %d] train %d rows (<=%d), val %d, test %d ... fitting"
          % (T, tr.sum(), T - 2, va.sum(), te.sum()))
    cb = fit_cb(X[tr], y[tr], X[va], y[va])
    thr = best_f1_thr(y[va], cb.predict_proba(X[va])[:, 1])
    p = cb.predict_proba(X[te])[:, 1]
    lr = fit_lr(X[tr | va], y[tr | va])
    p_lr = lr.predict_proba(X[te][num_cols])[:, 1]
    rows.append({"test_season": T, "n_test": int(te.sum()), "base_rate": y[te].mean(),
                 "cb_roc_auc": roc_auc_score(y[te], p), "cb_pr_auc": average_precision_score(y[te], p),
                 "cb_f1": f1_score(y[te], (p >= thr).astype(int)), "threshold": thr,
                 "lr_roc_auc": roc_auc_score(y[te], p_lr)})
    print("   CatBoost ROC-AUC %.3f | LogReg ROC-AUC %.3f" % (rows[-1]["cb_roc_auc"], rows[-1]["lr_roc_auc"]))

res = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
res.to_csv("results/rolling_eval_v2.csv", index=False)
pd.set_option("display.width", 200)
print("\n" + "=" * 70)
print("ROLLING-ORIGIN RESULTS (new label + v2 features)")
print("=" * 70)
print(res.round(3).to_string(index=False))
print("\nmean CatBoost ROC-AUC %.3f (sd %.3f) | mean LogReg ROC-AUC %.3f"
      % (res.cb_roc_auc.mean(), res.cb_roc_auc.std(), res.lr_roc_auc.mean()))
base = pd.read_csv("results/rolling_eval.csv")
print("before (v1 features): mean CatBoost ROC-AUC %.3f" % base.cb_roc_auc.mean())
imp = pd.Series(cb.get_feature_importance(), index=X.columns).sort_values(ascending=False)
print("\ntop 15 features (model tested on 2024):")
print(imp.head(15).round(2).to_string())
print("Saved results/rolling_eval_v2.csv")
