import glob, os
import numpy as np
import pandas as pd

print("=" * 60)
print("PART 1: transfer coverage in the raw DuckDB")
print("=" * 60)
dbs = sorted(set(glob.glob(os.path.expanduser("~/Transferlens/**/*.duckdb"), recursive=True)))
print("duckdb files found:", dbs if dbs else "NONE")
if dbs:
    import duckdb
    con = duckdb.connect(dbs[0], read_only=True)
    print("tables:", [t[0] for t in con.execute("SHOW TABLES").fetchall()])
    q = """
        SELECT transfer_season, COUNT(*) AS transfers,
               COUNT(DISTINCT player_id) AS players
        FROM transfers GROUP BY 1 ORDER BY 1
    """
    print(con.execute(q).fetchdf().to_string(index=False))
    con.close()

print("\n" + "=" * 60)
print("PART 2: does the model memorise players or drift over time?")
print("=" * 60)
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import roc_auc_score

SRC = "train_model.py"
lines = open(SRC).read().splitlines()
cut = next(i for i, l in enumerate(lines) if "TRAIN / TEST SPLIT" in l)
while cut > 0 and lines[cut - 1].strip().startswith("#"):
    cut -= 1
ns = {"__name__": "feature_export", "__file__": os.path.abspath(SRC)}
exec(compile("\n".join(lines[:cut]), SRC, "exec"), ns)
X, y, df = ns["X"].copy(), pd.Series(ns["y"]).astype(int), ns["df"]
season = df.loc[X.index, "season_year_numeric"].astype(float)
pid = df.loc[X.index, "player_id"]
keep = (season <= 2024) & pid.notna()
X, y, pid = X[keep], y[keep], pid[keep]
cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
for c in cat_cols:
    X[c] = X[c].astype(str)


def auc(tr_idx, te_idx):
    Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.1, random_state=42, stratify=ytr)
    m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, eval_metric="AUC",
                           random_seed=42, verbose=0, od_type="Iter", od_wait=100,
                           cat_features=cat_cols)
    m.fit(Xa, ya, eval_set=(Xv, yv), use_best_model=True)
    return roc_auc_score(y.iloc[te_idx], m.predict_proba(X.iloc[te_idx])[:, 1])


idx = np.arange(len(X))
r_tr, r_te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
print("\n[1/2] random split, 2012-2024 ... fitting")
a_rand = auc(r_tr, r_te)
g_tr, g_te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(idx, y, pid))
print("[2/2] split by player (no player in both), 2012-2024 ... fitting")
a_grp = auc(g_tr, g_te)
print("\nROC-AUC random split          : %.3f" % a_rand)
print("ROC-AUC grouped by player     : %.3f" % a_grp)
print("(time split from before       : 0.603)")
