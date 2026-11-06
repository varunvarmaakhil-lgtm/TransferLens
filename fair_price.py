import os
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

D = "/Users/varunvarma/Transferlens/data/processed/"
MODEL_DIR = "/Users/varunvarma/Transferlens/models/"
TEST_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
ALPHA = 0.2
SEED = 42

b = pd.read_csv(D + "player_seasons.csv", low_memory=False)
b = b[b["mv_end"].notna() & (b["mv_end"] > 0)].copy()
y_all = np.log(b["mv_end"])

drop = {"player_id", "season", "name", "date_of_birth", "club_id", "club_name", "mv_end",
        "mv_log_change", "mv_vs_peak", "mv_vs_club_median", "mv_rank_in_club_pct"}
cat_cols = ["position", "sub_position", "foot", "country_of_citizenship", "main_comp"]
feats = [c for c in b.columns if c not in drop]
X_all = b[feats].copy()
for c in cat_cols:
    X_all[c] = X_all[c].fillna("missing").astype(str)
season = b["season"]
print("rows %d | features %d" % (len(b), len(feats)))


def fit(Xt, yt):
    m = CatBoostRegressor(loss_function="MultiQuantile:alpha=0.1,0.5,0.9", iterations=800,
                          learning_rate=0.06, depth=6, random_seed=SEED, verbose=0,
                          cat_features=cat_cols)
    return m.fit(Xt, yt)


def pred(m, Xp):
    q = np.sort(np.asarray(m.predict(Xp)).reshape(len(Xp), 3), axis=1)
    return q[:, 0], q[:, 1], q[:, 2]


def conformal(lo, hi, yc):
    e = np.maximum(lo - yc, yc - hi)
    n = len(e)
    return float(np.quantile(e, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)))


rows = []
for T in TEST_SEASONS:
    tr, ca, te = season <= T - 2, season == T - 1, season == T
    m = fit(X_all[tr], y_all[tr])
    lo_c, _, hi_c = pred(m, X_all[ca])
    qhat = conformal(lo_c, hi_c, y_all[ca].values)
    lo, med, hi = pred(m, X_all[te])
    lo, hi = lo - qhat, hi + qhat
    yt = y_all[te].values
    cover = np.mean((yt >= lo) & (yt <= hi))
    ape = np.abs(np.exp(med) - np.exp(yt)) / np.exp(yt)
    prev = b.loc[te, "mv_end_prev1"].values
    has = ~np.isnan(prev)
    ape_naive = np.abs(prev[has] - np.exp(yt[has])) / np.exp(yt[has])
    rows.append({"test_season": T, "n": int(te.sum()), "coverage_80": cover,
                 "median_width_ratio": float(np.median(np.exp(hi - lo))),
                 "mdape_model": float(np.median(ape)),
                 "mdape_model_same_rows": float(np.median(ape[has])),
                 "mdape_last_value": float(np.median(ape_naive))})
    r = rows[-1]
    print("[test %d] coverage %.3f | width x%.2f | MdAPE model %.3f vs last-value %.3f"
          % (T, r["coverage_80"], r["median_width_ratio"], r["mdape_model_same_rows"], r["mdape_last_value"]))

res = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
res.to_csv("results/fair_price_eval.csv", index=False)
pd.set_option("display.width", 200)
print("\nFAIR-PRICE MODEL, ROLLING ORIGIN (80% conformal range)")
print(res.round(3).to_string(index=False))
print("mean coverage %.3f (target 0.80) | mean MdAPE model %.3f vs last-value baseline %.3f"
      % (res.coverage_80.mean(), res.mdape_model_same_rows.mean(), res.mdape_last_value.mean()))

tr, ca, now = season <= 2023, season == 2024, season == 2025
m = fit(X_all[tr], y_all[tr])
lo_c, _, hi_c = pred(m, X_all[ca])
qhat = conformal(lo_c, hi_c, y_all[ca].values)
os.makedirs(MODEL_DIR, exist_ok=True)
m.save_model(MODEL_DIR + "fair_price_v1.cbm")
with open(MODEL_DIR + "fair_price_v1.json", "w") as fh:
    json.dump({"target": "log market value, last valuation before 1 June after the season",
               "quantiles": [0.1, 0.5, 0.9], "conformal_alpha": ALPHA, "conformal_qhat_log": qhat,
               "trained_on": "2012-2023", "calibrated_on": "2024",
               "features": feats, "cat_features": cat_cols,
               "eval": {"coverage_mean": round(float(res.coverage_80.mean()), 4),
                        "mdape_model": round(float(res.mdape_model_same_rows.mean()), 4),
                        "mdape_last_value": round(float(res.mdape_last_value.mean()), 4)}},
              fh, indent=2)

cur = b[now].copy()
lo, med, hi = pred(m, X_all[now])
cur["fair_low"], cur["fair_mid"], cur["fair_high"] = np.exp(lo - qhat), np.exp(med), np.exp(hi + qhat)
cur["value_gap"] = cur["fair_mid"] / cur["mv_end"]
under = cur[(cur["mv_end"] < cur["fair_low"]) & (cur["minutes_share"] >= 0.3)]
under = under.sort_values("value_gap", ascending=False)
cols = ["name", "club_name", "position", "age", "minutes_share", "ga_p90",
        "mv_end", "fair_low", "fair_mid", "fair_high", "value_gap"]
under[["player_id"] + cols].to_csv("results/undervalued_2025.csv", index=False)
cur[["player_id"] + cols].to_csv(D + "fair_price_2025.csv", index=False)
show = under[cols].head(15).copy()
for c in ["mv_end", "fair_low", "fair_mid", "fair_high"]:
    show[c] = (show[c] / 1e6).round(1)
show["age"] = show["age"].round(0)
show["minutes_share"] = show["minutes_share"].round(2)
show["ga_p90"] = show["ga_p90"].round(2)
show["value_gap"] = show["value_gap"].round(2)
print("\n2025/26: %d players scored, %d flagged undervalued (value below the 80%% range, minutes share >= 0.3)"
      % (len(cur), len(under)))
print("top 15 (values in EUR millions):")
print(show.to_string(index=False))
print("\nSaved models/fair_price_v1.cbm, results/fair_price_eval.csv, results/undervalued_2025.csv")
