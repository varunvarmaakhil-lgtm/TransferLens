"""Find players with a similar profile.

Usage:
  python similar.py "Vangelis Pavlidis"
  python similar.py "Vangelis Pavlidis" --k 10 --max-age 25 --max-value 20
  python similar.py --eval
"""
import argparse
import unicodedata
import numpy as np
import pandas as pd

D = "/Users/varunvarma/Transferlens/data/processed/"
MIN_MINUTES = 900
FEATS = {"goals_p90": 2.0, "assists_p90": 1.5, "ga_p90_prev1": 1.0,
         "minutes_share": 1.5, "apps_share": 0.5, "minutes_share_prev1": 0.5,
         "yellow_p90": 0.5, "log_club_value": 1.5, "age": 1.0, "height_in_cm": 0.75}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def load(season):
    b = pd.read_csv(D + "player_seasons.csv", low_memory=False)
    b = b[(b["season"] == season) & (b["minutes_all"] >= MIN_MINUTES)].copy()
    b["yellow_p90"] = b["yellow"] / b["minutes_all"] * 90
    b["log_club_value"] = np.log1p(b["club_median_mv"])
    for c in ["goals_p90", "assists_p90", "ga_p90"]:
        b[c] = b[c].fillna(0)
    b["ga_p90_prev1"] = b["ga_p90_prev1"].fillna(b["ga_p90"])
    b["minutes_share_prev1"] = b["minutes_share_prev1"].fillna(b["minutes_share"])
    for c in FEATS:
        b[c] = b[c].fillna(b[c].median())
    b["grp"] = b["sub_position"].fillna(b["position"]).astype(str)
    b["key"] = b["name"].map(norm)
    return b.reset_index(drop=True)


def zmatrix(b, ref=None):
    ref = b if ref is None else ref
    mu, sd = ref[list(FEATS)].mean(), ref[list(FEATS)].std().replace(0, 1)
    w = np.array(list(FEATS.values()))
    return ((b[list(FEATS)] - mu) / sd).values * w


def similar(name, k=10, max_age=None, max_value=None, season=2025):
    b = load(season)
    hit = b[b["key"].str.contains(norm(name), regex=False)]
    if hit.empty:
        raise SystemExit("No player matching '%s' with >= %d minutes in %d/%d"
                         % (name, MIN_MINUTES, season, season + 1))
    exact = hit[hit["key"] == norm(name)]
    hit = pd.concat([exact, hit.drop(exact.index).sort_values("minutes_all", ascending=False)])
    q = hit.iloc[0]
    if len(hit) > 1:
        print("several matches, using %s (%s); others: %s"
              % (q["name"], q["club_name"], ", ".join(hit["name"].iloc[1:6])))
    pool = b[(b["grp"] == q["grp"]) & (b["player_id"] != q["player_id"])].copy()
    Z = zmatrix(pd.concat([pool, q.to_frame().T]).astype({c: float for c in FEATS}), ref=b)
    pool["distance"] = np.sqrt(((Z[:-1] - Z[-1]) ** 2).sum(axis=1))
    if max_age is not None:
        pool = pool[pool["age"] <= max_age]
    if max_value is not None:
        pool = pool[pool["mv_end"] <= max_value * 1e6]
    try:
        fp = pd.read_csv(D + "fair_price_%d.csv" % season, usecols=["player_id", "fair_low", "fair_mid"])
        pool = pool.merge(fp, on="player_id", how="left")
        pool["undervalued"] = np.where(pool["mv_end"] < pool["fair_low"], "yes", "")
    except FileNotFoundError:
        pool["fair_mid"], pool["undervalued"] = np.nan, ""
    out = pool.sort_values("distance").head(k)
    print("\n%s | %s | %s | age %.0f | G+A/90 %.2f | minutes share %.2f | value EUR %.1fM"
          % (q["name"], q["club_name"], q["grp"], q["age"], q["ga_p90"], q["minutes_share"], q["mv_end"] / 1e6))
    t = out[["name", "club_name", "age", "ga_p90", "minutes_share", "mv_end", "fair_mid", "undervalued", "distance"]].copy()
    t["mv_end"], t["fair_mid"] = (t["mv_end"] / 1e6).round(1), (t["fair_mid"] / 1e6).round(1)
    t = t.round({"age": 0, "ga_p90": 2, "minutes_share": 2, "distance": 2})
    t.columns = ["name", "club", "age", "G+A/90", "min_share", "value_M", "fair_M", "underval", "distance"]
    filt = []
    if max_age is not None:
        filt.append("age <= %s" % max_age)
    if max_value is not None:
        filt.append("value <= EUR %sM" % max_value)
    print("most similar %s%s:" % (q["grp"], (" (" + ", ".join(filt) + ")") if filt else ""))
    pd.set_option("display.width", 200)
    print(t.to_string(index=False))
    return out


def evaluate(season=2025):
    cur, prev = load(season), load(season - 1)
    Zc, Zp = zmatrix(cur), zmatrix(prev, ref=cur)
    pidx = {p: i for i, p in enumerate(prev["player_id"])}
    ranks, pools = [], []
    for g, idx in cur.groupby("grp").groups.items():
        idx = np.asarray(idx)
        for i in idx:
            j = pidx.get(cur.at[i, "player_id"])
            if j is None or prev.at[j, "grp"] != g:
                continue
            d = np.sqrt(((Zc[idx] - Zp[j]) ** 2).sum(axis=1))
            self_d = d[idx == i][0]
            ranks.append(int((d < self_d).sum()) + 1)
            pools.append(len(idx))
    r, n = np.array(ranks), np.array(pools)
    print("re-identification test, %d/%d profile vs %d/%d pool, %d players"
          % (season - 1, season, season, season + 1, len(r)))
    print("  found himself at rank 1: %.1f%% | in top 10: %.1f%% | median rank %d of a median pool of %d"
          % (100 * np.mean(r == 1), 100 * np.mean(r <= 10), int(np.median(r)), int(np.median(n))))
    print("  random guessing would put him in the top 10 about %.1f%% of the time" % (100 * np.mean(np.minimum(10, n) / n)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-age", type=float)
    ap.add_argument("--max-value", type=float, help="EUR millions")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.eval or not a.name:
        evaluate()
    if a.name:
        similar(a.name, a.k, a.max_age, a.max_value)
