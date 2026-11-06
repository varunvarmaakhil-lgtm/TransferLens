import os
import duckdb
import numpy as np
import pandas as pd

DB = "/Users/varunvarma/Transferlens/data/raw/transfermarkt-datasets.duckdb"
LAB = "/Users/varunvarma/Transferlens/data/processed/labels_from_appearances.csv"
OUT = "/Users/varunvarma/Transferlens/data/processed/features_v2.csv"


def season_of(d, cut_month):
    return d.dt.year - (d.dt.month < cut_month).astype(int)


con = duckdb.connect(DB, read_only=True)
lab = pd.read_csv(LAB).sort_values(["player_id", "season"]).reset_index(drop=True)
f = lab[["player_id", "season", "club_id", "minutes", "apps"]].copy()
done = []

app = con.execute("SELECT player_id, game_id, player_club_id, date FROM appearances").fetchdf()
app["season"] = season_of(pd.to_datetime(app["date"]), 7)
cg = (app.groupby(["player_club_id", "season"])["game_id"].nunique()
         .rename("club_games").reset_index().rename(columns={"player_club_id": "club_id"}))
f = f.merge(cg, on=["club_id", "season"], how="left")
f["minutes_share"] = f["minutes"] / (f["club_games"] * 90)
f["apps_share"] = f["apps"] / f["club_games"]
g = f.groupby("player_id")
consec = g["season"].shift(1) == f["season"] - 1
f["minutes_share_change"] = np.where(consec, f["minutes_share"] - g["minutes_share"].shift(1), np.nan)
new_spell = (g["club_id"].shift(1) != f["club_id"]) | ~consec
f["spell"] = new_spell.astype(int).groupby(f["player_id"]).cumsum()
f["club_tenure"] = f.groupby(["player_id", "spell"]).cumcount() + 1
f["new_arrival"] = new_spell.astype(int)
f["squad_size"] = f.groupby(["club_id", "season"])["player_id"].transform("size")
f["minutes_rank_pct"] = f.groupby(["club_id", "season"])["minutes"].rank(ascending=False, pct=True)
f["club_turnover"] = f.groupby(["club_id", "season"])["new_arrival"].transform("mean")
done.append("A playing time + tenure")

try:
    pv = con.execute("SELECT player_id, date, market_value_in_eur FROM player_valuations").fetchdf()
    pv["date"] = pd.to_datetime(pv["date"])
    pv["season"] = season_of(pv["date"], 6)
    pv = pv.sort_values(["player_id", "date"])
    last = pv.groupby(["player_id", "season"])["market_value_in_eur"].last().rename("mv_end").reset_index()
    last["mv_peak"] = last.groupby("player_id")["mv_end"].cummax()
    f = f.merge(last, on=["player_id", "season"], how="left")
    g = f.groupby("player_id")
    consec = g["season"].shift(1) == f["season"] - 1
    f["mv_log_change"] = np.where(consec, np.log1p(f["mv_end"]) - np.log1p(g["mv_end"].shift(1)), np.nan)
    f["mv_vs_peak"] = f["mv_end"] / f["mv_peak"]
    club_med = f.groupby(["club_id", "season"])["mv_end"].transform("median")
    f["mv_vs_club_median"] = f["mv_end"] / club_med
    f["club_median_mv"] = club_med
    f["mv_rank_in_club_pct"] = f.groupby(["club_id", "season"])["mv_end"].rank(ascending=False, pct=True)
    done.append("B market value")
except Exception as e:
    print("B market value skipped:", e)

try:
    try:
        cgm = con.execute(
            "SELECT c.club_id, c.game_id, c.own_position, c.own_manager_name, c.is_win, "
            "g.date, g.competition_type FROM club_games c JOIN games g USING (game_id)").fetchdf()
    except Exception:
        cgm = con.execute(
            "SELECT c.club_id, c.game_id, c.own_position, c.own_manager_name, c.is_win, "
            "g.date FROM club_games c JOIN games g USING (game_id)").fetchdf()
    cgm["date"] = pd.to_datetime(cgm["date"])
    cgm["season"] = season_of(cgm["date"], 7)
    cgm = cgm.sort_values("date")
    club = cgm.groupby(["club_id", "season"]).agg(
        club_win_rate=("is_win", "mean"),
        club_managers=("own_manager_name", "nunique")).reset_index()
    lg = cgm
    if "competition_type" in cgm.columns:
        lg = cgm[cgm["competition_type"] == "domestic_league"]
    pos = (lg.dropna(subset=["own_position"]).groupby(["club_id", "season"])["own_position"]
             .last().rename("club_final_position").reset_index())
    club = club.merge(pos, on=["club_id", "season"], how="left")
    f = f.merge(club, on=["club_id", "season"], how="left")
    done.append("C club situation")
except Exception as e:
    print("C club situation skipped:", e)
con.close()

feat_cols = [c for c in f.columns if c not in
             ("club_id", "minutes", "apps", "club_games", "spell", "mv_peak")]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
f[feat_cols].to_csv(OUT, index=False)
print("feature blocks built:", done)
print("coverage (share non-missing):")
print(f[feat_cols].drop(columns=["player_id", "season"]).notna().mean().round(2).to_string())
print("Saved", len(f), "rows to", OUT)
