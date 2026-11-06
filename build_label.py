import os
import duckdb
import numpy as np
import pandas as pd

DB = "/Users/varunvarma/Transferlens/data/raw/transfermarkt-datasets.duckdb"
TRAIN = "/Users/varunvarma/Transferlens/data/transfer_training_updated.csv"
OUT = "/Users/varunvarma/Transferlens/data/processed/labels_from_appearances.csv"

con = duckdb.connect(DB, read_only=True)
app = con.execute(
    "SELECT player_id, player_club_id, date, minutes_played FROM appearances"
).fetchdf()
con.close()
app["date"] = pd.to_datetime(app["date"])
app["season"] = app["date"].dt.year - (app["date"].dt.month < 7).astype(int)
app["minutes_played"] = pd.to_numeric(app["minutes_played"], errors="coerce").fillna(0)
print("appearances:", len(app), "| dates", app["date"].min().date(), "to", app["date"].max().date())

g = (app.groupby(["player_id", "season", "player_club_id"])
        .agg(minutes=("minutes_played", "sum"), apps=("date", "size"))
        .reset_index()
        .sort_values(["player_id", "season", "minutes", "apps"],
                     ascending=[True, True, False, False]))
n_clubs = g.groupby(["player_id", "season"]).size().rename("clubs_in_season")
main = (g.drop_duplicates(["player_id", "season"])
         .merge(n_clubs, on=["player_id", "season"])
         .rename(columns={"player_club_id": "club_id"}))

nxt = main[["player_id", "season", "club_id"]].rename(columns={"club_id": "next_club_id"})
nxt["season"] = nxt["season"] - 1
lab = main.merge(nxt, on=["player_id", "season"], how="left")
lab["seen_next"] = lab["next_club_id"].notna()
lab["moved"] = np.where(lab["seen_next"],
                        (lab["club_id"] != lab["next_club_id"]).astype(float), np.nan)

summary = lab.groupby("season").agg(
    player_seasons=("player_id", "size"),
    seen_next_rate=("seen_next", "mean"),
    moved_rate=("moved", "mean"),
    multi_club_rate=("clubs_in_season", lambda s: (s > 1).mean()))
print("\nNEW LABEL BY SEASON (moved_rate = among players seen next season)")
print(summary.round(3).to_string())

tr = pd.read_csv(TRAIN, usecols=["player_id", "season", "transferred_next_season"])
tr = tr[tr["player_id"].notna()].copy()
tr["s"] = pd.to_numeric(tr["season"].astype(str).str[:4], errors="coerce")
print("\nOLD vs NEW LABEL")
for off in (0, -1):
    m = tr.assign(season=tr["s"] + off).merge(
        lab[["player_id", "season", "moved", "seen_next"]], on=["player_id", "season"], how="left")
    found = m["seen_next"].notna().mean()
    both = m.dropna(subset=["moved"])
    agree = (both["moved"] == both["transferred_next_season"]).mean() if len(both) else float("nan")
    print("training season %+d -> matched %.1f%% of rows | label agreement %.1f%% on %d rows"
          % (off, 100 * found, 100 * agree, len(both)))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
lab.to_csv(OUT, index=False)
print("\nSaved", len(lab), "player-seasons to", OUT)
