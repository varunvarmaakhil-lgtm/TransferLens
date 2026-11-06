import os
import duckdb
import numpy as np
import pandas as pd

DB = "/Users/varunvarma/Transferlens/data/raw/transfermarkt-datasets.duckdb"
D = "/Users/varunvarma/Transferlens/data/processed/"
FV2, OUT = D + "features_v2.csv", D + "player_seasons.csv"

con = duckdb.connect(DB, read_only=True)
app = con.execute(
    "SELECT player_id, competition_id, date, goals, assists, yellow_cards, red_cards, minutes_played "
    "FROM appearances").fetchdf()
pl = con.execute(
    "SELECT player_id, name, date_of_birth, position, sub_position, foot, height_in_cm, "
    "country_of_citizenship FROM players").fetchdf()
clubs = con.execute("SELECT club_id, name AS club_name FROM clubs").fetchdf()
con.close()
clubs["club_id"] = pd.to_numeric(clubs["club_id"], errors="coerce")

d = pd.to_datetime(app["date"])
app["season"] = d.dt.year - (d.dt.month < 7).astype(int)
for c in ["goals", "assists", "yellow_cards", "red_cards", "minutes_played"]:
    app[c] = pd.to_numeric(app[c], errors="coerce").fillna(0)

per = app.groupby(["player_id", "season"]).agg(
    goals=("goals", "sum"), assists=("assists", "sum"), yellow=("yellow_cards", "sum"),
    red=("red_cards", "sum"), minutes_all=("minutes_played", "sum"),
    apps_all=("minutes_played", "size")).reset_index()
comp = (app.groupby(["player_id", "season", "competition_id"])["minutes_played"].sum()
           .reset_index().sort_values("minutes_played", ascending=False)
           .drop_duplicates(["player_id", "season"])
           .rename(columns={"competition_id": "main_comp"})[["player_id", "season", "main_comp"]])

fv = pd.read_csv(FV2)
lab = pd.read_csv(D + "labels_from_appearances.csv", usecols=["player_id", "season", "club_id"])
b = (fv.merge(lab, on=["player_id", "season"], how="left")
       .merge(per, on=["player_id", "season"], how="left")
       .merge(comp, on=["player_id", "season"], how="left")
       .merge(pl, on="player_id", how="left")
       .merge(clubs, on="club_id", how="left"))

dob = pd.to_datetime(b["date_of_birth"], errors="coerce")
start = pd.to_datetime(b["season"].astype(int).astype(str) + "-09-01")
b["age"] = (start - dob).dt.days / 365.25
m90 = b["minutes_all"].where(b["minutes_all"] >= 270)
b["goals_p90"] = b["goals"] / m90 * 90
b["assists_p90"] = b["assists"] / m90 * 90
b["ga_p90"] = (b["goals"] + b["assists"]) / m90 * 90

b = b.sort_values(["player_id", "season"]).reset_index(drop=True)
g = b.groupby("player_id")
for k in (1, 2):
    ok = g["season"].shift(k) == b["season"] - k
    for c in ["mv_end", "minutes_all", "ga_p90", "minutes_share"]:
        b["%s_prev%d" % (c, k)] = g[c].shift(k).where(ok)
b["mv_peak_prev"] = g["mv_end"].transform(lambda s: s.shift(1).cummax())
b["seasons_in_data"] = g.cumcount() + 1

b.to_csv(OUT, index=False)
print("player_seasons: %d rows, %d columns, seasons %d-%d"
      % (len(b), b.shape[1], b["season"].min(), b["season"].max()))
print("rows with a market value (target):", int(b["mv_end"].notna().sum()))
print("Saved", OUT)
