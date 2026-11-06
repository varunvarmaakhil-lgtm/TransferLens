import pandas as pd
import os

# ============================================================
# TRANSFERLENS - BUILD TRAINING DATA
# ============================================================

OLD_FILE = "data/transfer_training.csv"
NEW_PLAYER_FILE = "data/processed/player_stats_2025_26.csv"
NEW_TRANSFER_FILE = "data/processed/transfer_history_2025_26.csv"

OUTPUT_FILE = "data/transfer_training_updated.csv"

print("=" * 60)
print("TRANSFERLENS - BUILDING TRAINING DATA")
print("=" * 60)

# ------------------------------------------------------------
# 1. LOAD OLD DATA
# ------------------------------------------------------------

old_df = pd.read_csv(OLD_FILE)

print(f"\nOld training data: {len(old_df):,} rows")

# ------------------------------------------------------------
# 2. LOAD 2025-26 DATA
# ------------------------------------------------------------

players = pd.read_csv(NEW_PLAYER_FILE)
transfers = pd.read_csv(NEW_TRANSFER_FILE)

print(f"New player data:   {len(players):,} rows")
print(f"Transfer history:  {len(transfers):,} rows")

# ------------------------------------------------------------
# 3. CLEAN PLAYER NAMES
# ------------------------------------------------------------

players["player_name_clean"] = (
    players["player_name"]
    .astype(str)
    .str.strip()
    .str.lower()
)

transfers["player_name_clean"] = (
    transfers["player_name"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# ------------------------------------------------------------
# 4. DETERMINE WHICH 2025-26 PLAYERS TRANSFERRED
# ------------------------------------------------------------

# A player is considered transferred if they appear
# in the transfer history for the same season.

transferred_players = set(
    transfers["player_name_clean"].dropna()
)

players["transferred_next_season"] = (
    players["player_name_clean"]
    .isin(transferred_players)
    .astype(int)
)

# ------------------------------------------------------------
# 5. CONVERT 2025-26 PLAYER DATA TO OLD FORMAT
# ------------------------------------------------------------

new_df = pd.DataFrame()

# Basic player information
new_df["player_id"] = pd.NA
new_df["season"] = players["season"]
new_df["appearances"] = players["appearances"]
new_df["minutes_played"] = players["minutes_played"]
new_df["goals"] = players["goals"]
new_df["assists"] = players["assists"]
new_df["yellow_cards"] = players["yellow_cards"]
new_df["red_cards"] = players["red_cards"]

# Player information
new_df["name"] = players["player_name"]
new_df["position"] = players["position"]
new_df["country_of_birth"] = players["nationality"]

# Date of birth is not available in the new dataset
new_df["date_of_birth"] = pd.NA

# Market value
if "market_value_in_eur" in players.columns:
    new_df["market_value_in_eur"] = players["market_value_in_eur"]
else:
    new_df["market_value_in_eur"] = 0

# Season year
# 2025-26 -> 2025
new_df["season_year"] = (
    players["season"]
    .astype(str)
    .str[:4]
    .astype(int)
)

# Transfer label
new_df["transferred_next_season"] = players[
    "transferred_next_season"
]

# ------------------------------------------------------------
# 6. MAKE SURE COLUMNS MATCH OLD DATA
# ------------------------------------------------------------

required_columns = old_df.columns.tolist()

for column in required_columns:
    if column not in new_df.columns:
        new_df[column] = pd.NA

# Keep exactly the same column order
new_df = new_df[required_columns]

# ------------------------------------------------------------
# 7. REMOVE DUPLICATES
# ------------------------------------------------------------

# Avoid adding the same player/season twice.

old_keys = set(
    zip(
        old_df["name"].astype(str).str.lower(),
        old_df["season_year"]
    )
)

new_df["key"] = list(
    zip(
        new_df["name"].astype(str).str.lower(),
        new_df["season_year"]
    )
)

new_df = new_df[
    ~new_df["key"].isin(old_keys)
].copy()

new_df.drop(columns=["key"], inplace=True)

# ------------------------------------------------------------
# 8. COMBINE DATASETS
# ------------------------------------------------------------

updated_df = pd.concat(
    [old_df, new_df],
    ignore_index=True
)

# ------------------------------------------------------------
# 9. CLEAN DATA
# ------------------------------------------------------------

updated_df["season_year"] = pd.to_numeric(
    updated_df["season_year"],
    errors="coerce"
)

updated_df = updated_df.dropna(
    subset=["season_year"]
)

updated_df["season_year"] = (
    updated_df["season_year"].astype(int)
)

# ------------------------------------------------------------
# 10. SAVE
# ------------------------------------------------------------

os.makedirs("data", exist_ok=True)

updated_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# 11. RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET UPDATED")
print("=" * 60)

print(f"Old rows:   {len(old_df):,}")
print(f"New rows:   {len(new_df):,}")
print(f"Final rows:  {len(updated_df):,}")

print(
    f"\nSeason range: "
    f"{updated_df['season_year'].min()} - "
    f"{updated_df['season_year'].max()}"
)

print("\nTransfer distribution:")
print(
    updated_df["transferred_next_season"]
    .value_counts()
    .sort_index()
)

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\nLatest seasons:")
print(
    updated_df["season_year"]
    .value_counts()
    .sort_index()
    .tail(8)
)

print("\nDone!")