import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "player_stats_2025_26.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "player_stats_2025_26.csv"

df = pd.read_csv(INPUT_FILE)

# Add squad number column.
# We will fill the legitimate numbers in the next step.
if "squad_number" not in df.columns:
    df["squad_number"] = pd.NA

# Put squad number near the player information
columns = [
    "player_name",
    "squad_number",
    "nationality",
    "position",
    "club",
    "league",
    "age",
    "appearances",
    "starts",
    "minutes_played",
    "goals",
    "assists",
    "penalties_scored",
    "yellow_cards",
    "red_cards",
    "season"
]

df = df[[c for c in columns if c in df.columns]]

df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("SQUAD NUMBER FIELD ADDED")
print("=" * 60)
print(f"Players: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Saved: {OUTPUT_FILE}")
print("\nColumns:")
print(df.columns.tolist())