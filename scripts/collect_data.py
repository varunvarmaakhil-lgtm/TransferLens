import pandas as pd
from pathlib import Path

# ============================================================
# FOOTBALL TRANSFER DATASET
# Season: 2025-26
# Source: Collected raw player statistics
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RAW_DIR / "players_data-2025_2026.csv"
OUTPUT_FILE = OUTPUT_DIR / "player_stats_2025_26.csv"


def clean_columns(df):

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(
                str(x).strip()
                for x in column
                if str(x).strip() != "nan"
            )
            for column in df.columns
        ]

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df


def main():

    print("=" * 60)
    print("FOOTBALL DATASET BUILDER")
    print("Season: 2025-26")
    print("=" * 60)

    # --------------------------------------------------------
    # Check raw file
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        print("\nERROR:")
        print(f"Raw data file not found:")
        print(INPUT_FILE)
        return

    print(f"\nReading:")
    print(INPUT_FILE)

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(f"\nRaw rows: {len(df)}")
    print(f"Raw columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df = clean_columns(df)

    # --------------------------------------------------------
    # Remove duplicate header rows
    # --------------------------------------------------------

    if "player" in df.columns:

        df = df[
            df["player"].astype(str).str.lower() != "player"
        ]

    # --------------------------------------------------------
    # Remove completely empty rows
    # --------------------------------------------------------

    df = df.dropna(how="all")

    # --------------------------------------------------------
    # Keep useful columns
    # --------------------------------------------------------

    wanted_columns = [
        "player",
        "nation",
        "pos",
        "squad",
        "comp",
        "age",
        "mp",
        "starts",
        "min",
        "gls",
        "ast",
        "pk",
        "crdy",
        "crdr"
    ]

    available_columns = [
        column
        for column in wanted_columns
        if column in df.columns
    ]

    df = df[available_columns].copy()

    # --------------------------------------------------------
    # Rename columns
    # --------------------------------------------------------

    rename_map = {

        "player": "player_name",
        "nation": "nationality",
        "pos": "position",
        "squad": "club",
        "comp": "league",

        "age": "age",
        "mp": "appearances",
        "starts": "starts",
        "min": "minutes_played",

        "gls": "goals",
        "ast": "assists",
        "pk": "penalties_scored",

        "crdy": "yellow_cards",
        "crdr": "red_cards"
    }

    df = df.rename(columns=rename_map)

    # --------------------------------------------------------
    # Clean player names
    # --------------------------------------------------------

    if "player_name" in df.columns:

        df["player_name"] = (
            df["player_name"]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Remove rows without player names
    # --------------------------------------------------------

    if "player_name" in df.columns:

        df = df[
            df["player_name"].notna()
        ]

        df = df[
            df["player_name"] != ""
        ]

    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    numeric_columns = [
        "age",
        "appearances",
        "starts",
        "minutes_played",
        "goals",
        "assists",
        "penalties_scored",
        "yellow_cards",
        "red_cards"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Add season
    # --------------------------------------------------------

    df["season"] = "2025-26"

    # --------------------------------------------------------
    # Save cleaned dataset
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET COMPLETE")
    print("=" * 60)

    print(f"Total players: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f" - {column}")

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nFirst 5 players:")
    print(
        df.head(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()