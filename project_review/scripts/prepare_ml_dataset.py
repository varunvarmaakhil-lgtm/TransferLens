import pandas as pd
from pathlib import Path


# ============================================================
# TRANSFERLENS - ML DATASET PREPARATION
# Season: 2025-26
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PLAYER_FILE = BASE_DIR / "data" / "processed" / "player_stats_2025_26.csv"
TRANSFER_FILE = BASE_DIR / "data" / "processed" / "transfer_history_2025_26.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "ml_dataset_2025_26.csv"


def main():

    print("=" * 60)
    print("TRANSFERLENS - ML DATASET PREPARATION")
    print("Season: 2025-26")
    print("=" * 60)


    # ========================================================
    # 1. LOAD PLAYER DATA
    # ========================================================

    print("\nLoading player data...")

    if not PLAYER_FILE.exists():
        print("ERROR: Player dataset not found.")
        print(PLAYER_FILE)
        return

    players = pd.read_csv(PLAYER_FILE)

    print(f"Players loaded: {len(players)}")


    # ========================================================
    # 2. LOAD TRANSFER DATA
    # ========================================================

    print("\nLoading transfer data...")

    if not TRANSFER_FILE.exists():
        print("ERROR: Transfer dataset not found.")
        print(TRANSFER_FILE)
        return

    transfers = pd.read_csv(TRANSFER_FILE)

    print(f"Transfers loaded: {len(transfers)}")


    # ========================================================
    # 3. CHECK REQUIRED COLUMNS
    # ========================================================

    print("\nChecking columns...")

    required_player_columns = [
        "player_name",
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
        "red_cards"
    ]

    required_transfer_columns = [
        "player_name",
        "from_club_name",
        "to_club_name",
        "transfer_date",
        "transfer_fee",
        "market_value_in_eur"
    ]

    missing_player = [
        col for col in required_player_columns
        if col not in players.columns
    ]

    missing_transfer = [
        col for col in required_transfer_columns
        if col not in transfers.columns
    ]

    if missing_player:
        print("\nERROR: Missing player columns:")
        print(missing_player)
        return

    if missing_transfer:
        print("\nERROR: Missing transfer columns:")
        print(missing_transfer)
        return

    print("All required columns found.")


    # ========================================================
    # 4. CLEAN PLAYER NAMES
    # ========================================================

    print("\nCleaning player names...")

    players["player_name"] = (
        players["player_name"]
        .astype(str)
        .str.strip()
    )

    transfers["player_name"] = (
        transfers["player_name"]
        .astype(str)
        .str.strip()
    )


    # ========================================================
    # 5. CLEAN NUMERIC PLAYER FEATURES
    # ========================================================

    print("Cleaning player statistics...")

    numeric_player_columns = [
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

    for column in numeric_player_columns:

        players[column] = pd.to_numeric(
            players[column],
            errors="coerce"
        )

        players[column] = players[column].fillna(0)


    # ========================================================
    # 6. CLEAN TRANSFER DATA
    # ========================================================

    print("Cleaning transfer data...")

    transfers["transfer_fee"] = pd.to_numeric(
        transfers["transfer_fee"],
        errors="coerce"
    ).fillna(0)

    transfers["market_value_in_eur"] = pd.to_numeric(
        transfers["market_value_in_eur"],
        errors="coerce"
    ).fillna(0)

    transfers["transfer_date"] = pd.to_datetime(
        transfers["transfer_date"],
        errors="coerce"
    )


    # ========================================================
    # 7. CREATE TRANSFER SUMMARY PER PLAYER
    # ========================================================
    #
    # The player dataset does not have player_id.
    # Therefore we match players using player_name.
    #
    # We group transfers by player_name so that one player
    # having multiple transfer records does NOT create
    # duplicate player rows in our ML dataset.
    # ========================================================

    print("Creating transfer summary...")

    transfer_summary = (
        transfers
        .groupby("player_name", as_index=False)
        .agg(
            transfer_count=("player_name", "count"),
            total_transfer_fee=("transfer_fee", "sum"),
            max_market_value=("market_value_in_eur", "max")
        )
    )

    transfer_summary["transferred"] = 1


    # ========================================================
    # 8. MERGE PLAYER + TRANSFER DATA
    # ========================================================

    print("Matching players with transfer records...")

    ml_data = players.merge(
        transfer_summary,
        on="player_name",
        how="left"
    )


    # ========================================================
    # 9. CREATE TARGET VARIABLE
    # ========================================================

    ml_data["transferred"] = (
        ml_data["transferred"]
        .fillna(0)
        .astype(int)
    )

    ml_data["transfer_count"] = (
        ml_data["transfer_count"]
        .fillna(0)
        .astype(int)
    )

    ml_data["total_transfer_fee"] = (
        ml_data["total_transfer_fee"]
        .fillna(0)
    )

    ml_data["max_market_value"] = (
        ml_data["max_market_value"]
        .fillna(0)
    )


    # ========================================================
    # 10. CREATE PERFORMANCE FEATURES
    # ========================================================

    print("Creating performance features...")

    ml_data["goals_per_90"] = (
        ml_data["goals"]
        / ml_data["minutes_played"].replace(0, pd.NA)
        * 90
    )

    ml_data["assists_per_90"] = (
        ml_data["assists"]
        / ml_data["minutes_played"].replace(0, pd.NA)
        * 90
    )

    ml_data["starts_ratio"] = (
        ml_data["starts"]
        / ml_data["appearances"].replace(0, pd.NA)
    )


    ml_data["goals_per_90"] = (
        ml_data["goals_per_90"]
        .fillna(0)
    )

    ml_data["assists_per_90"] = (
        ml_data["assists_per_90"]
        .fillna(0)
    )

    ml_data["starts_ratio"] = (
        ml_data["starts_ratio"]
        .fillna(0)
    )


    # ========================================================
    # 11. REMOVE DUPLICATE PLAYER RECORDS
    # ========================================================

    ml_data = ml_data.drop_duplicates(
        subset=["player_name"]
    )


    # ========================================================
    # 12. ADD SEASON
    # ========================================================

    ml_data["season"] = "2025-26"


    # ========================================================
    # 13. SAVE ML DATASET
    # ========================================================

    ml_data.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ========================================================
    # 14. SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("ML DATASET CREATED")
    print("=" * 60)

    print(f"Players: {len(ml_data)}")
    print(f"Columns: {len(ml_data.columns)}")

    transferred_count = ml_data["transferred"].sum()

    print(f"Transferred players: {transferred_count}")
    print(
        f"Non-transferred players: "
        f"{len(ml_data) - transferred_count}"
    )

    print("\nML Dataset columns:")

    for column in ml_data.columns:
        print(f" - {column}")

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nFirst 10 records:")

    print(
        ml_data.head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()