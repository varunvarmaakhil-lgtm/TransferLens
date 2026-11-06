import pandas as pd
import duckdb
from pathlib import Path


# ============================================================
# TRANSFERLENS - TRANSFER DATASET BUILDER
# Season: 2025-26
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "transfermarkt-datasets.duckdb"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "transfer_history_2025_26.csv"
)


def main():

    print("=" * 60)
    print("TRANSFERLENS - TRANSFER DATASET BUILDER")
    print("Season: 2025-26")
    print("=" * 60)


    # --------------------------------------------------------
    # Check database
    # --------------------------------------------------------

    if not DATABASE_FILE.exists():

        print("\nERROR: Transfer database not found.")
        print(DATABASE_FILE)
        return


    # --------------------------------------------------------
    # Connect to DuckDB
    # --------------------------------------------------------

    con = duckdb.connect(str(DATABASE_FILE), read_only=True)


    # --------------------------------------------------------
    # Check transfers table
    # --------------------------------------------------------

    print("\nLoading transfer data...")

    try:

        transfers = con.execute("""
            SELECT
                player_id,
                transfer_date,
                transfer_season,
                from_club_id,
                to_club_id,
                from_club_name,
                to_club_name,
                transfer_fee,
                market_value_in_eur,
                player_name
            FROM transfers
            WHERE transfer_season = '25/26'
        """).fetchdf()

    except Exception as e:

        print("\nERROR: Could not read transfers table.")
        print(e)
        con.close()
        return


    con.close()


    # --------------------------------------------------------
    # Check data
    # --------------------------------------------------------

    print(f"\nTransfers loaded: {len(transfers)}")


    if transfers.empty:

        print("\nERROR: No transfers found for 2025-26.")
        return


    # --------------------------------------------------------
    # Rename season
    # --------------------------------------------------------

    transfers["season"] = "2025-26"


    # --------------------------------------------------------
    # Clean text fields
    # --------------------------------------------------------

    text_columns = [
        "player_name",
        "from_club_name",
        "to_club_name"
    ]

    for column in text_columns:

        transfers[column] = (
            transfers[column]
            .astype(str)
            .str.strip()
        )


    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    transfers["transfer_date"] = pd.to_datetime(
        transfers["transfer_date"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Convert transfer fee
    # --------------------------------------------------------

    transfers["transfer_fee"] = pd.to_numeric(
        transfers["transfer_fee"],
        errors="coerce"
    ).fillna(0)


    # --------------------------------------------------------
    # Convert market value
    # --------------------------------------------------------

    transfers["market_value_in_eur"] = pd.to_numeric(
        transfers["market_value_in_eur"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Remove duplicate transfers
    # --------------------------------------------------------

    transfers = transfers.drop_duplicates(
        subset=[
            "player_id",
            "transfer_date",
            "from_club_id",
            "to_club_id"
        ]
    )


    # --------------------------------------------------------
    # Create transfer type
    #
    # The source dataset does not contain a reliable
    # transfer_type column, so we do NOT invent one.
    # --------------------------------------------------------

    transfers["transfer_type"] = "Unknown"


    # --------------------------------------------------------
    # Select final columns
    # --------------------------------------------------------

    transfers = transfers[
        [
            "player_id",
            "player_name",
            "from_club_id",
            "from_club_name",
            "to_club_id",
            "to_club_name",
            "transfer_date",
            "transfer_fee",
            "market_value_in_eur",
            "transfer_type",
            "season"
        ]
    ]


    # --------------------------------------------------------
    # Create output directory if needed
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------

    transfers.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRANSFER DATASET CREATED")
    print("=" * 60)

    print(f"Transfers: {len(transfers)}")
    print(f"Columns: {len(transfers.columns)}")

    print("\nColumns:")

    for column in transfers.columns:
        print(f" - {column}")

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nFirst 10 records:")

    print(
        transfers.head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()