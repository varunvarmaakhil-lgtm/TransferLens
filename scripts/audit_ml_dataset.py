import pandas as pd
from pathlib import Path


# ============================================================
# TRANSFERLENS - QUICK ML DATASET AUDIT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "ml_dataset_2025_26.csv"


print("=" * 60)
print("TRANSFERLENS - QUICK ML DATASET AUDIT")
print("=" * 60)


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ------------------------------------------------------------
# Target distribution
# ------------------------------------------------------------

print("\nTARGET DISTRIBUTION")
print("-" * 40)

print(df["transferred"].value_counts())
print("\nPercentage:")

print(
    df["transferred"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ------------------------------------------------------------
# Check duplicates
# ------------------------------------------------------------

print("\nDUPLICATES")
print("-" * 40)

print(
    f"Duplicate player names: "
    f"{df['player_name'].duplicated().sum()}"
)


# ------------------------------------------------------------
# Check missing values
# ------------------------------------------------------------

print("\nMISSING VALUES")
print("-" * 40)

missing = df.isnull().sum()

missing = missing[missing > 0]

if missing.empty:
    print("No missing values.")
else:
    print(missing)


# ------------------------------------------------------------
# Check suspicious transfer features
# ------------------------------------------------------------

print("\nTRANSFER-RELATED FEATURES")
print("-" * 40)

transfer_features = [
    "transfer_count",
    "total_transfer_fee",
    "transfer_fee",
    "max_market_value",
    "market_value_in_eur"
]

for column in transfer_features:
    if column in df.columns:
        print(
            f"{column}: "
            f"min={df[column].min()}, "
            f"max={df[column].max()}"
        )


# ------------------------------------------------------------
# Check whether transfer_count directly matches target
# ------------------------------------------------------------

if "transfer_count" in df.columns:

    print("\nTRANSFER COUNT vs TARGET")
    print("-" * 40)

    print(
        pd.crosstab(
            df["transfer_count"] > 0,
            df["transferred"],
            margins=True
        )
    )


# ------------------------------------------------------------
# Feature list
# ------------------------------------------------------------

print("\nALL FEATURES")
print("-" * 40)

for column in df.columns:
    print(f" - {column}")


# ------------------------------------------------------------
# Finish
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)