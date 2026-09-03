import os
import joblib
import pandas as pd


# ============================================================
# TRANSFERLENS — PLAYER TRANSFER PREDICTION
# ============================================================

MODEL_PATH = "models/best_tuned_transfer_model.pkl"
DATASET_PATH = "data/processed/ml_dataset_2025_26.csv"

FEATURES = [
    "nationality",
    "position",
    "club",
    "league",
    "season",
    "age",
    "appearances",
    "starts",
    "minutes_played",
    "goals",
    "assists",
    "penalties_scored",
    "yellow_cards",
    "red_cards",
    "max_market_value",
    "goals_per_90",
    "assists_per_90",
    "starts_ratio"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("TRANSFERLENS — TRANSFER PREDICTION")
print("Season: 2025-26")
print("=" * 60)


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("\nLoading trained model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading player database...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print(f"Players loaded: {len(df)}")


# ============================================================
# 3. FIND PLAYER NAME COLUMN
# ============================================================

possible_name_columns = [
    "player_name",
    "player",
    "name"
]

name_column = None

for column in possible_name_columns:
    if column in df.columns:
        name_column = column
        break

if name_column is None:
    raise ValueError(
        "Could not find player name column. "
        "Expected one of: player_name, player, name"
    )


# ============================================================
# 4. GET PLAYER NAME
# ============================================================

print("\n" + "-" * 60)
print("ENTER PLAYER")
print("-" * 60)

player_name = input("Player name: ").strip()

if not player_name:
    raise ValueError("Player name cannot be empty.")


# ============================================================
# 5. SEARCH FOR PLAYER
# ============================================================

# Case-insensitive exact match
matches = df[
    df[name_column]
    .astype(str)
    .str.strip()
    .str.lower()
    == player_name.lower()
]


# ============================================================
# 6. IF EXACT MATCH NOT FOUND
# ============================================================

if len(matches) == 0:

    # Try partial search
    partial_matches = df[
        df[name_column]
        .astype(str)
        .str.lower()
        .str.contains(player_name.lower(), na=False)
    ]

    if len(partial_matches) == 0:

        print("\nPlayer not found.")

        print("\nTry checking the spelling or use:")
        print("python3 scripts/predict.py")

        raise SystemExit

    elif len(partial_matches) == 1:

        matches = partial_matches

    else:

        print("\nMultiple players found:")

        for i, name in enumerate(
            partial_matches[name_column].drop_duplicates(),
            start=1
        ):
            print(f"{i}. {name}")

        choice = input("\nEnter player number: ").strip()

        try:
            choice = int(choice)

            selected_name = (
                partial_matches[name_column]
                .drop_duplicates()
                .iloc[choice - 1]
            )

            matches = partial_matches[
                partial_matches[name_column] == selected_name
            ]

        except (ValueError, IndexError):

            raise ValueError("Invalid player selection.")


# ============================================================
# 7. SELECT PLAYER ROW
# ============================================================

player_row = matches.iloc[0]

actual_player_name = player_row[name_column]


# ============================================================
# 8. CHECK REQUIRED FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    print("\nMissing features in dataset:")

    for feature in missing_features:
        print(f"- {feature}")

    raise ValueError(
        "\nDataset does not contain all required model features."
    )


# ============================================================
# 9. CREATE MODEL INPUT
# ============================================================

player_data = pd.DataFrame(
    [player_row[FEATURES].to_dict()]
)


# ============================================================
# 10. DISPLAY PLAYER INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("PLAYER FOUND")
print("=" * 60)

print(f"Player             : {actual_player_name}")
print(f"Nationality        : {player_row['nationality']}")
print(f"Position           : {player_row['position']}")
print(f"Club               : {player_row['club']}")
print(f"League             : {player_row['league']}")
print(f"Season             : {player_row['season']}")
print(f"Age                : {player_row['age']}")
print(f"Appearances        : {player_row['appearances']}")
print(f"Starts             : {player_row['starts']}")
print(f"Minutes played     : {player_row['minutes_played']}")
print(f"Goals              : {player_row['goals']}")
print(f"Assists            : {player_row['assists']}")
print(f"Penalties scored   : {player_row['penalties_scored']}")
print(f"Yellow cards       : {player_row['yellow_cards']}")
print(f"Red cards          : {player_row['red_cards']}")
print(f"Market value       : {player_row['max_market_value']}")
print(f"Goals per 90       : {player_row['goals_per_90']}")
print(f"Assists per 90     : {player_row['assists_per_90']}")
print(f"Starts ratio       : {player_row['starts_ratio']}")


# ============================================================
# 11. GENERATE PREDICTION
# ============================================================

print("\nGenerating prediction...")

prediction = model.predict(player_data)[0]

probability = model.predict_proba(
    player_data
)[0][1]


# ============================================================
# 12. RESULT
# ============================================================

print("\n" + "=" * 60)
print("TRANSFER PREDICTION")
print("=" * 60)

print(f"Player               : {actual_player_name}")

if prediction == 1:

    print(
        "Prediction           : LIKELY TO BE TRANSFERRED"
    )

else:

    print(
        "Prediction           : NOT LIKELY TO BE TRANSFERRED"
    )

print(
    f"Transfer probability : {probability * 100:.2f}%"
)


# ============================================================
# 13. INTERPRETATION
# ============================================================

print("\nINTERPRETATION")
print("-" * 40)

if probability >= 0.80:

    print("Very high transfer likelihood.")

elif probability >= 0.60:

    print("High transfer likelihood.")

elif probability >= 0.40:

    print("Moderate transfer likelihood.")

elif probability >= 0.20:

    print("Low transfer likelihood.")

else:

    print("Very low transfer likelihood.")


# ============================================================
# 14. COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)