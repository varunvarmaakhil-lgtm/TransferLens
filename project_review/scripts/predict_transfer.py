import pandas as pd
import joblib

# ============================================================
# TRANSFERLENS - PLAYER PREDICTION SYSTEM
# ============================================================

MODEL_PATH = "models/transfer_prediction_model.pkl"
DATA_PATH = "data/processed/ml_dataset_2025_26.csv"

print("=" * 60)
print("TRANSFERLENS - PLAYER PREDICTION SYSTEM")
print("Season: 2025-26")
print("=" * 60)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

print("\nLoading player dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Players loaded: {len(df)}")

# ------------------------------------------------------------
# Ask for player
# ------------------------------------------------------------

player_name = input("\nEnter player name: ").strip()

# Case-insensitive search
matches = df[
    df["player_name"]
    .astype(str)
    .str.lower()
    .str.contains(player_name.lower(), na=False)
]

if matches.empty:
    print(f"\nPlayer '{player_name}' not found.")
    print("Try entering part of the player's name.")
    exit()

# Use first matching player
player = matches.iloc[0]

# ------------------------------------------------------------
# PLAYER PROFILE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PLAYER PROFILE")
print("=" * 60)

print(f"Name:          {player['player_name']}")
print(f"Nationality:   {player['nationality']}")
print(f"Position:      {player['position']}")
print(f"Club:          {player['club']}")
print(f"League:        {player['league']}")
print(f"Age:           {player['age']}")

print("\nPERFORMANCE")
print("-" * 40)

print(f"Appearances:   {player['appearances']}")
print(f"Starts:        {player['starts']}")
print(f"Minutes:       {player['minutes_played']}")
print(f"Goals:         {player['goals']}")
print(f"Assists:       {player['assists']}")
print(f"Goals/90:      {player['goals_per_90']:.2f}")
print(f"Assists/90:    {player['assists_per_90']:.2f}")

print("\nDISCIPLINE")
print("-" * 40)

print(f"Yellow cards:  {player['yellow_cards']}")
print(f"Red cards:     {player['red_cards']}")

print("\nMARKET / TRANSFER DATA")
print("-" * 40)

print(f"Market value:  €{player['max_market_value']:,.0f}")
print(f"Transfers:     {player['transfer_count']}")

# ------------------------------------------------------------
# Prepare features for model
# ------------------------------------------------------------

categorical_features = [
    "nationality",
    "position",
    "club",
    "league"
]

numerical_features = [
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

feature_columns = categorical_features + numerical_features

X_player = pd.DataFrame([player[feature_columns].to_dict()])

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRANSFER PREDICTION")
print("=" * 60)

prediction = model.predict(X_player)[0]

probabilities = model.predict_proba(X_player)[0]

# Find probability of transferred class
classes = model.classes_

if 1 in classes:
    transferred_index = list(classes).index(1)
    transfer_probability = probabilities[transferred_index]
else:
    transfer_probability = probabilities[-1]

if prediction == 1:
    result = "LIKELY TO BE TRANSFERRED"
else:
    result = "NOT LIKELY TO BE TRANSFERRED"

print(f"\nPrediction: {result}")
print(f"Transfer probability: {transfer_probability * 100:.2f}%")

print("\n" + "=" * 60)
print("END")
print("=" * 60)