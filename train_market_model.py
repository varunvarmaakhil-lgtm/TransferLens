import warnings
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


# ============================================================
# TRANSFERLENS
# MARKET VALUE MODEL V2
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "transfer_training_updated.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "market_value_model.joblib"
)

INFO_PATH = (
    BASE_DIR
    / "market_value_model_info.joblib"
)

TARGET = "market_value_in_eur"

RANDOM_STATE = 42


# ============================================================
# EXACTLY 18 FEATURES
# ============================================================

FEATURES = [

    "position",
    "season",
    "season_year",
    "age",

    "appearances",
    "minutes_played",
    "minutes_per_appearance",
    "starts_ratio",

    "goals",
    "assists",
    "goals_per_90",
    "assists_per_90",

    "goals_per_appearance",
    "assists_per_appearance",

    "previous_market_value",
    "previous_market_value_change",
    "previous_market_value_growth",

    "career_seasons_before"
]


CATEGORICAL_FEATURES = [
    "position",
    "season"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("                    TRANSFERLENS")
print("             MARKET VALUE MODEL V2")
print("=" * 65)


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )


print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print(
    f"Dataset loaded: {df.shape}"
)


# ============================================================
# BASIC COLUMNS
# ============================================================

if TARGET not in df.columns:

    raise ValueError(
        f"Missing target column: {TARGET}"
    )


if "name" not in df.columns:

    if "player_name" in df.columns:

        df["name"] = df["player_name"]

    else:

        raise ValueError(
            "Dataset needs name or player_name."
        )


if "player_id" in df.columns:

    PLAYER_KEY = "player_id"

else:

    PLAYER_KEY = "name"


if "position" not in df.columns:

    df["position"] = "Unknown"


# ============================================================
# SEASON
# ============================================================

if "season" not in df.columns:

    if "season_year" in df.columns:

        df["season"] = (
            df["season_year"]
            .astype(str)
        )

    else:

        raise ValueError(
            "Dataset needs season or season_year."
        )


if "season_year" not in df.columns:

    df["season_year"] = (
        df["season"]
        .astype(str)
        .str.extract(
            r"(\d{4})"
        )[0]
    )


df["season_year"] = pd.to_numeric(
    df["season_year"],
    errors="coerce"
)


# ============================================================
# NUMERIC COLUMNS
# ============================================================

numeric_columns = [

    "appearances",
    "minutes_played",
    "starts",
    "goals",
    "assists",
    TARGET
]


for column in numeric_columns:

    if column not in df.columns:

        df[column] = 0

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0)


# ============================================================
# DATE OF BIRTH
# ============================================================

if "date_of_birth" in df.columns:

    df["date_of_birth"] = pd.to_datetime(
        df["date_of_birth"],
        errors="coerce"
    )

else:

    df["date_of_birth"] = pd.NaT


# ============================================================
# AGE
# ============================================================

print("\nBuilding age feature...")


def calculate_age(
    dob,
    season_year
):

    if pd.isna(dob):

        return np.nan

    if pd.isna(season_year):

        return np.nan

    try:

        start = pd.Timestamp(
            year=int(season_year),
            month=8,
            day=1
        )

        age = (
            start.year
            - dob.year
        )

        if (
            start.month,
            start.day
        ) < (
            dob.month,
            dob.day
        ):

            age -= 1

        return age

    except Exception:

        return np.nan


df["age"] = df.apply(

    lambda r:
        calculate_age(
            r["date_of_birth"],
            r["season_year"]
        ),

    axis=1
)


df["age"] = pd.to_numeric(
    df["age"],
    errors="coerce"
)


df["age"] = df["age"].clip(
    15,
    45
)


df["age"] = df["age"].fillna(
    25
)


# ============================================================
# SAFE DIVISION
# ============================================================

def safe_divide(
    a,
    b
):

    b = b.replace(
        0,
        np.nan
    )

    result = (
        a.divide(b)
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    return result


# ============================================================
# PERFORMANCE FEATURES
# ============================================================

print(
    "Building performance features..."
)


df["minutes_per_appearance"] = (
    safe_divide(
        df["minutes_played"],
        df["appearances"]
    )
)


df["starts_ratio"] = (
    safe_divide(
        df["starts"],
        df["appearances"]
    )
)


df["goals_per_90"] = (
    safe_divide(
        df["goals"] * 90,
        df["minutes_played"]
    )
)


df["assists_per_90"] = (
    safe_divide(
        df["assists"] * 90,
        df["minutes_played"]
    )
)


df["goals_per_appearance"] = (
    safe_divide(
        df["goals"],
        df["appearances"]
    )
)


df["assists_per_appearance"] = (
    safe_divide(
        df["assists"],
        df["appearances"]
    )
)


# ============================================================
# TARGET CLEANING
# ============================================================

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)


df = df.dropna(
    subset=[
        TARGET,
        "season_year"
    ]
).copy()


df[TARGET] = df[TARGET].clip(
    lower=0
)


# ============================================================
# PLAYER HISTORY
# ============================================================

print(
    "\nBuilding player market-value history..."
)


df[PLAYER_KEY] = (
    df[PLAYER_KEY]
    .fillna("")
    .astype(str)
)


df = df.sort_values(
    [
        PLAYER_KEY,
        "season_year"
    ]
).reset_index(
    drop=True
)


# Previous season's market value

df["previous_market_value"] = (

    df.groupby(
        PLAYER_KEY
    )[TARGET]
    .shift(1)
)


# Market-value change

previous_two = (

    df.groupby(
        PLAYER_KEY
    )[TARGET]
    .shift(2)
)


df["previous_market_value_change"] = (

    df["previous_market_value"]
    - previous_two
)


# Percentage growth

df["previous_market_value_growth"] = (

    safe_divide(
        df[
            "previous_market_value_change"
        ],
        previous_two.abs()
    )
)


# Number of seasons before current season

df["career_seasons_before"] = (

    df.groupby(
        PLAYER_KEY
    ).cumcount()
)


# ============================================================
# CLEAN HISTORY FEATURES
# ============================================================

history_columns = [

    "previous_market_value",
    "previous_market_value_change",
    "previous_market_value_growth",
    "career_seasons_before"
]


for column in history_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    df[column] = (
        df[column]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

for column in CATEGORICAL_FEATURES:

    df[column] = (
        df[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


# ============================================================
# NUMERIC FEATURES
# ============================================================

for column in FEATURES:

    if column not in CATEGORICAL_FEATURES:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df[column] = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )


# ============================================================
# ONLY USE ROWS WITH HISTORY
# ============================================================

train_df = df[
    df["previous_market_value"] > 0
].copy()


if len(train_df) < 100:

    raise ValueError(
        "Not enough player seasons with "
        "previous market values."
    )


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
# ============================================================

latest_season = int(
    train_df[
        "season_year"
    ].max()
)


test_df = train_df[
    train_df["season_year"]
    == latest_season
].copy()


train_df_old = train_df[
    train_df["season_year"]
    < latest_season
].copy()


# Fallback if newest season is too small

if (
    len(test_df) < 50
    or len(train_df_old) < 100
):

    cutoff = int(
        train_df[
            "season_year"
        ].quantile(0.80)
    )

    train_df_old = train_df[
        train_df["season_year"]
        < cutoff
    ].copy()

    test_df = train_df[
        train_df["season_year"]
        >= cutoff
    ].copy()


X_train = train_df_old[
    FEATURES
].copy()


X_test = test_df[
    FEATURES
].copy()


# ============================================================
# LOG TARGET
# ============================================================

y_train = np.log1p(
    train_df_old[
        TARGET
    ].values
)


y_test = np.log1p(
    test_df[
        TARGET
    ].values
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n" + "=" * 65)
print("MODEL SETUP")
print("=" * 65)


print(
    f"\nPlayer key       : {PLAYER_KEY}"
)


print(
    f"Training rows    : {len(X_train):,}"
)


print(
    f"Testing rows     : {len(X_test):,}"
)


print(
    f"Latest season    : {latest_season}"
)


print(
    f"Total features   : {len(FEATURES)}"
)


print("\nFeatures:")

for feature in FEATURES:

    print(
        f" - {feature}"
    )


# ============================================================
# CATBOOST
# ============================================================

print("\n" + "=" * 65)
print("MODEL TRAINING")
print("=" * 65)


print(
    "\nTraining CatBoost market-value model..."
)


model = CatBoostRegressor(

    iterations=2500,

    depth=8,

    learning_rate=0.035,

    loss_function="RMSE",

    eval_metric="RMSE",

    l2_leaf_reg=8,

    random_strength=0.5,

    random_seed=RANDOM_STATE,

    od_type="Iter",

    od_wait=150,

    verbose=250,

    allow_writing_files=False
)


model.fit(

    X_train,

    y_train,

    cat_features=CATEGORICAL_FEATURES,

    eval_set=(
        X_test,
        y_test
    ),

    use_best_model=True
)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("MODEL EVALUATION")
print("=" * 65)


predicted_log = model.predict(
    X_test
)


predictions = np.expm1(
    predicted_log
)


actual = np.expm1(
    y_test
)


predictions = np.maximum(
    predictions,
    0
)


mae = mean_absolute_error(
    actual,
    predictions
)


rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)


r2 = r2_score(
    actual,
    predictions
)


median_ape = np.median(

    np.abs(
        predictions
        - actual
    )

    /

    np.maximum(
        actual,
        1
    )

) * 100


print(
    f"\nMAE        : €{mae:,.0f}"
)


print(
    f"RMSE       : €{rmse:,.0f}"
)


print(
    f"R²         : {r2:.4f}"
)


print(
    f"Median APE : {median_ape:.2f}%"
)


# ============================================================
# SAVE MODEL
# ============================================================

print(
    "\nSaving market-value model..."
)


joblib.dump(
    model,
    MODEL_PATH
)


model_info = {

    "model_type":
        "CatBoostRegressor",

    "target":
        TARGET,

    "features":
        FEATURES,

    "feature_columns":
        FEATURES,

    "categorical_features":
        CATEGORICAL_FEATURES,

    "player_key":
        PLAYER_KEY,

    "target_transform":
        "log1p",

    "latest_training_season":
        latest_season,

    "mae":
        float(mae),

    "rmse":
        float(rmse),

    "r2":
        float(r2),

    "median_ape":
        float(median_ape),

    "random_state":
        RANDOM_STATE
}


joblib.dump(
    model_info,
    INFO_PATH
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 65)
print("              MARKET MODEL COMPLETE")
print("=" * 65)


print(
    "\n✓ Previous market value included."
)


print(
    "✓ Market-value trend included."
)


print(
    "✓ Log-transformed target used."
)


print(
    "✓ Time-based validation used."
)


print(
    "✓ Exactly 18 features used."
)


print(
    f"\n✓ Model saved:"
)


print(
    MODEL_PATH
)


print(
    "\n✓ Model information saved:"
)


print(
    INFO_PATH
)