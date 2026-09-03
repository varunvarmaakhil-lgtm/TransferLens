import warnings
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")


# ============================================================
# TRANSFERLENS
# PLAYER ANALYSIS
# ============================================================

print("=" * 65)
print("                    TRANSFERLENS")
print("                 PLAYER ANALYSIS")
print("=" * 65)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "transfer_training_updated.csv"
)

TRANSFER_MODEL_PATH = (
    BASE_DIR
    / "transfer_model.joblib"
)

TRANSFER_INFO_PATH = (
    BASE_DIR
    / "transfer_model_info.joblib"
)


# ============================================================
# KNOWN CORRECTIONS
# ============================================================

KNOWN_DOB = {
    "jude bellingham": pd.Timestamp("2003-06-29")
}


KNOWN_STARTS = {
    ("jude bellingham", 2020): 41,
    ("jude bellingham", 2021): 36,
    ("jude bellingham", 2022): 40,
    ("jude bellingham", 2023): 38,
    ("jude bellingham", 2024): 29,
    ("jude bellingham", 2025): 22,
}


KNOWN_MARKET_VALUES = {
    ("jude bellingham", 2020): 27_000_000,
    ("jude bellingham", 2021): 75_000_000,
    ("jude bellingham", 2022): 110_000_000,
    ("jude bellingham", 2023): 180_000_000,
    ("jude bellingham", 2024): 180_000_000,
    ("jude bellingham", 2025): 130_000_000,
}


# ============================================================
# MARKET MODEL FEATURES
# ============================================================

MARKET_FEATURES = [
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
    "career_seasons_before",
]


MARKET_CATEGORICAL = [
    "position",
    "season",
]


# ============================================================
# FUNCTIONS
# ============================================================

def safe_divide(a, b):

    denominator = b.replace(
        0,
        np.nan
    )

    result = a.divide(
        denominator
    )

    return (
        result
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


def calculate_age(
    dob,
    season_year
):

    if pd.isna(dob):
        return np.nan

    if pd.isna(season_year):
        return np.nan

    try:

        dob = pd.to_datetime(
            dob,
            errors="coerce"
        )

        if pd.isna(dob):
            return np.nan

        season_start = datetime(
            int(season_year),
            8,
            1
        )

        age = (
            season_start.year
            - dob.year
        )

        if (
            season_start.month,
            season_start.day
        ) < (
            dob.month,
            dob.day
        ):
            age -= 1

        return age

    except Exception:

        return np.nan


def show(
    label,
    value
):

    print(
        f"{label:<25}: {value}"
    )


def format_money(value):

    if value is None:
        return "N/A"

    try:
        value = float(value)
    except Exception:
        return "N/A"

    if not np.isfinite(value):
        return "N/A"

    value = max(
        0,
        value
    )

    if value >= 1_000_000:

        return (
            f"€{value / 1_000_000:.2f}M"
        )

    if value >= 1_000:

        return (
            f"€{value / 1_000:.0f}K"
        )

    return (
        f"€{value:,.0f}"
    )


# ============================================================
# LOAD TRANSFER MODEL
# ============================================================

print(
    "\nLoading transfer model..."
)

if not TRANSFER_MODEL_PATH.exists():

    print(
        "\n❌ Transfer model not found."
    )

    print(
        f"Expected: {TRANSFER_MODEL_PATH}"
    )

    raise SystemExit


try:

    transfer_model = joblib.load(
        TRANSFER_MODEL_PATH
    )

    print(
        "✓ Transfer model loaded."
    )

except Exception as e:

    print(
        "\n❌ Could not load transfer model."
    )

    print(
        "Error:",
        e
    )

    raise SystemExit


# ============================================================
# TRANSFER MODEL INFORMATION
# ============================================================

transfer_info = {}


if TRANSFER_INFO_PATH.exists():

    try:

        transfer_info = joblib.load(
            TRANSFER_INFO_PATH
        )

    except Exception:

        transfer_info = {}


transfer_features = (
    transfer_info.get("features")
    or transfer_info.get(
        "feature_columns"
    )
    or []
)


transfer_categorical = (
    transfer_info.get(
        "categorical_features"
    )
    or transfer_info.get(
        "categorical_columns"
    )
    or []
)


transfer_threshold = float(
    transfer_info.get(
        "threshold",
        0.50
    )
)


if not transfer_features:

    try:

        transfer_features = list(
            transfer_model.feature_names_
        )

    except Exception:

        transfer_features = []


if not transfer_features:

    print(
        "\n❌ Could not determine transfer-model features."
    )

    raise SystemExit


transfer_features = list(
    dict.fromkeys(
        transfer_features
    )
)


transfer_categorical = list(
    dict.fromkeys(
        transfer_categorical
    )
)


print(
    f"✓ Transfer model expects "
    f"{len(transfer_features)} features."
)


# ============================================================
# LOAD DATASET
# ============================================================

print(
    "\nLoading player database..."
)


if not DATA_PATH.exists():

    print(
        "\n❌ Dataset not found."
    )

    print(
        f"Expected: {DATA_PATH}"
    )

    raise SystemExit


df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)


print(
    f"✓ Records loaded: "
    f"{len(df):,}"
)


# ============================================================
# BASIC COLUMNS
# ============================================================

if "name" not in df.columns:

    if "player_name" in df.columns:

        df["name"] = (
            df["player_name"]
        )

    else:

        print(
            "\n❌ Player name column not found."
        )

        raise SystemExit


df["name"] = (
    df["name"]
    .fillna("")
    .astype(str)
    .str.strip()
)


if "season" not in df.columns:

    if "season_year" in df.columns:

        df["season"] = (
            df["season_year"]
            .astype(str)
        )

    else:

        print(
            "\n❌ Season column not found."
        )

        raise SystemExit


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
    "starts",
    "minutes_played",
    "goals",
    "assists",
    "penalties_scored",
    "yellow_cards",
    "red_cards",
    "market_value_in_eur",
]


for column in numeric_columns:

    if column not in df.columns:

        df[column] = 0

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0)


# ============================================================
# POSITION
# ============================================================

if "position" not in df.columns:

    if "Pos" in df.columns:

        df["position"] = (
            df["Pos"]
        )

    else:

        df["position"] = "Unknown"


df["position"] = (
    df["position"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)


# ============================================================
# DATE OF BIRTH
# ============================================================

if "date_of_birth" in df.columns:

    df["date_of_birth"] = (
        pd.to_datetime(
            df["date_of_birth"],
            errors="coerce"
        )
    )

else:

    df["date_of_birth"] = pd.NaT


def recover_dob(row):

    player = (
        str(row["name"])
        .strip()
        .lower()
    )

    if pd.notna(
        row["date_of_birth"]
    ):

        return row["date_of_birth"]

    return KNOWN_DOB.get(
        player,
        pd.NaT
    )


df["date_of_birth"] = df.apply(
    recover_dob,
    axis=1
)


# ============================================================
# AGE
# ============================================================

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


df["age"] = (
    df["age"]
    .clip(
        lower=15,
        upper=45
    )
    .fillna(25)
)


# ============================================================
# PERFORMANCE FEATURES
# ============================================================

print(
    "\nPreparing player statistics..."
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


df["minutes_per_appearance"] = (
    safe_divide(
        df["minutes_played"],
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


df["starts_ratio"] = (
    safe_divide(
        df["starts"],
        df["appearances"]
    )
)


df["cards_per_appearance"] = (
    safe_divide(
        df["yellow_cards"]
        + df["red_cards"],
        df["appearances"]
    )
)


# ============================================================
# DISPLAY STARTS
# ============================================================

def get_display_starts(row):

    key = (
        str(row["name"])
        .strip()
        .lower(),
        int(row["season_year"])
        if pd.notna(
            row["season_year"]
        )
        else -1
    )

    if key in KNOWN_STARTS:

        return KNOWN_STARTS[key]

    value = row.get(
        "starts",
        np.nan
    )

    if pd.notna(value):

        return int(value)

    return np.nan


df["display_starts"] = df.apply(
    get_display_starts,
    axis=1
)


# ============================================================
# DISPLAY MARKET VALUE
# ============================================================

def get_display_market_value(row):

    key = (
        str(row["name"])
        .strip()
        .lower(),
        int(row["season_year"])
        if pd.notna(
            row["season_year"]
        )
        else -1
    )

    if key in KNOWN_MARKET_VALUES:

        return KNOWN_MARKET_VALUES[key]

    value = pd.to_numeric(
        row.get(
            "market_value_in_eur",
            np.nan
        ),
        errors="coerce"
    )

    if (
        pd.notna(value)
        and value > 0
    ):

        return float(value)

    return np.nan


df["display_market_value"] = (
    df.apply(
        get_display_market_value,
        axis=1
    )
)


# ============================================================
# PLAYER SEARCH
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "                    ENTER PLAYER"
)

print(
    "=" * 65
)


search = input(
    "\nPlayer name: "
).strip()


if not search:

    print(
        "\n❌ Please enter a player name."
    )

    raise SystemExit


matches = df[
    df["name"].str.contains(
        search,
        case=False,
        na=False,
        regex=False
    )
].copy()


if matches.empty:

    print(
        "\n❌ PLAYER NOT FOUND"
    )

    raise SystemExit


# ============================================================
# PLAYERS FOUND
# ============================================================

player_names = (
    matches["name"]
    .drop_duplicates()
    .tolist()
)


print(
    "\n" + "=" * 65
)

print(
    "                  PLAYERS FOUND"
)

print(
    "=" * 65
)


for i, name in enumerate(
    player_names,
    1
):

    print(
        f"{i}. {name}"
    )


# ============================================================
# SELECT PLAYER
# ============================================================

if len(player_names) == 1:

    selected_name = (
        player_names[0]
    )

else:

    while True:

        try:

            choice = int(
                input(
                    "\nSelect player number: "
                )
            )

            if (
                1 <= choice
                <= len(player_names)
            ):

                selected_name = (
                    player_names[
                        choice - 1
                    ]
                )

                break

        except ValueError:

            pass

        print(
            "❌ Invalid selection."
        )


player_matches = matches[
    matches["name"]
    == selected_name
].copy()


# ============================================================
# AVAILABLE SEASONS
# ============================================================

seasons = sorted(
    player_matches[
        "season_year"
    ]
    .dropna()
    .astype(int)
    .unique()
)


if not seasons:

    print(
        "\n❌ No seasons available."
    )

    raise SystemExit


print(
    "\nAvailable seasons:\n"
)


for i, season in enumerate(
    seasons,
    1
):

    print(
        f"{i}. "
        f"{season}-"
        f"{str(season + 1)[-2:]} "
        f"({season})"
    )


# ============================================================
# SELECT SEASON
# ============================================================

while True:

    choice = input(
        "\nSelect season "
        "(number or year, e.g. 2025): "
    ).strip()

    try:

        choice_int = int(
            choice
        )

        if (
            1 <= choice_int
            <= len(seasons)
        ):

            selected_season = (
                seasons[
                    choice_int - 1
                ]
            )

            break

        if choice_int in seasons:

            selected_season = (
                choice_int
            )

            break

    except ValueError:

        pass

    print(
        "Please select a valid season."
    )


season_label = (
    f"{selected_season}-"
    f"{str(selected_season + 1)[-2:]}"
)


# ============================================================
# YEAR-BY-YEAR STATS
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "                  YEAR-BY-YEAR STATS"
)

print(
    "=" * 65
)


print(
    f"\n{'Season':<12}"
    f"{'Apps':<7}"
    f"{'Starts':<8}"
    f"{'Minutes':<10}"
    f"{'Goals':<8}"
    f"{'Assists':<9}"
    f"{'YC':<6}"
    f"{'RC':<6}"
    f"Market Value"
)


print(
    "-" * 90
)


for season in seasons:

    season_data = (
        player_matches[
            player_matches[
                "season_year"
            ] == season
        ]
    )

    if season_data.empty:
        continue

    r = season_data.iloc[0]

    apps = int(
        r.get(
            "appearances",
            0
        )
    )

    starts = int(
        r.get(
            "display_starts",
            r.get(
                "starts",
                0
            )
        )
    )

    minutes = int(
        r.get(
            "minutes_played",
            0
        )
    )

    goals = int(
        r.get(
            "goals",
            0
        )
    )

    assists = int(
        r.get(
            "assists",
            0
        )
    )

    yellow = int(
        r.get(
            "yellow_cards",
            0
        )
    )

    red = int(
        r.get(
            "red_cards",
            0
        )
    )

    mv = r.get(
        "display_market_value",
        np.nan
    )

    if (
        pd.isna(mv)
        or mv <= 0
    ):

        mv_text = "N/A"

    else:

        mv_text = format_money(
            mv
        )

    print(
        f"{season}-"
        f"{str(season + 1)[-2:]:<5}"
        f"{apps:<7}"
        f"{starts:<8}"
        f"{minutes:<10}"
        f"{goals:<8}"
        f"{assists:<9}"
        f"{yellow:<6}"
        f"{red:<6}"
        f"{mv_text}"
    )


# ============================================================
# SELECTED PLAYER ROW
# ============================================================

player_data = player_matches[
    player_matches[
        "season_year"
    ] == selected_season
]


if player_data.empty:

    print(
        "\n❌ No data found."
    )

    raise SystemExit


row = player_data.iloc[0]


# ============================================================
# PLAYER INFORMATION
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "                 PLAYER INFORMATION"
)

print(
    "=" * 65
)


show(
    "Player",
    row["name"]
)


show(
    "Position",
    row.get(
        "position",
        "N/A"
    )
)


show(
    "Nationality",
    row.get(
        "country_of_birth",
        "N/A"
    )
)


show(
    "Season",
    season_label
)


display_age = row.get(
    "age",
    np.nan
)


if pd.notna(display_age):

    show(
        "Age",
        int(display_age)
    )

else:

    show(
        "Age",
        "N/A"
    )


# ============================================================
# PLAYER STATS
# ============================================================

show(
    "Appearances",
    int(
        row.get(
            "appearances",
            0
        )
    )
)


show(
    "Starts",
    int(
        row.get(
            "display_starts",
            row.get(
                "starts",
                0
            )
        )
    )
)


show(
    "Minutes played",
    int(
        row.get(
            "minutes_played",
            0
        )
    )
)


show(
    "Goals",
    int(
        row.get(
            "goals",
            0
        )
    )
)


show(
    "Assists",
    int(
        row.get(
            "assists",
            0
        )
    )
)


show(
    "Penalties scored",
    int(
        row.get(
            "penalties_scored",
            0
        )
    )
)


show(
    "Yellow cards",
    int(
        row.get(
            "yellow_cards",
            0
        )
    )
)


show(
    "Red cards",
    int(
        row.get(
            "red_cards",
            0
        )
    )
)


display_market_value = row.get(
    "display_market_value",
    np.nan
)


show(
    "Market value",
    format_money(
        display_market_value
    )
)


# ============================================================
# ENGINEERED STATS
# ============================================================

show(
    "Goals per 90",
    f"{row.get('goals_per_90', 0):.4f}"
)


show(
    "Assists per 90",
    f"{row.get('assists_per_90', 0):.4f}"
)


show(
    "Goals per appearance",
    f"{row.get('goals_per_appearance', 0):.4f}"
)


show(
    "Assists per appearance",
    f"{row.get('assists_per_appearance', 0):.4f}"
)


show(
    "Minutes per appearance",
    f"{row.get('minutes_per_appearance', 0):.2f}"
)


show(
    "Starts ratio",
    f"{safe_divide(
        pd.Series([row.get('display_starts', 0)]),
        pd.Series([row.get('appearances', 0)])
    ).iloc[0]:.4f}"
)


show(
    "Cards per appearance",
    f"{row.get('cards_per_appearance', 0):.4f}"
)


show(
    "Minutes/game equivalent",
    f"{row.get('minutes_played', 0) / 90:.2f}"
)


# ============================================================
# TRANSFER MODEL INPUT
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "                 PREPARING TRANSFER MODEL"
)

print(
    "=" * 65
)


X_transfer = pd.DataFrame(
    index=[0]
)


for feature in transfer_features:

    if feature in row.index:

        X_transfer[
            feature
        ] = [
            row[feature]
        ]

    else:

        X_transfer[
            feature
        ] = [0]


# ============================================================
# TRANSFER CATEGORICAL
# ============================================================

for column in transfer_categorical:

    if column in X_transfer.columns:

        X_transfer[column] = (
            X_transfer[column]
            .fillna("Unknown")
            .astype(str)
        )


# ============================================================
# TRANSFER NUMERIC
# ============================================================

for column in X_transfer.columns:

    if (
        column
        not in transfer_categorical
    ):

        X_transfer[column] = (
            pd.to_numeric(
                X_transfer[column],
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )


print(
    f"✓ Transfer model input prepared: "
    f"{len(X_transfer.columns)} features"
)


# ============================================================
# TRANSFER PREDICTION
# ============================================================

print(
    "\nGenerating transfer prediction..."
)


try:

    transfer_prediction = (
        transfer_model
        .predict(
            X_transfer
        )[0]
    )

    transfer_probabilities = (
        transfer_model
        .predict_proba(
            X_transfer
        )[0]
    )

    transfer_probability = float(
        transfer_probabilities[1]
    )

except Exception as e:

    print(
        "\n❌ TRANSFER PREDICTION FAILED"
    )

    print(
        "\nError:"
    )

    print(e)

    raise SystemExit


# ============================================================
# TRANSFER RESULT
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "                 TRANSFER PREDICTION"
)

print(
    "=" * 65
)


show(
    "Player",
    row["name"]
)


show(
    "Prediction",
    (
        "LIKELY TO BE TRANSFERRED"
        if transfer_prediction == 1
        else
        "NOT LIKELY TO BE TRANSFERRED"
    )
)


show(
    "Transfer probability",
    f"{transfer_probability * 100:.2f}%"
)


print(
    "\nINTERPRETATION"
)

print(
    "--------------------------------"
)


if transfer_probability >= 0.75:

    print(
        "Very high transfer likelihood."
    )

elif transfer_probability >= 0.55:

    print(
        "High transfer likelihood."
    )

elif transfer_probability >= 0.40:

    print(
        "Moderate transfer likelihood."
    )

elif transfer_probability >= 0.25:

    print(
        "Low transfer likelihood."
    )

else:

    print(
        "Very low transfer likelihood."
    )


# ============================================================
# MARKET VALUE MODEL
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "              MARKET VALUE PREDICTION"
)

print(
    "=" * 65
)


print(
    "\nBuilding historical market-value dataset..."
)


# ============================================================
# IMPORTANT FIX:
# USE PLAYER NAME AS HISTORY KEY
# ============================================================

PLAYER_KEY = (
    "__player_history_key"
)


df[PLAYER_KEY] = (
    df["name"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# SORT HISTORY
# ============================================================

history = (
    df.copy()
    .sort_values(
        [
            PLAYER_KEY,
            "season_year"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# MARKET VALUES
# ============================================================

history[
    "model_market_value"
] = (
    history[
        "display_market_value"
    ]
)


# ============================================================
# PREVIOUS MARKET VALUE
# ============================================================

history[
    "previous_market_value"
] = (
    history
    .groupby(
        PLAYER_KEY
    )[
        "model_market_value"
    ]
    .shift(1)
)


# ============================================================
# TWO SEASONS BACK
# ============================================================

history[
    "two_back_market_value"
] = (
    history
    .groupby(
        PLAYER_KEY
    )[
        "model_market_value"
    ]
    .shift(2)
)


# ============================================================
# PREVIOUS CHANGE
# ============================================================

history[
    "previous_market_value_change"
] = (
    history[
        "previous_market_value"
    ]
    -
    history[
        "two_back_market_value"
    ]
)


# ============================================================
# PREVIOUS GROWTH
# ============================================================

history[
    "previous_market_value_growth"
] = safe_divide(
    history[
        "previous_market_value_change"
    ],
    history[
        "two_back_market_value"
    ]
)


# ============================================================
# CAREER SEASONS
# ============================================================

history[
    "career_seasons_before"
] = (
    history
    .groupby(
        PLAYER_KEY
    )
    .cumcount()
)


# ============================================================
# CLEAN HISTORY
# ============================================================

history_numeric = [
    "previous_market_value",
    "previous_market_value_change",
    "previous_market_value_growth",
    "career_seasons_before",
]


for column in history_numeric:

    history[column] = (
        pd.to_numeric(
            history[column],
            errors="coerce"
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


# ============================================================
# TRAINING DATA
# ============================================================
#
# IMPORTANT:
# We only train on seasons BEFORE the selected season.
#
# This prevents the selected season's actual market value
# from leaking into the prediction model.
# ============================================================

market_training = history[
    (
        history[
            "model_market_value"
        ] > 0
    )
    &
    (
        history[
            "previous_market_value"
        ] > 0
    )
    &
    (
        history[
            "season_year"
        ] < selected_season
    )
].copy()


print(
    f"✓ Historical training rows: "
    f"{len(market_training):,}"
)


print(
    f"✓ Market model features: "
    f"{len(MARKET_FEATURES)}"
)


# ============================================================
# PREPARE MISSING FEATURES
# ============================================================

for feature in MARKET_FEATURES:

    if feature not in history.columns:

        if (
            feature
            in MARKET_CATEGORICAL
        ):

            history[feature] = (
                "Unknown"
            )

        else:

            history[feature] = 0


    if feature not in market_training.columns:

        market_training[feature] = (
            history.loc[
                market_training.index,
                feature
            ]
        )


# ============================================================
# CATEGORICAL CLEANING
# ============================================================

for column in MARKET_CATEGORICAL:

    history[column] = (
        history[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    market_training[column] = (
        market_training[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


# ============================================================
# NUMERIC CLEANING
# ============================================================

for column in MARKET_FEATURES:

    if (
        column
        not in MARKET_CATEGORICAL
    ):

        history[column] = (
            pd.to_numeric(
                history[column],
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )

        market_training[column] = (
            pd.to_numeric(
                market_training[column],
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )


# ============================================================
# VALUE RATIO
# ============================================================

market_training[
    "value_ratio"
] = (
    market_training[
        "model_market_value"
    ]
    /
    market_training[
        "previous_market_value"
    ]
)


# ============================================================
# REMOVE EXTREME TRANSITIONS
# ============================================================

market_training = (
    market_training[
        (
            market_training[
                "value_ratio"
            ] >= 0.20
        )
        &
        (
            market_training[
                "value_ratio"
            ] <= 5.00
        )
    ]
    .copy()
)


# ============================================================
# LOG VALUE RATIO
# ============================================================

market_training[
    "log_value_ratio"
] = np.log(
    market_training[
        "value_ratio"
    ]
)


print(
    f"✓ Historical transitions: "
    f"{len(market_training):,}"
)


# ============================================================
# CHECK TRAINING DATA
# ============================================================

if len(market_training) < 100:

    print(
        "\n❌ Not enough historical market-value "
        "transitions to train the model."
    )

    print(
        "The player may not have enough historical "
        "market-value data."
    )

    raise SystemExit


# ============================================================
# TRAIN MARKET MODEL
# ============================================================

print(
    "\nTraining market-value model..."
)


X_market_train = (
    market_training[
        MARKET_FEATURES
    ]
    .copy()
)


y_market_train = (
    market_training[
        "log_value_ratio"
    ]
    .astype(float)
)


market_model = CatBoostRegressor(

    iterations=1800,

    depth=7,

    learning_rate=0.03,

    loss_function="RMSE",

    l2_leaf_reg=12,

    random_strength=0.20,

    random_seed=42,

    verbose=False,

    allow_writing_files=False
)


market_model.fit(

    X_market_train,

    y_market_train,

    cat_features=MARKET_CATEGORICAL
)


print(
    "✓ Market-value model trained."
)


# ============================================================
# SELECTED PLAYER HISTORY
# ============================================================

selected_player_key = (
    str(row["name"])
    .strip()
    .lower()
)


selected_player_history = (
    history[
        history[
            PLAYER_KEY
        ]
        ==
        selected_player_key
    ]
    .copy()
)


# ============================================================
# PREVIOUS SEASONS
# ============================================================

previous_rows = (
    selected_player_history[
        selected_player_history[
            "season_year"
        ]
        <
        selected_season
    ]
    .sort_values(
        "season_year"
    )
)


# ============================================================
# PREVIOUS MARKET VALUE
# ============================================================

previous_market_value = 0.0


if not previous_rows.empty:

    valid_values = (
        previous_rows[
            "model_market_value"
        ]
        .dropna()
    )

    valid_values = valid_values[
        valid_values > 0
    ]


    if not valid_values.empty:

        previous_market_value = float(
            valid_values.iloc[-1]
        )


# ============================================================
# TWO BACK VALUE
# ============================================================

previous_two_value = 0.0


if len(previous_rows) >= 2:

    valid_values = (
        previous_rows[
            "model_market_value"
        ]
        .dropna()
    )

    valid_values = valid_values[
        valid_values > 0
    ]


    if len(valid_values) >= 2:

        previous_two_value = float(
            valid_values.iloc[-2]
        )


# ============================================================
# PREVIOUS CHANGE
# ============================================================

previous_market_value_change = (
    previous_market_value
    -
    previous_two_value
)


# ============================================================
# PREVIOUS GROWTH
# ============================================================

if previous_two_value > 0:

    previous_market_value_growth = (
        previous_market_value_change
        /
        previous_two_value
    )

else:

    previous_market_value_growth = 0.0


# ============================================================
# CAREER SEASONS
# ============================================================

career_seasons_before = len(
    previous_rows
)


# ============================================================
# MARKET MODEL INPUT
# ============================================================

X_market = pd.DataFrame(
    index=[0]
)


for feature in MARKET_FEATURES:

    if (
        feature
        == "previous_market_value"
    ):

        value = (
            previous_market_value
        )

    elif (
        feature
        == "previous_market_value_change"
    ):

        value = (
            previous_market_value_change
        )

    elif (
        feature
        == "previous_market_value_growth"
    ):

        value = (
            previous_market_value_growth
        )

    elif (
        feature
        == "career_seasons_before"
    ):

        value = (
            career_seasons_before
        )

    else:

        value = row.get(
            feature,
            0
        )

    X_market[
        feature
    ] = [
        value
    ]


# ============================================================
# CORRECT AGE
# ============================================================

market_age = calculate_age(
    row.get(
        "date_of_birth"
    ),
    selected_season
)


if pd.notna(market_age):

    X_market["age"] = [
        float(market_age)
    ]

else:

    X_market["age"] = [
        25.0
    ]


# ============================================================
# CATEGORICAL VALUES
# ============================================================

for column in MARKET_CATEGORICAL:

    X_market[column] = (
        X_market[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


# ============================================================
# NUMERIC VALUES
# ============================================================

for column in MARKET_FEATURES:

    if (
        column
        not in MARKET_CATEGORICAL
    ):

        X_market[column] = (
            pd.to_numeric(
                X_market[column],
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )


# ============================================================
# EXACT FEATURE ORDER
# ============================================================

X_market = (
    X_market[
        MARKET_FEATURES
    ]
)


print(
    f"✓ Market model input prepared: "
    f"{len(X_market.columns)} features."
)


print(
    f"✓ Previous market value: "
    f"{format_money(previous_market_value)}"
)


print(
    f"✓ Market-value history: "
    f"{career_seasons_before} previous seasons"
)


# ============================================================
# MARKET VALUE PREDICTION
# ============================================================

print(
    "\nGenerating market-value prediction..."
)


try:

    predicted_log_ratio = float(
        market_model.predict(
            X_market
        )[0]
    )


    predicted_ratio = float(
        np.exp(
            predicted_log_ratio
        )
    )


    # Keep one-season changes realistic.
    predicted_ratio = float(
        np.clip(
            predicted_ratio,
            0.55,
            1.45
        )
    )


    model_prediction = (
        previous_market_value
        *
        predicted_ratio
    )


    # Historical value is deliberately
    # given strong weight for established players.

    if career_seasons_before >= 3:

        previous_weight = 0.20

    elif career_seasons_before == 2:

        previous_weight = 0.15

    else:

        previous_weight = 0.10


    model_weight = (
        1.0
        -
        previous_weight
    )


    predicted_market_value = (
        model_weight
        *
        model_prediction
        +
        previous_weight
        *
        previous_market_value
    )


except Exception as e:

    print(
        "\n❌ MARKET VALUE PREDICTION FAILED"
    )

    print(
        "\nError:"
    )

    print(e)

    raise SystemExit


# ============================================================
# FINAL SANITY CHECK
# ============================================================

if not np.isfinite(
    predicted_market_value
):

    predicted_market_value = (
        previous_market_value
    )


predicted_market_value = max(
    0.0,
    float(
        predicted_market_value
    )
)


# ============================================================
# PREDICTED CHANGE
# ============================================================

if previous_market_value > 0:

    predicted_change_percentage = (
        (
            predicted_market_value
            -
            previous_market_value
        )
        /
        previous_market_value
    ) * 100

else:

    predicted_change_percentage = 0.0


print(
    f"✓ Predicted change from previous value: "
    f"{predicted_change_percentage:+.2f}%"
)


# ============================================================
# MARKET VALUE RESULT
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "              MARKET VALUE RESULT"
)

print(
    "=" * 65
)


show(
    "Player",
    row["name"]
)


show(
    "Predicted market value",
    format_money(
        predicted_market_value
    )
)


# ============================================================
# ACTUAL VALUE COMPARISON
# ============================================================

if (
    pd.notna(
        display_market_value
    )
    and
    display_market_value > 0
):

    difference = (
        predicted_market_value
        -
        display_market_value
    )


    percentage_difference = (
        difference
        /
        display_market_value
    ) * 100


    absolute_percentage_difference = (
        abs(
            percentage_difference
        )
    )


    show(
        "Actual market value",
        format_money(
            display_market_value
        )
    )


    show(
        "Prediction difference",
        f"€{difference:,.0f}"
    )


    show(
        "Percentage difference",
        f"{percentage_difference:+.2f}%"
    )


    # ========================================================
    # ACCURACY
    # ========================================================

    print(
        "\nACCURACY"
    )

    print(
        "--------------------------------"
    )


    if (
        absolute_percentage_difference
        <= 5
    ):

        print(
            "Excellent prediction — "
            "very close to actual value."
        )

    elif (
        absolute_percentage_difference
        <= 10
    ):

        print(
            "Very good prediction — "
            "close to actual value."
        )

    elif (
        absolute_percentage_difference
        <= 20
    ):

        print(
            "Good prediction — "
            "reasonably close to actual value."
        )

    elif (
        absolute_percentage_difference
        <= 30
    ):

        print(
            "Moderate prediction — "
            "noticeable difference."
        )

    else:

        print(
            "Prediction needs improvement — "
            "large difference."
        )


# ============================================================
# MARKET VALUE INTERPRETATION
# ============================================================

print(
    "\nINTERPRETATION"
)

print(
    "--------------------------------"
)


if predicted_market_value >= 150_000_000:

    print(
        "Very high / elite market-value range."
    )

elif predicted_market_value >= 100_000_000:

    print(
        "Elite market-value range."
    )

elif predicted_market_value >= 50_000_000:

    print(
        "High market-value range."
    )

elif predicted_market_value >= 20_000_000:

    print(
        "Strong market-value range."
    )

else:

    print(
        "Lower market-value range."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

print(
    "\nMODEL INFORMATION"
)

print(
    "--------------------------------"
)


show(
    "Algorithm",
    "CatBoost Regressor"
)


show(
    "Target",
    "Log(current value / previous value)"
)


show(
    "Training method",
    "Historical market-value transitions"
)


show(
    "Previous value",
    "Included"
)


show(
    "Performance",
    "Included"
)


show(
    "Age",
    "Included"
)


show(
    "Features",
    len(MARKET_FEATURES)
)


# ============================================================
# NOTE
# ============================================================

print(
    "\nNOTE"
)

print(
    "--------------------------------"
)


print(
    f"{season_label} is the selected season."
)


if previous_market_value > 0:

    print(
        "Market-value prediction uses the "
        "player's previous market value as "
        "the historical baseline."
    )

    print(
        "The model predicts the relative "
        "one-season change using performance, "
        "age, position and career information."
    )

else:

    print(
        "No previous market value was available, "
        "so prediction confidence may be lower."
    )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "                 ANALYSIS COMPLETE"
)

print(
    "=" * 65
)


print(
    "\n✓ Existing transfer model used."
)

print(
    "✓ Historical market-value model trained."
)

print(
    "✓ Stable player history key used."
)

print(
    "✓ Previous market value included."
)

print(
    "✓ Relative market-value prediction used."
)

print(
    "✓ Performance statistics included."
)

print(
    "✓ Age calculated from date of birth."
)

print(
    "✓ Starts corrections applied where available."
)

print(
    "✓ Market-value corrections preserved."
)

print(
    "✓ Partial player search enabled."
)

print(
    "✓ Multiple player matches supported."
)

print(
    "✓ Categorical/numeric features handled correctly."
)

print(
    "✓ Market-value prediction completed."
)

print(
    "=" * 65
)