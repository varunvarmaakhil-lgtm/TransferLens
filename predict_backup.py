import warnings
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")


# ============================================================
# TRANSFERLENS
# PLAYER ANALYSIS + TRANSFER + MARKET VALUE
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

MARKET_MODEL_PATH = (
    BASE_DIR
    / "market_value_model.joblib"
)

MARKET_INFO_PATH = (
    BASE_DIR
    / "market_value_model_info.joblib"
)


# ============================================================
# LOAD TRANSFER MODEL
# ============================================================

print("\nLoading saved transfer model...")

if not TRANSFER_MODEL_PATH.exists():

    print("\n❌ Transfer model not found.")
    print(f"Expected: {TRANSFER_MODEL_PATH}")
    raise SystemExit

try:

    transfer_model = joblib.load(
        TRANSFER_MODEL_PATH
    )

    print("✓ Transfer model loaded successfully.")

except Exception as e:

    print("\n❌ Could not load transfer model.")
    print("Error:", e)
    raise SystemExit


# ============================================================
# LOAD TRANSFER MODEL INFORMATION
# ============================================================

if not TRANSFER_INFO_PATH.exists():

    print("\n❌ Transfer model information not found.")
    print(f"Expected: {TRANSFER_INFO_PATH}")
    raise SystemExit

try:

    transfer_info = joblib.load(
        TRANSFER_INFO_PATH
    )

    transfer_features = (
        transfer_info.get("features")
        or transfer_info.get("feature_columns")
    )

    transfer_categorical = (
        transfer_info.get("categorical_features")
        or transfer_info.get("categorical_columns")
        or []
    )

    transfer_threshold = (
        transfer_info.get(
            "threshold",
            0.50
        )
    )

except Exception as e:

    print(
        "\n❌ Could not load transfer "
        "model information."
    )

    print("Error:", e)
    raise SystemExit


# ============================================================
# FALLBACK FOR TRANSFER FEATURES
# ============================================================

if not transfer_features:

    try:

        transfer_features = list(
            transfer_model.feature_names_
        )

    except Exception:

        transfer_features = []


if not transfer_features:

    print(
        "\n❌ Could not determine "
        "transfer-model features."
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

print(
    f"✓ Categorical features: "
    f"{transfer_categorical}"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading player database...")

if not DATA_PATH.exists():

    print("\n❌ Dataset not found.")
    print(f"Expected: {DATA_PATH}")
    raise SystemExit

try:

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False
    )

except Exception as e:

    print("\n❌ Could not load dataset.")
    print("Error:", e)
    raise SystemExit


print(
    f"✓ Records loaded: "
    f"{len(df):,}"
)


# ============================================================
# BASIC CLEANING
# ============================================================

if "name" not in df.columns:

    if "player_name" in df.columns:

        df["name"] = df["player_name"]

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


# ============================================================
# SEASON YEAR
# ============================================================

if "season_year" not in df.columns:

    if "season" in df.columns:

        df["season_year"] = (
            df["season"]
            .astype(str)
            .str.extract(
                r"(\d{4})"
            )[0]
        )

    else:

        print(
            "\n❌ Season column not found."
        )

        raise SystemExit


df["season_year"] = pd.to_numeric(
    df["season_year"],
    errors="coerce"
)


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
    "market_value_in_eur"
]


for column in numeric_columns:

    if column not in df.columns:

        df[column] = 0

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0)


# ============================================================
# AGE CALCULATION
#
# Age is calculated at the START of the season:
# August 1 of the season year.
# ============================================================

def calculate_age(
    date_of_birth,
    season_year
):

    if pd.isna(date_of_birth):
        return np.nan

    if pd.isna(season_year):
        return np.nan

    try:

        dob = pd.to_datetime(
            date_of_birth,
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
            - (
                (
                    season_start.month,
                    season_start.day
                )
                <
                (
                    dob.month,
                    dob.day
                )
            )
        )

        return age

    except Exception:

        return np.nan


df["age_display"] = df.apply(
    lambda r:
    calculate_age(
        r["date_of_birth"],
        r["season_year"]
    ),
    axis=1
)


# ============================================================
# AGE FEATURES USED BY TRANSFER MODEL
# ============================================================

df["birth_year"] = (
    df["date_of_birth"].dt.year
)

df["birth_month"] = (
    df["date_of_birth"].dt.month
)


df["age_calculated"] = (
    df["season_year"]
    - df["birth_year"]
)


if "age" in df.columns:

    existing_age = pd.to_numeric(
        df["age"],
        errors="coerce"
    )

    df["age_numeric"] = (
        existing_age
        .fillna(
            df["age_calculated"]
        )
    )

else:

    df["age_numeric"] = (
        df["age_calculated"]
    )


df["age_numeric"] = (
    df["age_numeric"]
    .clip(
        lower=15,
        upper=45
    )
)


df["age_squared"] = (
    df["age_numeric"] ** 2
)

df["age_peak_distance"] = abs(
    df["age_numeric"] - 25
)

df["young_player"] = (
    df["age_numeric"] <= 23
).astype(int)

df["prime_age"] = (
    (
        df["age_numeric"] >= 24
    )
    &
    (
        df["age_numeric"] <= 29
    )
).astype(int)

df["older_player"] = (
    df["age_numeric"] >= 30
).astype(int)


# ============================================================
# SAFE DIVISION
# ============================================================

def safe_divide(a, b):

    result = a / b.replace(
        0,
        np.nan
    )

    return (
        result
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


# ============================================================
# PERFORMANCE FEATURES
# ============================================================

appearances_safe = (
    df["appearances"]
    .replace(0, np.nan)
)

minutes_safe = (
    df["minutes_played"]
    .replace(0, np.nan)
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


df["minutes_per_appearance"] = (
    safe_divide(
        df["minutes_played"],
        df["appearances"]
    )
)


df["goal_contributions"] = (
    df["goals"]
    + df["assists"]
)


df["goal_contributions_per_appearance"] = (
    safe_divide(
        df["goal_contributions"],
        df["appearances"]
    )
)


df["goal_contributions_per_90"] = (
    safe_divide(
        df["goal_contributions"] * 90,
        df["minutes_played"]
    )
)


# ============================================================
# STARTS FEATURES
# ============================================================

df["starts_ratio"] = (
    safe_divide(
        df["starts"],
        df["appearances"]
    )
)


# ============================================================
# DISCIPLINE FEATURES
# ============================================================

df["yellow_cards_per_appearance"] = (
    safe_divide(
        df["yellow_cards"],
        df["appearances"]
    )
)


df["red_cards_per_appearance"] = (
    safe_divide(
        df["red_cards"],
        df["appearances"]
    )
)


df["total_cards"] = (
    df["yellow_cards"]
    + df["red_cards"]
)


df["cards_per_appearance"] = (
    safe_divide(
        df["total_cards"],
        df["appearances"]
    )
)


# ============================================================
# MARKET VALUE FEATURES
# ============================================================

df["market_value_in_eur"] = (
    pd.to_numeric(
        df["market_value_in_eur"],
        errors="coerce"
    )
    .clip(lower=0)
)


df["market_value_log"] = np.log1p(
    df["market_value_in_eur"]
)


# ============================================================
# SORT FOR HISTORICAL FEATURES
# ============================================================

sort_columns = []

if "player_id" in df.columns:

    sort_columns.append(
        "player_id"
    )

if "season_year" in df.columns:

    sort_columns.append(
        "season_year"
    )


if sort_columns:

    df = (
        df
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )


# ============================================================
# PLAYER HISTORY FEATURES
# ============================================================

history_columns = [
    "appearances",
    "minutes_played",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "market_value_in_eur"
]


if "player_id" in df.columns:

    grouped = df.groupby(
        "player_id",
        sort=False
    )

    for col in history_columns:

        if col not in df.columns:
            continue

        df[
            f"previous_{col}"
        ] = grouped[col].shift(1)

        df[
            f"two_season_avg_{col}"
        ] = (
            df.groupby(
                "player_id"
            )[col]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(
                    2,
                    min_periods=1
                )
                .mean()
            )
        )

        df[
            f"three_season_avg_{col}"
        ] = (
            df.groupby(
                "player_id"
            )[col]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(
                    3,
                    min_periods=1
                )
                .mean()
            )
        )


# ============================================================
# PERFORMANCE TRENDS
# ============================================================

if "player_id" in df.columns:

    trend_columns = [
        "goals",
        "assists",
        "minutes_played",
        "appearances",
        "market_value_in_eur"
    ]

    for col in trend_columns:

        previous = (
            df.groupby(
                "player_id"
            )[col]
            .shift(1)
        )

        previous_two = (
            df.groupby(
                "player_id"
            )[col]
            .shift(2)
        )

        df[
            f"{col}_change"
        ] = (
            df[col]
            - previous
        )

        df[
            f"{col}_trend"
        ] = (
            previous
            - previous_two
        )


# ============================================================
# PREVIOUS TRANSFER HISTORY
# ============================================================

TARGET = "transferred_next_season"


if (
    "player_id" in df.columns
    and TARGET in df.columns
):

    df["previous_transfer"] = (
        df.groupby(
            "player_id"
        )[TARGET]
        .shift(1)
    )

    df["previous_transfer_rate"] = (
        df.groupby(
            "player_id"
        )[TARGET]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )

    df["previous_transfer_count"] = (
        df.groupby(
            "player_id"
        )[TARGET]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .sum()
        )
    )

    df["previous_seasons_count"] = (
        df.groupby(
            "player_id"
        )[TARGET]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .count()
        )
    )


# ============================================================
# CAREER FEATURES
# ============================================================

if "player_id" in df.columns:

    df["career_seasons_before"] = (
        df.groupby(
            "player_id"
        )
        .cumcount()
    )


# ============================================================
# MARKET VALUE RELATIVE FEATURES
# ============================================================

season_median_value = (
    df.groupby(
        "season_year"
    )["market_value_in_eur"]
    .transform("median")
)


df["market_value_vs_season_median"] = (
    safe_divide(
        df["market_value_in_eur"],
        season_median_value
    )
)


# ============================================================
# INTERACTION FEATURES
# ============================================================

df["age_performance"] = (
    df["age_numeric"]
    * df["goal_contributions_per_90"]
)


df["value_performance"] = (
    np.log1p(
        df["market_value_in_eur"]
    )
    * df["goal_contributions_per_90"]
)


df["minutes_value_efficiency"] = (
    safe_divide(
        df["market_value_in_eur"],
        df["minutes_played"]
    )
)


df["goal_assist_ratio"] = (
    safe_divide(
        df["goals"],
        df["assists"] + 1
    )
)


df["minutes_per_appearance_squared"] = (
    df["minutes_per_appearance"] ** 2
)


# ============================================================
# SEASON NUMERIC
# ============================================================

df["season_year_numeric"] = (
    pd.to_numeric(
        df["season_year"],
        errors="coerce"
    ).fillna(0)
)


# ============================================================
# CLEAN
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# CORRECTED DISPLAY STARTS
#
# These do NOT overwrite the raw "starts"
# used by the trained transfer model.
# ============================================================

KNOWN_STARTS = {

    ("jude bellingham", 2020): 41,

    ("jude bellingham", 2021): 36,

    ("jude bellingham", 2022): 40,

    ("jude bellingham", 2023): 38,

    ("jude bellingham", 2024): 29,

    ("jude bellingham", 2025): 22,
}


def get_display_starts(row):

    name = (
        str(row["name"])
        .strip()
        .lower()
    )

    season = row["season_year"]

    if pd.notna(season):

        key = (
            name,
            int(season)
        )

        if key in KNOWN_STARTS:

            return KNOWN_STARTS[key]

    return int(
        row.get(
            "starts",
            0
        )
    )


df["display_starts"] = (
    df.apply(
        get_display_starts,
        axis=1
    )
)


# ============================================================
# CORRECTED DISPLAY MARKET VALUES
#
# These are ONLY for displaying known values.
# They are NOT fed into the market-value prediction.
# ============================================================

KNOWN_MARKET_VALUES = {

    ("jude bellingham", 2025):
        130_000_000,
}


def get_display_market_value(row):

    name = (
        str(row["name"])
        .strip()
        .lower()
    )

    season = row["season_year"]

    if pd.notna(season):

        key = (
            name,
            int(season)
        )

        if key in KNOWN_MARKET_VALUES:

            return KNOWN_MARKET_VALUES[key]

    value = row.get(
        "market_value_in_eur",
        np.nan
    )

    if pd.isna(value) or value <= 0:

        return np.nan

    return float(value)


df["display_market_value"] = (
    df.apply(
        get_display_market_value,
        axis=1
    )
)


# ============================================================
# DISPLAY PERFORMANCE FEATURES
# ============================================================

df["starts_ratio_display"] = (
    safe_divide(
        df["display_starts"],
        df["appearances"]
    )
)


# ============================================================
# PLAYER SEARCH
# ============================================================

print("\n" + "=" * 65)
print("                    ENTER PLAYER")
print("=" * 65)


search = input(
    "\nPlayer name: "
).strip()


if not search:

    print(
        "\n❌ Please enter a player name."
    )

    raise SystemExit


# ============================================================
# PARTIAL PLAYER SEARCH
# ============================================================

matches = df[
    df["name"]
    .str.contains(
        search,
        case=False,
        na=False,
        regex=False
    )
]


if matches.empty:

    print(
        "\n❌ PLAYER NOT FOUND"
    )

    print(
        "Please check the spelling "
        "and try again."
    )

    raise SystemExit


# ============================================================
# UNIQUE PLAYER NAMES
# ============================================================

player_names = (
    matches["name"]
    .drop_duplicates()
    .tolist()
)


print("\n" + "=" * 65)
print("                  PLAYERS FOUND")
print("=" * 65)


for i, name in enumerate(
    player_names,
    start=1
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
                1
                <= choice
                <= len(player_names)
            ):

                selected_name = (
                    player_names[
                        choice - 1
                    ]
                )

                break

            print(
                "❌ Invalid selection."
            )

        except ValueError:

            print(
                "❌ Please enter a number."
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


print("\nAvailable seasons:\n")


for i, season in enumerate(
    seasons,
    start=1
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
            1
            <= choice_int
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

        print(
            "Please select a valid season."
        )

    except ValueError:

        print(
            "Please enter a valid number."
        )


season_label = (
    f"{selected_season}-"
    f"{str(selected_season + 1)[-2:]}"
)


# ============================================================
# YEAR-BY-YEAR STATS
# ============================================================

print("\n" + "=" * 65)
print("                  YEAR-BY-YEAR STATS")
print("=" * 65)


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

print("-" * 90)


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


    if pd.isna(mv) or mv <= 0:

        mv_text = "N/A"

    elif mv >= 1_000_000:

        mv_text = (
            f"€{mv / 1_000_000:.1f}M"
        )

    elif mv >= 1_000:

        mv_text = (
            f"€{mv / 1_000:.0f}K"
        )

    else:

        mv_text = (
            f"€{mv:,.0f}"
        )


    print(
        f"{season}-{str(season + 1)[-2:]:<5}"
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
# GET SELECTED PLAYER ROW
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


player = player_data.iloc[
    [0]
].copy()


row = player.iloc[0]


# ============================================================
# PLAYER INFORMATION
# ============================================================

print("\n" + "=" * 65)
print("                 PLAYER INFORMATION")
print("=" * 65)


def show(label, value):

    print(
        f"{label:<25}: {value}"
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


# ============================================================
# AGE
# ============================================================

display_age = row.get(
    "age_display",
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
# STATS
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


# ============================================================
# ACTUAL MARKET VALUE
# ============================================================

display_market_value = row.get(
    "display_market_value",
    np.nan
)


if (
    pd.notna(
        display_market_value
    )
    and display_market_value > 0
):

    if display_market_value >= 1_000_000:

        market_text = (
            f"€"
            f"{display_market_value / 1_000_000:.1f}"
            f"M"
        )

    elif display_market_value >= 1_000:

        market_text = (
            f"€"
            f"{display_market_value / 1_000:.0f}"
            f"K"
        )

    else:

        market_text = (
            f"€"
            f"{display_market_value:,.0f}"
        )

else:

    market_text = "N/A"


show(
    "Market value",
    market_text
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
    f"{row.get('starts_ratio_display', 0):.4f}"
)


show(
    "Cards per appearance",
    f"{row.get('cards_per_appearance', 0):.4f}"
)


show(
    "Minutes/game equivalent",
    f"{row.get('minutes_per_game_equivalent', row.get('minutes_played', 0) / 90):.2f}"
)


if "previous_transfer_rate" in row.index:

    show(
        "Previous transfer rate",
        f"{row.get('previous_transfer_rate', 0):.4f}"
    )


# ============================================================
# PREPARE TRANSFER MODEL INPUT
# ============================================================

print("\n" + "=" * 65)
print("                 PREPARING TRANSFER MODEL")
print("=" * 65)


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
# HANDLE TRANSFER CATEGORICAL FEATURES
# ============================================================

for column in transfer_categorical:

    if column in X_transfer.columns:

        X_transfer[column] = (
            X_transfer[column]
            .fillna("Unknown")
            .astype(str)
        )


# ============================================================
# HANDLE TRANSFER NUMERIC FEATURES
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
            .fillna(0)
        )


print(
    f"✓ Transfer model input prepared: "
    f"{len(X_transfer.columns)} features"
)


# ============================================================
# TRANSFER PREDICTION
# ============================================================

print("\nGenerating transfer prediction...")


try:

    transfer_prediction = (
        transfer_model.predict(
            X_transfer
        )[0]
    )

    transfer_probabilities = (
        transfer_model.predict_proba(
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

    print(
        "\nExpected transfer features:"
    )

    print(transfer_features)

    raise SystemExit


# ============================================================
# TRANSFER RESULT
# ============================================================

print("\n" + "=" * 65)
print("                 TRANSFER PREDICTION")
print("=" * 65)


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


print("\nINTERPRETATION")
print("--------------------------------")


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

print("\n" + "=" * 65)
print("              MARKET VALUE PREDICTION")
print("=" * 65)


if not MARKET_MODEL_PATH.exists():

    print(
        "\n❌ Market-value model not found."
    )

    print(
        f"Expected: {MARKET_MODEL_PATH}"
    )

    print(
        "\nRun this ONCE:"
    )

    print(
        "python3 train_market_model.py"
    )

    raise SystemExit


try:

    market_model = joblib.load(
        MARKET_MODEL_PATH
    )

    print(
        "✓ Market-value model loaded."
    )

except Exception as e:

    print(
        "\n❌ Could not load "
        "market-value model."
    )

    print(
        "Error:",
        e
    )

    raise SystemExit


# ============================================================
# LOAD MARKET MODEL INFO
# ============================================================

market_features = []
market_categorical = []


if MARKET_INFO_PATH.exists():

    try:

        market_info = joblib.load(
            MARKET_INFO_PATH
        )

        market_features = (
            market_info.get(
                "feature_columns"
            )
            or market_info.get(
                "features"
            )
            or market_info.get(
                "feature_columns_used"
            )
            or []
        )

        market_categorical = (
            market_info.get(
                "categorical_features"
            )
            or market_info.get(
                "categorical_columns"
            )
            or []
        )

    except Exception:

        market_features = []
        market_categorical = []


# ============================================================
# FALLBACK MARKET FEATURES
#
# These are the 20 features used by
# the market-value model we trained.
# ============================================================

if not market_features:

    market_features = [

        "name",

        "position",

        "country_of_birth",

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

        "yellow_cards",

        "red_cards",

        "cards_per_appearance",

        "player_id"

    ]


if not market_categorical:

    market_categorical = [
        "season"
    ]


market_features = list(
    dict.fromkeys(
        market_features
    )
)

market_categorical = list(
    dict.fromkeys(
        market_categorical
    )
)


print(
    f"✓ Market model expects "
    f"{len(market_features)} features."
)


# ============================================================
# PREPARE MARKET MODEL ROW
# ============================================================

market_row = row.copy()


# Correct market-model age
# Use the same age representation
# that the market model was trained with.

if "age" in market_features:

    if pd.notna(
        row.get(
            "age_display",
            np.nan
        )
    ):

        market_row["age"] = (
            row["age_display"]
        )

    elif pd.notna(
        row.get(
            "age_numeric",
            np.nan
        )
    ):

        market_row["age"] = (
            row["age_numeric"]
        )


# Make sure starts ratio exists.

market_row["starts_ratio"] = (
    row.get(
        "starts_ratio",
        row.get(
            "starts_ratio_display",
            0
        )
    )
)


# Make sure season exists.

if "season" not in market_row.index:

    market_row["season"] = (
        season_label
    )


# ============================================================
# CREATE MARKET MODEL INPUT
# ============================================================

X_market = pd.DataFrame(
    index=[0]
)


for feature in market_features:

    if feature in market_row.index:

        X_market[
            feature
        ] = [
            market_row[feature]
        ]

    else:

        X_market[
            feature
        ] = [0]


# ============================================================
# MARKET CATEGORICAL FEATURES
# ============================================================

for column in market_categorical:

    if column in X_market.columns:

        X_market[column] = (
            X_market[column]
            .fillna("Unknown")
            .astype(str)
        )


# ============================================================
# MARKET NUMERIC FEATURES
# ============================================================

for column in X_market.columns:

    if (
        column
        not in market_categorical
    ):

        X_market[column] = (
            pd.to_numeric(
                X_market[column],
                errors="coerce"
            )
            .fillna(0)
        )


# ============================================================
# IMPORTANT:
# DO NOT USE ACTUAL MARKET VALUE
# ============================================================

if (
    "market_value_in_eur"
    in X_market.columns
):

    X_market = X_market.drop(
        columns=[
            "market_value_in_eur"
        ]
    )

    # This should never happen with
    # the current 20-feature model.
    # It is here as a safety check.


print(
    f"✓ Market model input prepared: "
    f"{len(X_market.columns)} features"
)


# ============================================================
# MARKET VALUE PREDICTION
# ============================================================

print(
    "\nGenerating market-value prediction..."
)


try:

    predicted_market_value = float(
        market_model.predict(
            X_market
        )[0]
    )

except Exception as e:

    print(
        "\n❌ MARKET VALUE PREDICTION FAILED"
    )

    print(
        "\nError:"
    )

    print(e)

    print(
        "\nMarket model expected features:"
    )

    print(market_features)

    raise SystemExit


# Never allow a negative prediction.

predicted_market_value = max(
    0,
    predicted_market_value
)


# ============================================================
# MARKET VALUE DISPLAY
# ============================================================

if predicted_market_value >= 1_000_000:

    predicted_market_text = (
        f"€"
        f"{predicted_market_value / 1_000_000:.2f}"
        f"M"
    )

elif predicted_market_value >= 1_000:

    predicted_market_text = (
        f"€"
        f"{predicted_market_value / 1_000:.0f}"
        f"K"
    )

else:

    predicted_market_text = (
        f"€"
        f"{predicted_market_value:,.0f}"
    )


# ============================================================
# MARKET VALUE RESULT
# ============================================================

print("\n" + "=" * 65)
print("              MARKET VALUE RESULT")
print("=" * 65)


show(
    "Player",
    row["name"]
)


show(
    "Predicted market value",
    predicted_market_text
)


# ============================================================
# ACTUAL VALUE COMPARISON
# ============================================================

if (
    pd.notna(
        display_market_value
    )
    and display_market_value > 0
):

    if display_market_value >= 1_000_000:

        actual_market_text = (
            f"€"
            f"{display_market_value / 1_000_000:.2f}"
            f"M"
        )

    elif display_market_value >= 1_000:

        actual_market_text = (
            f"€"
            f"{display_market_value / 1_000:.0f}"
            f"K"
        )

    else:

        actual_market_text = (
            f"€"
            f"{display_market_value:,.0f}"
        )


    show(
        "Actual market value",
        actual_market_text
    )


    difference = (
        predicted_market_value
        - display_market_value
    )


    percentage_difference = (
        difference
        / display_market_value
    ) * 100


    show(
        "Prediction difference",
        f"€{difference:,.0f}"
    )


    show(
        "Percentage difference",
        f"{percentage_difference:+.2f}%"
    )


# ============================================================
# MARKET VALUE INTERPRETATION
# ============================================================

print("\nINTERPRETATION")
print("--------------------------------")


if predicted_market_value >= 100_000_000:

    print(
        "Elite market-value range."
    )

elif predicted_market_value >= 50_000_000:

    print(
        "Very high market-value range."
    )

elif predicted_market_value >= 20_000_000:

    print(
        "High market-value range."
    )

elif predicted_market_value >= 5_000_000:

    print(
        "Moderate market-value range."
    )

else:

    print(
        "Lower market-value range."
    )


# ============================================================
# 2025-26 NOTE
# ============================================================

if selected_season == 2025:

    print("\nNOTE")
    print("--------------------------------")

    print(
        "2025-26 is the latest season "
        "available in the dataset."
    )

    print(
        "Transfer prediction uses the "
        "player's 2025-26 statistics."
    )

    print(
        "Market-value prediction is "
        "generated from player statistics "
        "and model features."
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 65)
print("                 ANALYSIS COMPLETE")
print("=" * 65)

print(
    "\n✓ No model training was performed."
)

print(
    "✓ Existing transfer model was used."
)

print(
    "✓ Existing market-value model was used."
)

print(
    "✓ Partial player search is enabled."
)

print(
    "✓ Multiple matching players are supported."
)

print(
    "✓ Age is calculated at the start "
    "of the football season."
)

print(
    "✓ Corrected display starts are preserved."
)

print(
    "✓ Corrected display market values "
    "are preserved."
)

print(
    "✓ Transfer prediction completed."
)

print(
    "✓ Market-value prediction completed."
)