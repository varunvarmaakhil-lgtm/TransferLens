import warnings
from pathlib import Path
from datetime import datetime
from difflib import get_close_matches

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="TransferLens",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# VISUAL POLISH
# Native Streamlit components only — no HTML div markup.
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 90% 0%, rgba(0, 255, 190, 0.07), transparent 30%),
            radial-gradient(circle at 0% 35%, rgba(80, 100, 255, 0.05), transparent 25%),
            #070a0f;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #111720, #0b1017);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 14px 14px;
        min-height: 92px;
    }

    [data-testid="stMetricLabel"] {
        color: #8993a3 !important;
        font-size: 0.78rem !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    [data-testid="stMetricValue"] {
        color: #f4f7fb !important;
        font-weight: 800 !important;
        font-size: 1.38rem !important;
        line-height: 1.15 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    [data-testid="stHorizontalBlock"]:has(.transferlens-model-marker) [data-testid="stMetricValue"] {
        font-size: 1.05rem !important;
        letter-spacing: -0.02em !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] > div {
        border-radius: 12px;
    }

    .stButton > button {
        min-height: 46px;
        border-radius: 12px;
        font-weight: 700;
    }

    hr {
        border-color: rgba(255, 255, 255, 0.08);
    }

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent


def find_file(filename):
    locations = [
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "data" / filename,
        PROJECT_ROOT / "models" / filename,
        APP_DIR / filename,
        APP_DIR / "data" / filename,
        APP_DIR / "models" / filename,
    ]
    for path in locations:
        if path.is_file():
            return path

    for root in [PROJECT_ROOT, APP_DIR]:
        try:
            matches = list(root.rglob(filename))
            if matches:
                return matches[0]
        except Exception:
            pass

    return None


DATA_PATH = find_file("transfer_training_updated.csv")
TRANSFER_MODEL_PATH = find_file("transfer_model.joblib")
TRANSFER_INFO_PATH = find_file("transfer_model_info.joblib")
MARKET_MODEL_PATH = find_file("market_value_model.joblib")
MARKET_INFO_PATH = find_file("market_value_model_info.joblib")


def safe_float(value):
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def safe_int(value):
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except Exception:
        return 0


def clean_text(value):
    if pd.isna(value):
        return "N/A"

    text = str(value).strip()

    if text.lower() in {"", "nan", "none", "null"}:
        return "N/A"

    return text


def safe_divide(a, b):
    if isinstance(b, pd.Series):
        denominator = b.replace(0, np.nan)

        return (
            a.divide(denominator)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    denominator = np.where(
        np.asarray(b) == 0,
        np.nan,
        b,
    )

    result = np.asarray(a) / denominator

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def season_to_year(value):
    if pd.isna(value):
        return np.nan

    try:
        return int(str(value).strip()[:4])
    except Exception:
        return np.nan


def season_label(year):
    try:
        year = int(year)
        return f"{year}-{str(year + 1)[-2:]}"
    except Exception:
        return str(year)


def calculate_age(dob, season_year):
    try:
        if dob is None or pd.isna(dob) or pd.isna(season_year):
            return np.nan

        dob = pd.to_datetime(dob, errors="coerce")
        if pd.isna(dob):
            return np.nan

        # Accept both 2025 and season labels such as 2025-26.
        season_text = str(season_year).strip()
        season_start_year = int(season_text.split("-")[0])

        # Football season age: age at the start of the season (1 August).
        season_start = datetime(season_start_year, 8, 1)
        age = season_start.year - dob.year

        if (season_start.month, season_start.day) < (dob.month, dob.day):
            age -= 1

        return int(age)
    except Exception:
        return np.nan


def format_money(value):
    try:
        value = float(value)
    except Exception:
        return "N/A"

    if not np.isfinite(value) or value <= 0:
        return "N/A"

    if value >= 1_000_000_000:
        return f"€{value / 1_000_000_000:.1f}B"

    if value >= 1_000_000:
        return f"€{value / 1_000_000:.1f}M"

    if value >= 1_000:
        return f"€{value / 1_000:.0f}K"

    return f"€{value:,.0f}"


def metric_text(value, percentage=False, decimals=4):
    if value is None:
        return "—"

    try:
        value = float(value)

        if percentage:
            return f"{value * 100:.2f}%"

        return f"{value:.{decimals}f}"

    except Exception:
        return "—"


def get_first(dictionary, keys, default=None):
    if not isinstance(dictionary, dict):
        return default

    for key in keys:
        if key in dictionary and dictionary[key] is not None:
            return dictionary[key]

    return default


@st.cache_data(show_spinner=False)
def load_dataset(path):
    return pd.read_csv(
        path,
        low_memory=False,
    )


@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_info(path):
    if path is None:
        return {}

    try:
        return joblib.load(path)
    except Exception:
        return {}


if DATA_PATH is None or TRANSFER_MODEL_PATH is None:
    st.error(
        "TransferLens could not find the required dataset/model files."
    )
    st.write(
        "Required: transfer_training_updated.csv and "
        "transfer_model.joblib"
    )
    st.stop()


try:
    raw_df = load_dataset(str(DATA_PATH))
    transfer_model = load_model(str(TRANSFER_MODEL_PATH))
    transfer_info = load_info(str(TRANSFER_INFO_PATH))

except Exception as exc:
    st.error("TransferLens could not load the project files.")
    st.exception(exc)
    st.stop()


market_model = None
market_info = {}

if MARKET_MODEL_PATH is not None:
    try:
        market_model = load_model(str(MARKET_MODEL_PATH))
        market_info = load_info(str(MARKET_INFO_PATH))
    except Exception:
        market_model = None
        market_info = {}


transfer_features = get_first(
    transfer_info,
    [
        "features",
        "feature_columns",
    ],
    [],
)

transfer_categorical = get_first(
    transfer_info,
    [
        "categorical_features",
        "categorical_columns",
        "cat_features",
    ],
    [],
)

if not transfer_features:
    try:
        transfer_features = list(
            transfer_model.feature_names_
        )
    except Exception:
        transfer_features = []


transfer_features = list(
    dict.fromkeys(
        [str(x) for x in transfer_features]
    )
)

transfer_categorical = list(
    dict.fromkeys(
        [str(x) for x in transfer_categorical]
    )
)


transfer_threshold = safe_float(
    get_first(
        transfer_info,
        [
            "threshold",
            "decision_threshold",
            "classification_threshold",
        ],
        0.51,
    )
)

accuracy = get_first(
    transfer_info,
    [
        "accuracy",
        "test_accuracy",
    ],
)

roc_auc = get_first(
    transfer_info,
    [
        "roc_auc",
        "roc_auc_score",
    ],
)

f1_value = get_first(
    transfer_info,
    [
        "f1_score",
        "f1",
    ],
)

if not transfer_features:
    st.error(
        "The saved transfer model does not expose "
        "its feature columns."
    )
    st.stop()


def build_model_features(data):
    df = data.copy()

    if "name" not in df.columns:
        if "player_name" in df.columns:
            df["name"] = df["player_name"]

        elif "Name" in df.columns:
            df["name"] = df["Name"]

        else:
            raise ValueError(
                "No player-name column was found."
            )

    df["name"] = (
        df["name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if "position" not in df.columns:
        if "Pos" in df.columns:
            df["position"] = df["Pos"]

        elif "Position" in df.columns:
            df["position"] = df["Position"]

        else:
            df["position"] = "Unknown"

    df["position"] = (
        df["position"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    if "country_of_birth" not in df.columns:
        if "nationality" in df.columns:
            df["country_of_birth"] = (
                df["nationality"]
            )

        else:
            df["country_of_birth"] = "Unknown"

    df["country_of_birth"] = (
        df["country_of_birth"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    if "season" not in df.columns:
        if "season_year" in df.columns:
            df["season"] = (
                df["season_year"].astype(str)
            )
        else:
            raise ValueError(
                "Dataset does not contain season information."
            )

    if "season_year" in df.columns:
        df["season_year_numeric"] = (
            df["season_year"].apply(season_to_year)
        )
    else:
        df["season_year_numeric"] = (
            df["season"].apply(season_to_year)
        )

    df["season_year_numeric"] = (
        pd.to_numeric(
            df["season_year_numeric"],
            errors="coerce",
        )
        .fillna(0)
    )

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

        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .fillna(0)
        )

    if "date_of_birth" in df.columns:
        df["date_of_birth"] = pd.to_datetime(
            df["date_of_birth"],
            errors="coerce",
        )
    else:
        df["date_of_birth"] = pd.NaT

    if "age" in df.columns:
        df["age_numeric"] = pd.to_numeric(
            df["age"],
            errors="coerce",
        )
    else:
        df["age_numeric"] = np.nan

    calculated_age = df.apply(
        lambda x: calculate_age(
            x["date_of_birth"],
            x["season_year_numeric"],
        ),
        axis=1,
    )

    df["age_numeric"] = (
        df["age_numeric"]
        .fillna(calculated_age)
        .clip(
            lower=15,
            upper=45,
        )
    )

    df["age_squared"] = (
        df["age_numeric"] ** 2
    )

    df["age_peak_distance"] = (
        df["age_numeric"] - 25
    ).abs()

    df["young_player"] = (
        df["age_numeric"] <= 23
    ).astype(int)

    df["prime_age"] = (
        (df["age_numeric"] >= 24)
        &
        (df["age_numeric"] <= 29)
    ).astype(int)

    df["older_player"] = (
        df["age_numeric"] >= 30
    ).astype(int)

    df["goals_per_appearance"] = safe_divide(
        df["goals"],
        df["appearances"],
    )

    df["assists_per_appearance"] = safe_divide(
        df["assists"],
        df["appearances"],
    )

    df["goals_per_90"] = safe_divide(
        df["goals"] * 90,
        df["minutes_played"],
    )

    df["assists_per_90"] = safe_divide(
        df["assists"] * 90,
        df["minutes_played"],
    )

    df["minutes_per_appearance"] = safe_divide(
        df["minutes_played"],
        df["appearances"],
    )

    df["starts_ratio"] = safe_divide(
        df["starts"],
        df["appearances"],
    )

    df["goal_contributions"] = (
        df["goals"] + df["assists"]
    )

    df["goal_contributions_per_appearance"] = (
        safe_divide(
            df["goal_contributions"],
            df["appearances"],
        )
    )

    df["goal_contributions_per_90"] = (
        safe_divide(
            df["goal_contributions"] * 90,
            df["minutes_played"],
        )
    )

    df["yellow_cards_per_appearance"] = safe_divide(
        df["yellow_cards"],
        df["appearances"],
    )

    df["red_cards_per_appearance"] = safe_divide(
        df["red_cards"],
        df["appearances"],
    )

    df["total_cards"] = (
        df["yellow_cards"] + df["red_cards"]
    )

    df["cards_per_appearance"] = safe_divide(
        df["total_cards"],
        df["appearances"],
    )

    df["market_value_in_eur"] = (
        df["market_value_in_eur"].clip(lower=0)
    )

    df["market_value_log"] = np.log1p(
        df["market_value_in_eur"]
    )

    player_group = (
        "player_id"
        if "player_id" in df.columns
        else "name"
    )

    df = (
        df.sort_values(
            [
                player_group,
                "season_year_numeric",
            ]
        )
        .reset_index(drop=True)
    )

    grouped = df.groupby(
        player_group,
        sort=False,
    )

    history_columns = [
        "appearances",
        "minutes_played",
        "goals",
        "assists",
        "yellow_cards",
        "red_cards",
        "market_value_in_eur",
    ]

    for column in history_columns:
        df[
            f"previous_{column}"
        ] = grouped[column].shift(1)

        df[
            f"two_season_avg_{column}"
        ] = (
            df.groupby(player_group)[column]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(
                    2,
                    min_periods=1,
                )
                .mean()
            )
        )

        df[
            f"three_season_avg_{column}"
        ] = (
            df.groupby(player_group)[column]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(
                    3,
                    min_periods=1,
                )
                .mean()
            )
        )

    trend_columns = [
        "goals",
        "assists",
        "minutes_played",
        "appearances",
        "market_value_in_eur",
    ]

    for column in trend_columns:
        previous = (
            df.groupby(player_group)[column]
            .shift(1)
        )

        previous_two = (
            df.groupby(player_group)[column]
            .shift(2)
        )

        df[
            f"{column}_change"
        ] = (
            df[column] - previous
        )

        df[
            f"{column}_trend"
        ] = (
            previous - previous_two
        )

    if "transferred_next_season" in df.columns:
        transfer_series = (
            df.groupby(
                player_group
            )["transferred_next_season"]
        )

        df["previous_transfer"] = (
            transfer_series.shift(1)
        )

        df["previous_transfer_rate"] = (
            transfer_series.transform(
                lambda x:
                x.shift(1)
                .expanding()
                .mean()
            )
        )

        df["previous_transfer_count"] = (
            transfer_series.transform(
                lambda x:
                x.shift(1)
                .expanding()
                .sum()
            )
        )

        df["previous_seasons_count"] = (
            transfer_series.transform(
                lambda x:
                x.shift(1)
                .expanding()
                .count()
            )
        )

    else:
        df["previous_transfer"] = 0
        df["previous_transfer_rate"] = 0
        df["previous_transfer_count"] = 0
        df["previous_seasons_count"] = 0

    df["career_seasons_before"] = (
        df.groupby(
            player_group
        ).cumcount()
    )

    season_median = (
        df.groupby(
            "season_year_numeric"
        )["market_value_in_eur"]
        .transform("median")
    )

    df["market_value_vs_season_median"] = (
        safe_divide(
            df["market_value_in_eur"],
            season_median,
        )
    )

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

    df["minutes_value_efficiency"] = safe_divide(
        df["market_value_in_eur"],
        df["minutes_played"],
    )

    df["goal_assist_ratio"] = safe_divide(
        df["goals"],
        df["assists"] + 1,
    )

    df["minutes_per_appearance_squared"] = (
        df["minutes_per_appearance"] ** 2
    )

    df["display_starts"] = df["starts"]

    df["display_market_value"] = (
        pd.to_numeric(
            df["market_value_in_eur"],
            errors="coerce",
        )
        .where(
            df["market_value_in_eur"] > 0
        )
    )

    return df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


try:
    df = build_model_features(raw_df)
except Exception as exc:
    st.error("Feature preparation failed.")
    st.exception(exc)
    st.stop()


def build_model_input(
    row,
    features,
    categoricals,
    overrides=None,
):
    overrides = overrides or {}

    output = pd.DataFrame(
        index=[0]
    )

    for feature in features:
        if feature in overrides:
            value = overrides[feature]

        elif feature in row.index:
            value = row[feature]

        else:
            value = np.nan

        output[feature] = [value]

    for column in categoricals:
        if column in output.columns:
            output[column] = (
                output[column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )

    for column in features:
        if column not in categoricals:
            output[column] = (
                pd.to_numeric(
                    output[column],
                    errors="coerce",
                )
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .fillna(0)
            )

    return output[features]


def predict_transfer(row):
    age_value = calculate_age(
        row.get("date_of_birth"),
        row.get("season_year_numeric"),
    )

    overrides = {}

    if pd.notna(age_value):
        if "age_numeric" in transfer_features:
            overrides["age_numeric"] = float(
                age_value
            )

        if "age" in transfer_features:
            overrides["age"] = float(
                age_value
            )

    X = build_model_input(
        row,
        transfer_features,
        transfer_categorical,
        overrides,
    )

    probability = float(
        transfer_model.predict_proba(
            X
        )[0][1]
    )

    probability = float(
        np.clip(
            probability,
            0,
            1,
        )
    )

    return probability, X


def likelihood_band(probability):
    percentage = probability * 100

    if percentage >= 75:
        return "Very high likelihood"

    if percentage >= 55:
        return "High likelihood"

    if percentage >= 40:
        return "Moderate likelihood"

    if percentage >= 25:
        return "Low likelihood"

    return "Very low likelihood"


market_features = []
market_categorical = []

if market_model is not None:
    market_features = get_first(
        market_info,
        [
            "features",
            "feature_columns",
        ],
        [],
    )

    market_categorical = get_first(
        market_info,
        [
            "categorical_features",
            "categorical_columns",
        ],
        [],
    )

    if not market_features:
        try:
            market_features = list(
                market_model.feature_names_
            )
        except Exception:
            market_features = []


if market_model is not None and not market_features:
    market_features = [
        "position",
        "season",
        "season_year_numeric",
        "age_numeric",
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

    market_categorical = [
        "position",
        "season",
    ]


market_features = list(
    dict.fromkeys(
        [str(x) for x in market_features]
    )
)

market_categorical = list(
    dict.fromkeys(
        [str(x) for x in market_categorical]
    )
)


def market_target_mode():
    value = get_first(
        market_info,
        [
            "target_transform",
            "prediction_target",
            "target",
            "target_column",
            "output_type",
        ],
        None,
    )

    if value is None:
        return "log_ratio"

    text = str(value).lower()

    if (
        "log" in text
        and "ratio" in text
    ):
        return "log_ratio"

    if "ratio" in text:
        return "ratio"

    if (
        "market_value" in text
        or "value" in text
    ):
        return "direct"

    return "log_ratio"


def predict_market_value(
    row,
    selected_season,
):
    if market_model is None:
        return None

    current_value = safe_float(
        row.get(
            "market_value_in_eur"
        )
    )

    if current_value <= 0:
        return None

    if "player_id" in df.columns:
        player_key = str(
            row.get(
                "player_id",
                "",
            )
        ).strip()

        history = df[
            df["player_id"]
            .fillna("")
            .astype(str)
            .str.strip()
            == player_key
        ].copy()

    else:
        player_key = str(
            row.get(
                "name",
                "",
            )
        ).strip()

        history = df[
            df["name"]
            .astype(str)
            .str.strip()
            == player_key
        ].copy()

    history = history.sort_values(
        "season_year_numeric"
    )

    previous_rows = history[
        history["season_year_numeric"]
        < selected_season
    ]

    previous_values = (
        previous_rows[
            previous_rows[
                "market_value_in_eur"
            ] > 0
        ]["market_value_in_eur"]
        .tolist()
    )

    previous_value = (
        float(previous_values[-1])
        if previous_values
        else 0.0
    )

    next_year = (
        int(selected_season) + 1
    )

    next_season = season_label(
        next_year
    )

    current_age = calculate_age(
        row.get("date_of_birth"),
        selected_season,
    )

    if pd.isna(current_age):
        current_age = safe_float(
            row.get("age_numeric")
        )

    value_change = (
        current_value - previous_value
        if previous_value > 0
        else 0.0
    )

    value_growth = (
        value_change / previous_value
        if previous_value > 0
        else 0.0
    )

    career_next = len(
        history[
            history["season_year_numeric"]
            <= selected_season
        ]
    )

    overrides = {
        "season": next_season,
        "season_year_numeric": next_year,
        "age_numeric": (
            float(current_age) + 1
        ),
        "previous_market_value": current_value,
        "previous_market_value_change": value_change,
        "previous_market_value_growth": value_growth,
        "career_seasons_before": career_next,
    }

    X = build_model_input(
        row,
        market_features,
        market_categorical,
        overrides,
    )

    raw_prediction = float(
        market_model.predict(X)[0]
    )

    mode = market_target_mode()

    if mode == "log_ratio":
        ratio = float(
            np.exp(raw_prediction)
        )

    elif mode == "ratio":
        ratio = raw_prediction

    else:
        predicted_value = max(
            raw_prediction,
            0.0,
        )

        return {
            "predicted_value": predicted_value,
            "current_value": current_value,
            "previous_value": previous_value,
            "next_season": next_season,
            "ratio": (
                predicted_value
                / current_value
                if current_value > 0
                else np.nan
            ),
        }

    ratio = float(
        np.clip(
            ratio,
            0.55,
            1.45,
        )
    )

    predicted_value = (
        current_value * ratio
    )

    return {
        "predicted_value": float(
            predicted_value
        ),
        "current_value": current_value,
        "previous_value": previous_value,
        "next_season": next_season,
        "ratio": ratio,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("⚽ TransferLens")
    st.caption(
        "Football intelligence dashboard"
    )

    st.divider()

    st.subheader("System Status")

    st.success("Dataset connected")
    st.success("Transfer model loaded")

    if market_model is not None:
        st.success("Market model loaded")
    else:
        st.warning(
            "Market model unavailable"
        )

    st.divider()

    st.subheader("Dataset")

    st.write(
        f"**{len(df):,}** player-season rows"
    )

    st.write(
        f"**{df['name'].nunique():,}** "
        "unique players"
    )

    valid_years = df.loc[
        df["season_year_numeric"] > 0,
        "season_year_numeric",
    ]

    if not valid_years.empty:
        st.write(
            f"{season_label(valid_years.min())} "
            f"→ "
            f"{season_label(valid_years.max())}"
        )

    st.divider()

    st.subheader("Transfer Model")

    st.write(
        f"Algorithm: "
        f"**{type(transfer_model).__name__}**"
    )

    st.write(
        f"Features: **{len(transfer_features)}**"
    )

    st.write(
        f"Threshold: "
        f"**{transfer_threshold:.3f}**"
    )

    if st.button(
        "Clear analysis",
        width="stretch",
    ):
        for key in [
            "selected_player",
            "selected_season",
            "analysis_key",
            "transfer_result",
            "market_result",
        ]:
            st.session_state.pop(
                key,
                None,
            )

        st.rerun()


# ============================================================
# HERO
# ============================================================

st.title("⚽ TransferLens")

st.subheader(
    "AI-powered football transfer intelligence"
)

st.caption(
    "Explore player performance, inspect every available season, "
    "and estimate transfer likelihood and next-season market value."
)

hero_cols = st.columns(4)

hero_items = [
    (
        "Player-season rows",
        f"{len(df):,}",
    ),
    (
        "Unique players",
        f"{df['name'].nunique():,}",
    ),
    (
        "Transfer features",
        str(len(transfer_features)),
    ),
    (
        "Dataset coverage",
        (
            f"{season_label(valid_years.min())} "
            f"→ "
            f"{season_label(valid_years.max())}"
        )
        if not valid_years.empty
        else "N/A",
    ),
]

for col, (label, value) in zip(
    hero_cols,
    hero_items,
):
    with col:
        st.metric(
            label,
            value,
        )


st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("Model Performance")

st.caption(
    "Performance metrics reported by the saved TransferLens model."
)

metrics = st.columns(5)

with metrics[0]:
    st.markdown(
        '<span class="transferlens-model-marker"></span>',
        unsafe_allow_html=True,
    )
    st.metric(
        "Model",
        type(transfer_model).__name__,
    )

with metrics[1]:
    st.metric(
        "Features",
        len(transfer_features),
    )

with metrics[2]:
    st.metric(
        "Test Accuracy",
        metric_text(
            accuracy,
            percentage=True,
        ),
    )

with metrics[3]:
    st.metric(
        "ROC-AUC",
        metric_text(
            roc_auc
        ),
    )

with metrics[4]:
    st.metric(
        "F1 Score",
        metric_text(
            f1_value
        ),
    )


st.divider()


# ============================================================
# PLAYER SEARCH
# ============================================================

st.header("🔎 Player Analysis")

st.caption(
    "Search the real dataset, select a player, "
    "then choose any season actually available "
    "for that player."
)

search = st.text_input(
    "Search player",
    placeholder="Type a player name...",
)

search = search.strip()

if not search:
    st.info(
        "Start by typing a player name. "
        "Player names, seasons and statistics are "
        "loaded dynamically from the dataset."
    )
    st.stop()


name_series = (
    df["name"]
    .fillna("")
    .astype(str)
    .str.strip()
)

matches = (
    name_series[
        name_series.str.contains(
            search,
            case=False,
            na=False,
            regex=False,
        )
    ]
    .drop_duplicates()
    .sort_values()
    .tolist()
)


if not matches:
    all_names = (
        name_series[
            name_series != ""
        ]
        .drop_duplicates()
        .tolist()
    )

    suggestions = get_close_matches(
        search,
        all_names,
        n=8,
        cutoff=0.35,
    )

    st.warning(
        f"No player found for '{search}'."
    )

    if suggestions:
        st.write(
            "Possible matches: "
            + ", ".join(suggestions)
        )

    st.stop()


selected_name = st.selectbox(
    "Player",
    matches,
    key="selected_player",
)

player_matches = df[
    df["name"] == selected_name
].copy()


season_values = (
    player_matches[
        "season_year_numeric"
    ]
    .dropna()
    .astype(int)
    .drop_duplicates()
    .sort_values(
        ascending=False
    )
    .tolist()
)


if not season_values:
    st.warning(
        "No seasons are available for this player."
    )
    st.stop()


season_options = {
    season_label(year): year
    for year in season_values
}


selected_season_label = st.selectbox(
    "Season",
    list(
        season_options.keys()
    ),
    key="selected_season",
)

selected_season = season_options[
    selected_season_label
]


# Reset predictions whenever player/season changes.
current_analysis_key = (
    f"{selected_name}::{selected_season}"
)

if (
    st.session_state.get("analysis_key")
    != current_analysis_key
):
    st.session_state["analysis_key"] = (
        current_analysis_key
    )
    st.session_state.pop(
        "transfer_result",
        None,
    )
    st.session_state.pop(
        "market_result",
        None,
    )


selected_rows = player_matches[
    player_matches[
        "season_year_numeric"
    ]
    == selected_season
].copy()


if selected_rows.empty:
    st.warning(
        "No data is available for the selected season."
    )
    st.stop()


if len(selected_rows) > 1:
    selected_rows = selected_rows.sort_values(
        "minutes_played",
        ascending=False,
    )


row = selected_rows.iloc[0].copy()


# ============================================================
# PLAYER OVERVIEW
# ============================================================

st.header(
    f"👤 {selected_name}"
)

age_value = calculate_age(
    row.get("date_of_birth"),
    selected_season,
)

# If DOB is unavailable, use the engineered age already present in
# the model feature layer. This prevents a valid age from showing N/A.
if pd.isna(age_value):
    fallback_age = row.get("age_numeric", np.nan)
    if pd.notna(fallback_age):
        age_value = float(fallback_age)

age_display = (
    "N/A"
    if pd.isna(age_value)
    else str(int(float(age_value)))
)


club_column = next(
    (
        candidate
        for candidate in [
            "club",
            "current_club",
            "club_name",
            "team",
        ]
        if candidate in row.index
    ),
    None,
)


overview = st.columns(5)

with overview[0]:
    st.metric(
        "Position",
        clean_text(
            row.get("position")
        ),
    )

with overview[1]:
    st.metric(
        "Age",
        age_display,
    )

with overview[2]:
    st.metric(
        "Season",
        selected_season_label,
    )

with overview[3]:
    st.metric(
        "Nation",
        clean_text(
            row.get(
                "country_of_birth"
            )
        ),
    )

with overview[4]:
    st.metric(
        "Club",
        (
            clean_text(
                row.get(
                    club_column
                )
            )
            if club_column
            else "N/A"
        ),
    )


# ============================================================
# PERFORMANCE PROFILE
# ============================================================

st.header(
    "📊 Performance Profile"
)

st.caption(
    "All values below are calculated from "
    "the selected player-season record."
)


performance = st.columns(5)

performance_values = [
    (
        "Appearances",
        f"{safe_int(row.get('appearances')):,}",
    ),
    (
        "Starts",
        f"{safe_int(row.get('starts')):,}",
    ),
    (
        "Minutes",
        f"{safe_int(row.get('minutes_played')):,}",
    ),
    (
        "Goals",
        f"{safe_int(row.get('goals')):,}",
    ),
    (
        "Assists",
        f"{safe_int(row.get('assists')):,}",
    ),
]


for col, (label, value) in zip(
    performance,
    performance_values,
):
    with col:
        st.metric(
            label,
            value,
        )


advanced = st.columns(5)

advanced_values = [
    (
        "Goals / 90",
        f"{safe_float(row.get('goals_per_90')):.2f}",
    ),
    (
        "Assists / 90",
        f"{safe_float(row.get('assists_per_90')):.2f}",
    ),
    (
        "Contributions",
        f"{safe_int(row.get('goal_contributions')):,}",
    ),
    (
        "Contributions / 90",
        f"{safe_float(row.get('goal_contributions_per_90')):.2f}",
    ),
    (
        "Starts Ratio",
        f"{safe_float(row.get('starts_ratio')):.2f}",
    ),
]


for col, (label, value) in zip(
    advanced,
    advanced_values,
):
    with col:
        st.metric(
            label,
            value,
        )


extra = st.columns(4)

extra_values = [
    (
        "Goals / App",
        f"{safe_float(row.get('goals_per_appearance')):.2f}",
    ),
    (
        "Assists / App",
        f"{safe_float(row.get('assists_per_appearance')):.2f}",
    ),
    (
        "Minutes / App",
        f"{safe_float(row.get('minutes_per_appearance')):.2f}",
    ),
    (
        "Cards / App",
        f"{safe_float(row.get('cards_per_appearance')):.2f}",
    ),
]


for col, (label, value) in zip(
    extra,
    extra_values,
):
    with col:
        st.metric(
            label,
            value,
        )


st.divider()


# ============================================================
# YEAR-BY-YEAR CALCULATED STATISTICS
# ============================================================

st.header(
    "📅 Year-by-Year Statistics"
)

st.caption(
    "Every available season for this player, using "
    "the calculated statistics from the TransferLens "
    "analysis layer."
)


year_stats = []

for season_year in sorted(
    season_values,
    reverse=True,
):
    season_rows = player_matches[
        player_matches[
            "season_year_numeric"
        ]
        == season_year
    ].copy()

    if season_rows.empty:
        continue

    if len(season_rows) > 1:
        season_rows = season_rows.sort_values(
            "minutes_played",
            ascending=False,
        )

    season_row = season_rows.iloc[0]

    year_stats.append(
        {
            "Season": season_label(
                season_year
            ),
            "Goals / 90": round(
                safe_float(
                    season_row.get(
                        "goals_per_90"
                    )
                ),
                3,
            ),
            "Assists / 90": round(
                safe_float(
                    season_row.get(
                        "assists_per_90"
                    )
                ),
                3,
            ),
            "Goals / App": round(
                safe_float(
                    season_row.get(
                        "goals_per_appearance"
                    )
                ),
                3,
            ),
            "Assists / App": round(
                safe_float(
                    season_row.get(
                        "assists_per_appearance"
                    )
                ),
                3,
            ),
            "Contributions / 90": round(
                safe_float(
                    season_row.get(
                        "goal_contributions_per_90"
                    )
                ),
                3,
            ),
            "Starts Ratio": round(
                safe_float(
                    season_row.get(
                        "starts_ratio"
                    )
                ),
                3,
            ),
            "Minutes / App": round(
                safe_float(
                    season_row.get(
                        "minutes_per_appearance"
                    )
                ),
                2,
            ),
            "Cards / App": round(
                safe_float(
                    season_row.get(
                        "cards_per_appearance"
                    )
                ),
                3,
            ),
        }
    )


if year_stats:
    st.dataframe(
        pd.DataFrame(
            year_stats
        ),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# CAREER HISTORY
# ============================================================

st.header(
    "📈 Career History"
)

st.caption(
    "Historical player-season records available "
    "in the dataset."
)


history_rows = []

for season_year in sorted(
    season_values,
    reverse=True,
):
    season_rows = player_matches[
        player_matches[
            "season_year_numeric"
        ]
        == season_year
    ].copy()

    if season_rows.empty:
        continue

    if len(season_rows) > 1:
        season_rows = season_rows.sort_values(
            "minutes_played",
            ascending=False,
        )

    historical_row = season_rows.iloc[0]

    history_rows.append(
        {
            "Season": season_label(
                season_year
            ),
            "Club": (
                clean_text(
                    historical_row.get(
                        club_column
                    )
                )
                if club_column
                else "N/A"
            ),
            "Apps": safe_int(
                historical_row.get(
                    "appearances"
                )
            ),
            "Starts": safe_int(
                historical_row.get(
                    "starts"
                )
            ),
            "Minutes": safe_int(
                historical_row.get(
                    "minutes_played"
                )
            ),
            "Goals": safe_int(
                historical_row.get(
                    "goals"
                )
            ),
            "Assists": safe_int(
                historical_row.get(
                    "assists"
                )
            ),
            "Market Value": format_money(
                historical_row.get(
                    "market_value_in_eur"
                )
            ),
        }
    )


if history_rows:
    history_df = pd.DataFrame(
        history_rows
    )

    st.dataframe(
        history_df,
        width="stretch",
        hide_index=True,
    )

    chart_df = (
        history_df
        .copy()
        .sort_values("Season")
        .set_index("Season")
    )

    chart_tab1, chart_tab2 = st.tabs(
        [
            "⚽ Goals & Assists",
            "💰 Market Value",
        ]
    )

    with chart_tab1:
        st.line_chart(
            chart_df[
                [
                    "Goals",
                    "Assists",
                ]
            ],
            width="stretch",
        )

    with chart_tab2:
        market_chart = player_matches[
            player_matches[
                "market_value_in_eur"
            ] > 0
        ][
            [
                "season_year_numeric",
                "market_value_in_eur",
            ]
        ].copy()

        market_chart["Season"] = (
            market_chart[
                "season_year_numeric"
            ].apply(season_label)
        )

        market_chart = (
            market_chart[
                [
                    "Season",
                    "market_value_in_eur",
                ]
            ]
            .drop_duplicates("Season")
            .sort_values("Season")
            .set_index("Season")
        )

        market_chart.columns = [
            "Market Value (€)"
        ]

        if not market_chart.empty:
            st.line_chart(
                market_chart,
                width="stretch",
            )


# ============================================================
# COMPLETE DATASET FIELDS
# ============================================================

st.header(
    "📋 Complete Dataset Fields"
)

st.caption(
    "The full original dataset record(s) for this player. "
    "Nothing is hardcoded."
)


raw_name_column = (
    "name"
    if "name" in raw_df.columns
    else (
        "player_name"
        if "player_name" in raw_df.columns
        else None
    )
)


raw_player = raw_df.copy()

if raw_name_column is not None:
    raw_player = raw_player[
        raw_player[
            raw_name_column
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        == selected_name
    ].copy()


if "season_year" in raw_player.columns:
    raw_player = raw_player.sort_values(
        "season_year",
        ascending=False,
    )


display_raw = raw_player.copy()

# Fix Streamlit/PyArrow serialization for the raw dataset display only.
# The source dataset can contain mixed Python/NumPy types in a column named
# "Value". Keep the underlying dataset unchanged and make only the displayed
# copy consistent.
if "Value" in display_raw.columns:
    display_raw["Value"] = display_raw["Value"].map(
        lambda value: "" if pd.isna(value) else str(value)
    )

for column in display_raw.columns:

    if column == "market_value_in_eur":
        display_raw[column] = (
            display_raw[column]
            .apply(format_money)
        )

    elif column == "date_of_birth":
        display_raw[column] = (
            pd.to_datetime(
                display_raw[column],
                errors="coerce",
            )
            .dt.strftime("%Y-%m-%d")
            .fillna("N/A")
        )


st.dataframe(
    display_raw,
    width="stretch",
    hide_index=True,
)


# ============================================================
# TRANSFER MODEL FEATURE INPUTS
# ============================================================

with st.expander(
    "🧠 View transfer-model feature inputs"
):
    feature_preview = pd.DataFrame(
        {
            "Feature": transfer_features,
            "Value": [
                row.get(
                    feature,
                    0,
                )
                for feature in transfer_features
            ],
        }
    )

    for index, feature in enumerate(
        feature_preview["Feature"]
    ):
        value = feature_preview.at[
            index,
            "Value",
        ]

        if feature in transfer_categorical:
            feature_preview.at[
                index,
                "Value",
            ] = clean_text(value)

        elif (
            "market_value"
            in feature.lower()
        ):
            feature_preview.at[
                index,
                "Value",
            ] = format_money(value)

        elif isinstance(
            value,
            (
                float,
                np.floating,
            ),
        ):
            feature_preview.at[
                index,
                "Value",
            ] = round(
                float(value),
                4,
            )

    # Streamlit/PyArrow requires a consistent type in the Value column.
    # The model inputs can contain numpy scalars as well as formatted text,
    # so convert only this display column to strings before rendering.
    feature_preview["Value"] = feature_preview["Value"].apply(
        lambda value: clean_text(value)
        if isinstance(value, str)
        else str(value)
    )

    st.dataframe(
        feature_preview,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# AI FORECAST
# ============================================================

st.divider()

st.header(
    "🤖 AI Forecast"
)

st.caption(
    "Run the saved machine-learning models against "
    "the selected player-season."
)


predict_col1, predict_col2 = st.columns(2)


with predict_col1:
    run_transfer = st.button(
        "🎯 Predict Transfer Likelihood",
        type="primary",
        width="stretch",
    )


with predict_col2:
    run_market = st.button(
        "💰 Predict Next-Season Market Value",
        width="stretch",
        disabled=(
            market_model is None
        ),
    )


if run_transfer:
    try:
        probability, X_transfer = (
            predict_transfer(row)
        )

        st.session_state[
            "transfer_result"
        ] = {
            "probability": probability,
            "input": X_transfer,
        }

    except Exception as exc:
        st.error(
            "Transfer prediction failed."
        )
        st.exception(exc)


if run_market:
    try:
        result = predict_market_value(
            row,
            selected_season,
        )

        st.session_state[
            "market_result"
        ] = result

    except Exception as exc:
        st.error(
            "Market-value prediction failed."
        )
        st.exception(exc)


transfer_result = st.session_state.get(
    "transfer_result"
)

market_result = st.session_state.get(
    "market_result"
)


# ============================================================
# TRANSFER RESULT
# ============================================================

if transfer_result is not None:

    probability = (
        transfer_result[
            "probability"
        ]
    )

    percentage = (
        probability * 100
    )

    likely = (
        probability
        >= transfer_threshold
    )

    band = likelihood_band(
        probability
    )

    if likely:
        st.success(
            "LIKELY TO BE TRANSFERRED · "
            f"{percentage:.1f}%"
        )
    else:
        st.info(
            "NOT LIKELY TO BE TRANSFERRED · "
            f"{percentage:.1f}%"
        )

    result_cols = st.columns(3)

    with result_cols[0]:
        st.metric(
            "Transfer Probability",
            f"{percentage:.1f}%",
        )

    with result_cols[1]:
        st.metric(
            "Likelihood",
            band,
        )

    with result_cols[2]:
        st.metric(
            "Decision Threshold",
            f"{transfer_threshold:.3f}",
        )

    st.progress(
        probability,
        text=(
            f"{band} · "
            f"threshold {transfer_threshold:.3f}"
        ),
    )

    st.caption(
        "This is a machine-learning probability, "
        "not confirmation that an actual transfer "
        "will occur."
    )


# ============================================================
# MARKET VALUE RESULT
# ============================================================

if market_result is not None:

    predicted_value = safe_float(
        market_result.get(
            "predicted_value"
        )
    )

    current_value = safe_float(
        market_result.get(
            "current_value"
        )
    )

    previous_value = safe_float(
        market_result.get(
            "previous_value"
        )
    )

    next_season = market_result.get(
        "next_season",
        "N/A",
    )

    change_percentage = (
        (
            predicted_value
            - current_value
        )
        / current_value
        * 100
        if current_value > 0
        else 0
    )

    st.subheader(
        "💎 Market Value Projection"
    )

    market_cols = st.columns(4)

    with market_cols[0]:
        st.metric(
            "Predicted Next Season",
            format_money(
                predicted_value
            ),
        )

    with market_cols[1]:
        st.metric(
            "Current Value",
            format_money(
                current_value
            ),
        )

    with market_cols[2]:
        st.metric(
            "Previous Value",
            (
                format_money(
                    previous_value
                )
                if previous_value > 0
                else "N/A"
            ),
        )

    with market_cols[3]:
        st.metric(
            "Projected Season",
            next_season,
        )

    if change_percentage >= 0:
        st.success(
            f"Estimated change: "
            f"+{change_percentage:.1f}%"
        )
    else:
        st.warning(
            f"Estimated change: "
            f"{change_percentage:.1f}%"
        )

    if predicted_value >= 150_000_000:
        st.info(
            "Very high / elite market-value range."
        )

    elif predicted_value >= 100_000_000:
        st.info(
            "Elite market-value range."
        )

    elif predicted_value >= 50_000_000:
        st.info(
            "High market-value range."
        )

    elif predicted_value >= 20_000_000:
        st.info(
            "Strong market-value range."
        )

    else:
        st.info(
            "Lower market-value range."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TransferLens · Dataset-driven football intelligence · "
    "All player data and predictions are generated from "
    "the loaded project files."
)