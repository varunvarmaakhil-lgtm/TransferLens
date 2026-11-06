
import warnings
from pathlib import Path
from difflib import get_close_matches
from html import escape
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TransferLens",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# UI THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #070b12;
        --panel: #0d131d;
        --panel2: #111a26;
        --border: rgba(255,255,255,.08);
        --text: #f5f7fb;
        --muted: #8b98aa;
        --accent: #61e7c7;
        --accent2: #7aa7ff;
        --warning: #f6c85f;
        --danger: #ff7b8b;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 0%, rgba(97,231,199,.10), transparent 25%),
            radial-gradient(circle at 0% 20%, rgba(122,167,255,.09), transparent 28%),
            var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.4rem;
        padding-bottom: 4rem;
    }

    #MainMenu, footer { visibility: hidden; }
    header { background: transparent !important; }

    [data-testid="stSidebar"] {
        background: #09101a;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 22px;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        display: grid;
        place-items: center;
        border-radius: 13px;
        background: linear-gradient(135deg, #61e7c7, #7aa7ff);
        color: #071016;
        font-size: 21px;
        font-weight: 900;
        box-shadow: 0 10px 30px rgba(97,231,199,.18);
    }

    .brand-name {
        font-size: 19px;
        font-weight: 900;
        letter-spacing: -.5px;
    }

    .brand-caption {
        color: var(--muted);
        font-size: 11px;
        margin-top: 2px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 38px 42px;
        border-radius: 26px;
        border: 1px solid var(--border);
        background: linear-gradient(
            135deg,
            rgba(18,28,41,.98),
            rgba(9,14,22,.98)
        );
        box-shadow: 0 25px 70px rgba(0,0,0,.25);
        margin-bottom: 22px;
    }

    .hero:before {
        content: "";
        position: absolute;
        width: 430px;
        height: 430px;
        right: -190px;
        top: -260px;
        border-radius: 50%;
        background: rgba(97,231,199,.10);
        filter: blur(12px);
    }

    .hero-kicker {
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: clamp(38px, 5vw, 66px);
        line-height: .98;
        font-weight: 950;
        letter-spacing: -3px;
    }

    .hero-subtitle {
        color: #c2cbd7;
        font-size: 18px;
        margin-top: 12px;
        font-weight: 650;
    }

    .hero-copy {
        max-width: 780px;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.75;
        margin-top: 13px;
    }

    .hero-badges {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 20px;
    }

    .hero-badge {
        border: 1px solid rgba(255,255,255,.09);
        background: rgba(255,255,255,.035);
        color: #b8c3d1;
        padding: 7px 11px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 750;
    }

    .section-head {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 18px;
        margin-top: 34px;
        margin-bottom: 14px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 900;
        letter-spacing: -.7px;
        margin: 0;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 13px;
        margin-top: 4px;
        line-height: 1.55;
    }

    .metric-card,
    .stat-card {
        border: 1px solid var(--border);
        background: linear-gradient(145deg, #111a26, #0c121b);
        box-shadow: 0 12px 32px rgba(0,0,0,.15);
    }

    .metric-card {
        min-height: 112px;
        padding: 20px;
        border-radius: 17px;
    }

    .metric-label,
    .stat-label {
        color: #7f8c9e;
        font-size: 10px;
        font-weight: 850;
        letter-spacing: 1.35px;
        text-transform: uppercase;
    }

    .metric-value {
        margin-top: 9px;
        font-size: 27px;
        font-weight: 900;
        letter-spacing: -.5px;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: #657285;
        font-size: 11px;
        margin-top: 5px;
    }

    .player-card {
        padding: 25px 28px;
        border-radius: 22px;
        border: 1px solid rgba(97,231,199,.13);
        background: linear-gradient(
            135deg,
            rgba(19,30,43,.98),
            rgba(10,16,24,.98)
        );
        box-shadow: 0 18px 45px rgba(0,0,0,.18);
        margin: 16px 0;
    }

    .player-kicker {
        color: var(--accent);
        font-size: 10px;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        font-weight: 900;
    }

    .player-name {
        font-size: 34px;
        font-weight: 950;
        letter-spacing: -1.2px;
        margin-top: 5px;
    }

    .player-meta {
        color: #8997aa;
        font-size: 13px;
        margin-top: 7px;
    }

    .stat-card {
        min-height: 108px;
        padding: 19px;
        border-radius: 16px;
    }

    .stat-value {
        font-size: 25px;
        font-weight: 900;
        margin-top: 8px;
    }

    .result-card {
        padding: 31px;
        border-radius: 23px;
        border: 1px solid rgba(97,231,199,.13);
        background: linear-gradient(145deg, #111c27, #0b1119);
        text-align: center;
        box-shadow: 0 20px 55px rgba(0,0,0,.22);
    }

    .result-badge {
        display: inline-block;
        border: 1px solid rgba(255,255,255,.09);
        background: rgba(255,255,255,.04);
        padding: 6px 11px;
        border-radius: 999px;
        color: #aab6c5;
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 1.2px;
    }

    .result-title {
        font-size: 25px;
        font-weight: 950;
        margin-top: 13px;
    }

    .probability {
        font-size: 64px;
        line-height: 1;
        font-weight: 950;
        letter-spacing: -3px;
        margin: 14px 0 9px;
    }

    .result-band {
        color: #9aa7b7;
        font-size: 13px;
    }

    .result-foot {
        color: #687588;
        font-size: 11px;
        line-height: 1.6;
        margin-top: 13px;
    }

    .market-card {
        padding: 28px;
        border-radius: 22px;
        border: 1px solid rgba(122,167,255,.14);
        background: linear-gradient(145deg, #111b2a, #0b111a);
        box-shadow: 0 20px 55px rgba(0,0,0,.18);
    }

    .market-value {
        font-size: 48px;
        font-weight: 950;
        letter-spacing: -2px;
        margin-top: 6px;
    }

    .market-change {
        color: #8e9bad;
        font-size: 13px;
        margin-top: 5px;
    }

    .insight-card {
        padding: 19px 21px;
        border-radius: 17px;
        border: 1px solid var(--border);
        background: #0d141e;
        margin-top: 14px;
    }

    .insight-title {
        color: #c9d2de;
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .insight-text {
        color: #8491a3;
        font-size: 13px;
        line-height: 1.7;
        margin-top: 6px;
    }

    .status-ok { color: var(--accent); font-weight: 800; }
    .status-warn { color: var(--warning); font-weight: 800; }
    .status-bad { color: var(--danger); font-weight: 800; }

    .footer {
        color: #536073;
        text-align: center;
        font-size: 11px;
        margin-top: 65px;
        padding-top: 22px;
        border-top: 1px solid var(--border);
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] > div {
        border-radius: 12px;
    }

    .stButton > button {
        border-radius: 12px;
        min-height: 45px;
        font-weight: 850;
    }

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    @media (max-width: 800px) {
        .hero { padding: 28px 23px; }
        .player-name { font-size: 28px; }
        .probability { font-size: 52px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PATH DISCOVERY
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent


def find_project_file(filename):
    candidates = [
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "data" / filename,
        PROJECT_ROOT / "models" / filename,
        APP_DIR / filename,
        APP_DIR / "data" / filename,
        APP_DIR / "models" / filename,
    ]

    seen = set()

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)

        if candidate.is_file():
            return candidate

    for root in [PROJECT_ROOT, APP_DIR]:
        try:
            for candidate in root.glob(f"**/{filename}"):
                if candidate.is_file():
                    return candidate
        except Exception:
            pass

    return None


DATA_PATH = find_project_file("transfer_training_updated.csv")
TRANSFER_MODEL_PATH = find_project_file("transfer_model.joblib")
TRANSFER_INFO_PATH = find_project_file("transfer_model_info.joblib")
MARKET_MODEL_PATH = find_project_file("market_value_model.joblib")
MARKET_INFO_PATH = find_project_file("market_value_model_info.joblib")

# ============================================================
# LOADERS
# ============================================================


@st.cache_data(show_spinner=False)
def load_dataset(path):
    return pd.read_csv(path, low_memory=False)


@st.cache_resource(show_spinner=False)
def load_joblib(path):
    return joblib.load(path)


if DATA_PATH is None:
    st.error(
        "Dataset not found. TransferLens expected "
        "`transfer_training_updated.csv` inside the project."
    )
    st.stop()

if TRANSFER_MODEL_PATH is None:
    st.error(
        "Transfer model not found. TransferLens expected "
        "`transfer_model.joblib` inside the project."
    )
    st.stop()

try:
    raw_df = load_dataset(str(DATA_PATH)).copy()
except Exception as exc:
    st.error("TransferLens could not read the dataset.")
    st.exception(exc)
    st.stop()

try:
    transfer_model = load_joblib(str(TRANSFER_MODEL_PATH))
except Exception as exc:
    st.error("TransferLens could not load the transfer model.")
    st.exception(exc)
    st.stop()

if TRANSFER_INFO_PATH:
    try:
        transfer_info = load_joblib(str(TRANSFER_INFO_PATH))
    except Exception:
        transfer_info = {}
else:
    transfer_info = {}

market_model = None
market_info = {}

if MARKET_MODEL_PATH:
    try:
        market_model = load_joblib(str(MARKET_MODEL_PATH))
    except Exception:
        market_model = None

if MARKET_INFO_PATH:
    try:
        market_info = load_joblib(str(MARKET_INFO_PATH))
    except Exception:
        market_info = {}

# ============================================================
# GENERIC HELPERS
# ============================================================


def first_present(mapping, keys, default=None):
    if not isinstance(mapping, dict):
        return default

    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value

    return default


def clean_text(value, default="N/A"):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    text = str(value).strip()

    if text.lower() in {"", "nan", "none", "null", "nat"}:
        return default

    return text


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default

        value = float(value)

        return value if np.isfinite(value) else default
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default

        return int(float(value))
    except Exception:
        return default


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
    if value is None:
        return np.nan

    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass

    text = str(value).strip()

    if not text:
        return np.nan

    try:
        return int(text[:4])
    except Exception:
        return np.nan


def season_label(year):
    try:
        year = int(year)

        return (
            f"{year}-{str(year + 1)[-2:]}"
        )
    except Exception:
        return clean_text(year)


def calculate_age(dob, season_year):
    if dob is None or season_year is None:
        return np.nan

    try:
        if pd.isna(dob) or pd.isna(season_year):
            return np.nan

        dob = pd.to_datetime(
            dob,
            errors="coerce",
        )

        if pd.isna(dob):
            return np.nan

        season_start = datetime(
            int(season_year),
            8,
            1,
        )

        age = (
            season_start.year
            - dob.year
        )

        if (
            season_start.month,
            season_start.day,
        ) < (
            dob.month,
            dob.day,
        ):
            age -= 1

        return age

    except Exception:
        return np.nan


def format_money(value):
    value = safe_float(
        value,
        default=np.nan,
    )

    if (
        not np.isfinite(value)
        or value <= 0
    ):
        return "N/A"

    if value >= 1_000_000_000:
        return (
            f"€{value / 1_000_000_000:.2f}B"
        )

    if value >= 1_000_000:
        return (
            f"€{value / 1_000_000:.2f}M"
        )

    if value >= 1_000:
        return (
            f"€{value / 1_000:.0f}K"
        )

    return f"€{value:,.0f}"


def metric_text(
    value,
    percentage=False,
    decimals=4,
):
    if value is None:
        return "—"

    try:
        value = float(value)

        if percentage:
            return (
                f"{value * 100:.2f}%"
            )

        return (
            f"{value:.{decimals}f}"
        )

    except Exception:
        return "—"


def get_model_features(model, info):
    features = first_present(
        info,
        [
            "features",
            "feature_columns",
            "model_features",
            "columns",
        ],
        default=[],
    )

    if features is None:
        features = []

    if isinstance(features, tuple):
        features = list(features)

    if not isinstance(features, list):
        try:
            features = list(features)
        except Exception:
            features = []

    if not features:
        for attr in [
            "feature_names_",
            "feature_names",
        ]:
            try:
                value = getattr(
                    model,
                    attr,
                )

                if value is not None:
                    features = list(value)
                    break

            except Exception:
                pass

    return list(
        dict.fromkeys(
            [str(x) for x in features]
        )
    )


def get_model_categoricals(
    model,
    info,
    features,
):
    categorical = first_present(
        info,
        [
            "categorical_features",
            "categorical_columns",
            "cat_features",
            "categorical",
        ],
        default=[],
    )

    if categorical is None:
        categorical = []

    if not isinstance(
        categorical,
        list,
    ):
        try:
            categorical = list(
                categorical
            )
        except Exception:
            categorical = []

    categorical = [
        str(x)
        for x in categorical
    ]

    if not categorical:
        try:
            indices = (
                model
                .get_cat_feature_indices()
            )

            categorical = [
                features[i]
                for i in indices
                if 0 <= i < len(features)
            ]

        except Exception:
            pass

    return list(
        dict.fromkeys(categorical)
    )


def get_model_threshold(info):
    value = first_present(
        info,
        [
            "threshold",
            "decision_threshold",
            "classification_threshold",
        ],
        default=0.51,
    )

    try:
        value = float(value)

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    except Exception:
        return 0.51


# ============================================================
# TRANSFER MODEL METADATA
# ============================================================

transfer_features = get_model_features(
    transfer_model,
    transfer_info,
)

transfer_categorical = (
    get_model_categoricals(
        transfer_model,
        transfer_info,
        transfer_features,
    )
)

transfer_threshold = (
    get_model_threshold(
        transfer_info
    )
)

if not transfer_features:
    st.error(
        "The saved transfer model does not expose "
        "its feature columns. TransferLens cannot "
        "safely construct the model input."
    )
    st.stop()

# ============================================================
# FEATURE ENGINEERING
# ============================================================


def build_model_features(data):
    df = data.copy()

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    if "country_of_birth" not in df.columns:

        if "nationality" in df.columns:
            df["country_of_birth"] = (
                df["nationality"]
            )

        elif "nation" in df.columns:
            df["country_of_birth"] = (
                df["nation"]
            )

        else:
            df["country_of_birth"] = "Unknown"

    df["country_of_birth"] = (
        df["country_of_birth"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # SEASON
    # --------------------------------------------------------

    if "season" not in df.columns:

        if "season_year" in df.columns:
            df["season"] = (
                df["season_year"]
                .astype(str)
            )

        elif "Season" in df.columns:
            df["season"] = (
                df["Season"]
                .astype(str)
            )

        else:
            raise ValueError(
                "Dataset does not contain "
                "season information."
            )

    # --------------------------------------------------------
    # SEASON YEAR
    # --------------------------------------------------------

    if "season_year" in df.columns:
        df["season_year_numeric"] = (
            df["season_year"]
            .apply(season_to_year)
        )
    else:
        df["season_year_numeric"] = (
            df["season"]
            .apply(season_to_year)
        )

    df["season_year_numeric"] = (
        pd.to_numeric(
            df["season_year_numeric"],
            errors="coerce",
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # NUMERIC BASE COLUMNS
    # --------------------------------------------------------

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

    aliases = {
        "appearances": [
            "Apps",
            "appearances_total",
        ],
        "starts": [
            "Starts",
        ],
        "minutes_played": [
            "Minutes",
            "minutes",
        ],
        "goals": [
            "Goals",
        ],
        "assists": [
            "Assists",
        ],
        "yellow_cards": [
            "Yellow cards",
        ],
        "red_cards": [
            "Red cards",
        ],
        "market_value_in_eur": [
            "market_value",
            "Market Value",
        ],
    }

    for column in numeric_columns:

        if column not in df.columns:

            for alias in aliases.get(
                column,
                [],
            ):
                if alias in df.columns:
                    df[column] = (
                        df[alias]
                    )
                    break

        if column not in df.columns:
            df[column] = 0

        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .fillna(0)
        )

    # --------------------------------------------------------
    # DATE OF BIRTH
    # --------------------------------------------------------

    if "date_of_birth" in df.columns:

        df["date_of_birth"] = (
            pd.to_datetime(
                df["date_of_birth"],
                errors="coerce",
            )
        )

    else:
        df["date_of_birth"] = pd.NaT

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    if "age" in df.columns:

        df["age_numeric"] = (
            pd.to_numeric(
                df["age"],
                errors="coerce",
            )
        )

    else:
        df["age_numeric"] = np.nan

    calculated_age = df.apply(
        lambda r:
        calculate_age(
            r["date_of_birth"],
            r["season_year_numeric"],
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

    # --------------------------------------------------------
    # AGE FEATURES
    # --------------------------------------------------------

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
        & (df["age_numeric"] <= 29)
    ).astype(int)

    df["older_player"] = (
        df["age_numeric"] >= 30
    ).astype(int)

    # --------------------------------------------------------
    # PERFORMANCE FEATURES
    # --------------------------------------------------------

    df["goals_per_appearance"] = (
        safe_divide(
            df["goals"],
            df["appearances"],
        )
    )

    df["assists_per_appearance"] = (
        safe_divide(
            df["assists"],
            df["appearances"],
        )
    )

    df["goals_per_90"] = (
        safe_divide(
            df["goals"] * 90,
            df["minutes_played"],
        )
    )

    df["assists_per_90"] = (
        safe_divide(
            df["assists"] * 90,
            df["minutes_played"],
        )
    )

    df["minutes_per_appearance"] = (
        safe_divide(
            df["minutes_played"],
            df["appearances"],
        )
    )

    # Important: the previous app referenced this
    # feature but never actually constructed it.
    df["starts_ratio"] = (
        safe_divide(
            df["starts"],
            df["appearances"],
        )
    )

    df["goal_contributions"] = (
        df["goals"]
        + df["assists"]
    )

    df[
        "goal_contributions_per_appearance"
    ] = safe_divide(
        df["goal_contributions"],
        df["appearances"],
    )

    df[
        "goal_contributions_per_90"
    ] = safe_divide(
        df["goal_contributions"] * 90,
        df["minutes_played"],
    )

    # --------------------------------------------------------
    # DISCIPLINE
    # --------------------------------------------------------

    df["yellow_cards_per_appearance"] = (
        safe_divide(
            df["yellow_cards"],
            df["appearances"],
        )
    )

    df["red_cards_per_appearance"] = (
        safe_divide(
            df["red_cards"],
            df["appearances"],
        )
    )

    # --------------------------------------------------------
    # MARKET VALUE
    # --------------------------------------------------------

    df["market_value_in_eur"] = (
        df["market_value_in_eur"]
        .clip(lower=0)
    )

    df["market_value_log"] = (
        np.log1p(
            df["market_value_in_eur"]
        )
    )

    # --------------------------------------------------------
    # PLAYER GROUP
    # --------------------------------------------------------

    if "player_id" in df.columns:

        df["_player_group"] = (
            df["player_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    else:
        df["_player_group"] = df["name"]

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    df = (
        df.sort_values(
            [
                "_player_group",
                "season_year_numeric",
            ]
        )
        .reset_index(drop=True)
    )

    grouped = df.groupby(
        "_player_group",
        sort=False,
    )

    # --------------------------------------------------------
    # HISTORICAL FEATURES
    # --------------------------------------------------------

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
            df.groupby(
                "_player_group"
            )[column]
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
            df.groupby(
                "_player_group"
            )[column]
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

    # --------------------------------------------------------
    # PERFORMANCE TRENDS
    # --------------------------------------------------------

    trend_columns = [
        "goals",
        "assists",
        "minutes_played",
        "appearances",
        "market_value_in_eur",
    ]

    for column in trend_columns:

        previous = (
            grouped[column]
            .shift(1)
        )

        previous_two = (
            grouped[column]
            .shift(2)
        )

        df[
            f"{column}_change"
        ] = (
            df[column]
            - previous
        )

        df[
            f"{column}_trend"
        ] = (
            previous
            - previous_two
        )

    # --------------------------------------------------------
    # TRANSFER HISTORY
    # --------------------------------------------------------

    if "transferred_next_season" in df.columns:

        transfer_series = (
            df.groupby(
                "_player_group"
            )[
                "transferred_next_season"
            ]
        )

        df["previous_transfer"] = (
            transfer_series.shift(1)
        )

        df[
            "previous_transfer_rate"
        ] = (
            transfer_series.transform(
                lambda x:
                x.shift(1)
                .expanding()
                .mean()
            )
        )

        df[
            "previous_transfer_count"
        ] = (
            transfer_series.transform(
                lambda x:
                x.shift(1)
                .expanding()
                .sum()
            )
        )

        df[
            "previous_seasons_count"
        ] = (
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

    # --------------------------------------------------------
    # CAREER EXPERIENCE
    # --------------------------------------------------------

    df["career_seasons_before"] = (
        df.groupby(
            "_player_group"
        ).cumcount()
    )

    # --------------------------------------------------------
    # SEASON RELATIVE MARKET VALUE
    # --------------------------------------------------------

    season_median_value = (
        df.groupby(
            "season_year_numeric"
        )[
            "market_value_in_eur"
        ].transform("median")
    )

    df[
        "market_value_vs_season_median"
    ] = safe_divide(
        df["market_value_in_eur"],
        season_median_value,
    )

    # --------------------------------------------------------
    # INTERACTIONS
    # --------------------------------------------------------

    df["age_performance"] = (
        df["age_numeric"]
        * df[
            "goal_contributions_per_90"
        ]
    )

    df["value_performance"] = (
        np.log1p(
            df["market_value_in_eur"]
        )
        * df[
            "goal_contributions_per_90"
        ]
    )

    df["minutes_value_efficiency"] = (
        safe_divide(
            df["market_value_in_eur"],
            df["minutes_played"],
        )
    )

    df["goal_assist_ratio"] = (
        safe_divide(
            df["goals"],
            df["assists"] + 1,
        )
    )

    df[
        "minutes_per_appearance_squared"
    ] = (
        df["minutes_per_appearance"] ** 2
    )

    return df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


try:
    df = build_model_features(
        raw_df
    )

except Exception as exc:
    st.error(
        "Feature preparation failed."
    )
    st.exception(exc)
    st.stop()

# Make sure the saved model always receives
# exactly its own expected columns.
for feature in transfer_features:

    if feature not in df.columns:

        if feature in transfer_categorical:
            df[feature] = "Unknown"
        else:
            df[feature] = 0.0

# ============================================================
# MODEL INPUT
# ============================================================


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
            value = (
                "Unknown"
                if feature in categoricals
                else 0.0
            )

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
        row.get(
            "season_year_numeric"
        ),
    )

    overrides = {}

    if (
        "age_numeric"
        in transfer_features
        and pd.notna(age_value)
    ):
        overrides[
            "age_numeric"
        ] = float(age_value)

    if (
        "age"
        in transfer_features
        and pd.notna(age_value)
    ):
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
        transfer_model
        .predict_proba(X)[0][1]
    )

    probability = float(
        np.clip(
            probability,
            0,
            1,
        )
    )

    return (
        probability,
        X,
    )


def likelihood_band(
    probability
):
    pct = probability * 100

    if pct >= 75:
        return "Very high likelihood"

    if pct >= 55:
        return "High likelihood"

    if pct >= 40:
        return "Moderate likelihood"

    if pct >= 25:
        return "Low likelihood"

    return "Very low likelihood"


# ============================================================
# MARKET MODEL SUPPORT
# ============================================================

market_features = (
    get_model_features(
        market_model,
        market_info,
    )
    if market_model is not None
    else []
)

market_categorical = (
    get_model_categoricals(
        market_model,
        market_info,
        market_features,
    )
    if market_model is not None
    else []
)

# Existing TransferLens market model feature set.
if (
    not market_features
    and market_model is not None
):

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


def market_target_mode():

    value = first_present(
        market_info,
        [
            "target_transform",
            "prediction_target",
            "target",
            "target_column",
            "output_type",
        ],
        default=None,
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


def prepare_market_history():

    history = df.copy()

    history["_market_value"] = (
        pd.to_numeric(
            history[
                "market_value_in_eur"
            ],
            errors="coerce",
        )
        .fillna(0)
    )

    history = (
        history
        .sort_values(
            [
                "_player_group",
                "season_year_numeric",
            ]
        )
        .reset_index(drop=True)
    )

    history[
        "previous_market_value"
    ] = (
        history.groupby(
            "_player_group"
        )["_market_value"]
        .shift(1)
    )

    history[
        "two_back_market_value"
    ] = (
        history.groupby(
            "_player_group"
        )["_market_value"]
        .shift(2)
    )

    history[
        "previous_market_value_change"
    ] = (
        history[
            "previous_market_value"
        ]
        - history[
            "two_back_market_value"
        ]
    )

    history[
        "previous_market_value_growth"
    ] = safe_divide(
        history[
            "previous_market_value_change"
        ],
        history[
            "two_back_market_value"
        ],
    )

    history[
        "career_seasons_before"
    ] = (
        history.groupby(
            "_player_group"
        ).cumcount()
    )

    return history


def predict_market_value(
    row,
    selected_season,
):
    if market_model is None:
        return None

    current_value = safe_float(
        row.get(
            "market_value_in_eur"
        ),
        default=0,
    )

    if current_value <= 0:
        return None

    history = (
        prepare_market_history()
    )

    player_key = clean_text(
        row.get(
            "_player_group"
        ),
        default="",
    )

    previous_rows = (
        history[
            (
                history[
                    "_player_group"
                ].astype(str)
                == player_key
            )
            & (
                history[
                    "season_year_numeric"
                ]
                < selected_season
            )
        ]
        .sort_values(
            "season_year_numeric"
        )
    )

    previous_values = (
        previous_rows[
            previous_rows[
                "_market_value"
            ] > 0
        ]["_market_value"]
        .tolist()
    )

    previous_value = (
        float(
            previous_values[-1]
        )
        if previous_values
        else 0.0
    )

    next_year = (
        int(selected_season)
        + 1
    )

    next_label = season_label(
        next_year
    )

    current_age = calculate_age(
        row.get(
            "date_of_birth"
        ),
        selected_season,
    )

    if pd.isna(current_age):
        current_age = safe_float(
            row.get(
                "age_numeric"
            ),
            25,
        )

    else:
        current_age = float(
            current_age
        )

    value_change = (
        current_value
        - previous_value
        if previous_value > 0
        else 0.0
    )

    value_growth = (
        value_change
        / previous_value
        if previous_value > 0
        else 0.0
    )

    career_next = (
        len(previous_rows)
        + 1
    )

    overrides = {
        "season": next_label,
        "season_year_numeric": next_year,
        "age_numeric": current_age + 1,
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
            np.exp(
                raw_prediction
            )
        )

    elif mode == "ratio":

        ratio = raw_prediction

    else:

        predicted_value = max(
            raw_prediction,
            0.0,
        )

        return {
            "predicted_value":
                predicted_value,
            "current_value":
                current_value,
            "previous_value":
                previous_value,
            "next_season":
                next_label,
            "ratio":
                (
                    predicted_value
                    / current_value
                    if current_value > 0
                    else np.nan
                ),
            "X": X,
        }

    ratio = float(
        np.clip(
            ratio,
            0.55,
            1.45,
        )
    )

    predicted_value = (
        current_value
        * ratio
    )

    return {
        "predicted_value":
            float(predicted_value),
        "current_value":
            current_value,
        "previous_value":
            previous_value,
        "next_season":
            next_label,
        "ratio":
            ratio,
        "X": X,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">⚽</div>
            <div>
                <div class="brand-name">
                    TransferLens
                </div>
                <div class="brand-caption">
                    Football intelligence dashboard
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Project status"
    )

    st.markdown(
        "**Dataset**  \n"
        "<span class='status-ok'>● Connected</span>"
        if DATA_PATH
        else
        "<span class='status-bad'>● Missing</span>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "**Transfer model**  \n"
        "<span class='status-ok'>● Loaded</span>"
        if TRANSFER_MODEL_PATH
        else
        "<span class='status-bad'>● Missing</span>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "**Market model**  \n"
        "<span class='status-ok'>● Loaded</span>"
        if market_model is not None
        else
        "<span class='status-warn'>● Unavailable</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        "### Dataset"
    )

    st.caption(
        f"{len(df):,} player-season rows"
    )

    st.caption(
        f"{df['name'].nunique():,} unique players"
    )

    season_min = int(
        df[
            "season_year_numeric"
        ].min()
    )

    season_max = int(
        df[
            "season_year_numeric"
        ].max()
    )

    if (
        season_min > 0
        and season_max > 0
    ):
        st.caption(
            "Coverage: "
            f"{season_label(season_min)}"
            " → "
            f"{season_label(season_max)}"
        )

    st.divider()

    st.markdown(
        "### Model"
    )

    st.caption(
        "Algorithm: "
        f"{type(transfer_model).__name__}"
    )

    st.caption(
        f"Features: {len(transfer_features)}"
    )

    st.caption(
        f"Threshold: {transfer_threshold:.3f}"
    )

    st.divider()

    if st.button(
        "Clear analysis",
        width="stretch",
    ):

        for key in [
            "player_search",
            "selected_player",
            "selected_season",
            "run_transfer",
            "run_market",
        ]:
            st.session_state.pop(
                key,
                None,
            )

        st.rerun()

# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-kicker">
            Football Analytics · Machine Learning
        </div>

        <div class="hero-title">
            ⚽ TransferLens
        </div>

        <div class="hero-subtitle">
            Explore player performance. Understand career movement.
            Estimate transfer likelihood.
        </div>

        <div class="hero-copy">
            TransferLens combines real player-season data with the
            trained transfer model already present in this project.
            Select any player available in the dataset, inspect their
            history, and generate model-backed transfer and market-value
            projections.
        </div>

        <div class="hero-badges">
            <span class="hero-badge">
                {len(df):,} player-season rows
            </span>

            <span class="hero-badge">
                {df['name'].nunique():,} players
            </span>

            <span class="hero-badge">
                {len(transfer_features)} model features
            </span>

            <span class="hero-badge">
                Dataset-driven UI
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL SCOREBOARD
# ============================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">
                Model Performance
            </div>

            <div class="section-subtitle">
                Metrics are read from the saved model metadata
                when available.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

model_type = type(
    transfer_model
).__name__

accuracy = first_present(
    transfer_info,
    [
        "accuracy",
        "test_accuracy",
    ],
)

roc_auc = first_present(
    transfer_info,
    [
        "roc_auc",
        "roc_auc_score",
    ],
)

f1_value = first_present(
    transfer_info,
    [
        "f1_score",
        "f1",
    ],
)

score_items = [
    (
        "MODEL",
        model_type,
        "Saved transfer classifier",
    ),
    (
        "FEATURES",
        str(
            len(transfer_features)
        ),
        "Features expected by model",
    ),
    (
        "TEST ACCURACY",
        metric_text(
            accuracy,
            percentage=True,
        ),
        "Saved evaluation metric",
    ),
    (
        "ROC-AUC",
        metric_text(
            roc_auc
        ),
        "Saved evaluation metric",
    ),
    (
        "F1 SCORE",
        metric_text(
            f1_value
        ),
        "Saved evaluation metric",
    ),
]

cols = st.columns(
    len(score_items)
)

for col, (
    label,
    value,
    note,
) in zip(
    cols,
    score_items,
):

    with col:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    {escape(label)}
                </div>

                <div class="metric-value">
                    {escape(value)}
                </div>

                <div class="metric-note">
                    {escape(note)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# PLAYER SEARCH
# ============================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">
                Player Analysis
            </div>

            <div class="section-subtitle">
                Search the real dataset, choose a player,
                then select one of that player's actual seasons.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

search = st.text_input(
    "Search player",
    placeholder="Type a player name...",
    key="player_search",
)

search = search.strip()

if not search:

    st.markdown(
        """
        <div class="insight-card">
            <div class="insight-title">
                Start here
            </div>

            <div class="insight-text">
                Enter a player name above. Results are generated
                directly from the loaded dataset — no player names
                or statistics are hardcoded into the application.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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

    close_matches = (
        get_close_matches(
            search,
            all_names,
            n=8,
            cutoff=0.30,
        )
    )

    st.warning(
        f"No exact dataset match found for “{search}”."
    )

    if close_matches:

        st.caption(
            "Possible matches:"
        )

        st.write(
            " · ".join(
                close_matches
            )
        )

    st.stop()

# Keep selection valid when the search text changes.
if (
    st.session_state.get(
        "selected_player"
    )
    not in matches
):
    st.session_state[
        "selected_player"
    ] = matches[0]

selected_name = st.selectbox(
    "Player",
    matches,
    key="selected_player",
)

player_matches = df[
    df["name"]
    == selected_name
].copy()

if player_matches.empty:
    st.warning(
        "No player records are available."
    )
    st.stop()

season_values = (
    player_matches[
        "season_year_numeric"
    ]
    .dropna()
    .astype(int)
    .loc[
        lambda s: s > 0
    ]
    .drop_duplicates()
    .sort_values(
        ascending=False
    )
    .tolist()
)

if not season_values:

    st.warning(
        "No season information is available "
        "for this player."
    )

    st.stop()

season_options = {
    season_label(year): year
    for year in season_values
}

if (
    st.session_state.get(
        "selected_season"
    )
    not in season_options
):
    st.session_state[
        "selected_season"
    ] = next(
        iter(
            season_options
        )
    )

selected_season_label = (
    st.selectbox(
        "Season",
        list(
            season_options.keys()
        ),
        key="selected_season",
    )
)

selected_season = (
    season_options[
        selected_season_label
    ]
)

selected_rows = player_matches[
    player_matches[
        "season_year_numeric"
    ]
    == selected_season
].copy()

if selected_rows.empty:

    st.warning(
        "No data is available for "
        "the selected season."
    )

    st.stop()

# If multiple records exist for the same player-season,
# use the record with the greatest minutes.
if len(selected_rows) > 1:

    selected_rows = (
        selected_rows
        .sort_values(
            "minutes_played",
            ascending=False,
        )
    )

row = (
    selected_rows
    .iloc[0]
    .copy()
)

# ============================================================
# PLAYER HEADER
# ============================================================

age_value = calculate_age(
    row.get(
        "date_of_birth"
    ),
    selected_season,
)

if pd.isna(age_value):

    age_display = clean_text(
        row.get("age"),
        "N/A",
    )

else:
    age_display = str(
        int(age_value)
    )

position = clean_text(
    row.get(
        "position"
    ),
    "N/A",
)

nationality = clean_text(
    row.get(
        "country_of_birth"
    ),
    "N/A",
)

club = "N/A"

for club_column in [
    "club",
    "current_club",
    "club_name",
    "team",
    "squad",
]:

    if club_column in row.index:

        candidate = clean_text(
            row.get(
                club_column
            ),
            "N/A",
        )

        if candidate != "N/A":
            club = candidate
            break

st.markdown(
    f"""
    <div class="player-card">
        <div class="player-kicker">
            Selected player
        </div>

        <div class="player-name">
            {escape(selected_name)}
        </div>

        <div class="player-meta">
            {escape(position)}
            &nbsp;•&nbsp;
            Age {escape(age_display)}
            &nbsp;•&nbsp;
            {escape(selected_season_label)}
            &nbsp;•&nbsp;
            {escape(nationality)}
            &nbsp;•&nbsp;
            {escape(club)}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SEASON SNAPSHOT
# ============================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">
                Season Snapshot
            </div>

            <div class="section-subtitle">
                Actual values from the selected player-season row.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

apps = safe_int(
    row.get("appearances")
)

starts = safe_int(
    row.get("starts")
)

minutes = safe_int(
    row.get(
        "minutes_played"
    )
)

goals = safe_int(
    row.get("goals")
)

assists = safe_int(
    row.get("assists")
)

market_value = safe_float(
    row.get(
        "market_value_in_eur"
    ),
    0,
)

snapshot = [
    (
        "Appearances",
        f"{apps:,}",
    ),
    (
        "Starts",
        f"{starts:,}",
    ),
    (
        "Minutes",
        f"{minutes:,}",
    ),
    (
        "Goals",
        f"{goals:,}",
    ),
    (
        "Assists",
        f"{assists:,}",
    ),
    (
        "Market Value",
        format_money(
            market_value
        ),
    ),
]

cols = st.columns(
    len(snapshot)
)

for col, (
    label,
    value,
) in zip(
    cols,
    snapshot,
):

    with col:

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">
                    {escape(label)}
                </div>

                <div class="stat-value">
                    {escape(value)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# PERFORMANCE TABS
# ============================================================

(
    tab_overview,
    tab_history,
    tab_data,
) = st.tabs(
    [
        "📊 Performance",
        "📅 Career History",
        "🧾 Raw Season Data",
    ]
)

with tab_overview:

    more_stats = [
        (
            "Goals / 90",
            safe_float(
                row.get(
                    "goals_per_90"
                )
            ),
        ),
        (
            "Assists / 90",
            safe_float(
                row.get(
                    "assists_per_90"
                )
            ),
        ),
        (
            "Goals / Appearance",
            safe_float(
                row.get(
                    "goals_per_appearance"
                )
            ),
        ),
        (
            "Assists / Appearance",
            safe_float(
                row.get(
                    "assists_per_appearance"
                )
            ),
        ),
        (
            "Contributions / 90",
            safe_float(
                row.get(
                    "goal_contributions_per_90"
                )
            ),
        ),
        (
            "Starts Ratio",
            safe_float(
                row.get(
                    "starts_ratio"
                )
            ),
        ),
    ]

    cols = st.columns(3)

    for col, (
        label,
        value,
    ) in zip(
        cols,
        more_stats,
    ):

        with col:

            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">
                        {escape(label)}
                    </div>

                    <div class="stat-value">
                        {value:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    chart_data = (
        player_matches[
            [
                "season_year_numeric",
                "goals",
                "assists",
            ]
        ]
        .copy()
    )

    chart_data = (
        chart_data
        .dropna(
            subset=[
                "season_year_numeric"
            ]
        )
        .drop_duplicates(
            "season_year_numeric"
        )
        .sort_values(
            "season_year_numeric"
        )
        .set_index(
            "season_year_numeric"
        )
    )

    if len(chart_data) >= 2:

        st.markdown(
            "#### Goals & assists by season"
        )

        st.line_chart(
            chart_data[
                [
                    "goals",
                    "assists",
                ]
            ],
            width="stretch",
        )

with tab_history:

    history_rows = []

    for season_year in sorted(
        season_values,
        reverse=True,
    ):

        season_rows = (
            player_matches[
                player_matches[
                    "season_year_numeric"
                ]
                == season_year
            ]
            .copy()
        )

        if season_rows.empty:
            continue

        if len(season_rows) > 1:

            season_rows = (
                season_rows
                .sort_values(
                    "minutes_played",
                    ascending=False,
                )
            )

        history_row = (
            season_rows
            .iloc[0]
        )

        history_rows.append(
            {
                "Season":
                    season_label(
                        season_year
                    ),
                "Apps":
                    safe_int(
                        history_row.get(
                            "appearances"
                        )
                    ),
                "Starts":
                    safe_int(
                        history_row.get(
                            "starts"
                        )
                    ),
                "Minutes":
                    safe_int(
                        history_row.get(
                            "minutes_played"
                        )
                    ),
                "Goals":
                    safe_int(
                        history_row.get(
                            "goals"
                        )
                    ),
                "Assists":
                    safe_int(
                        history_row.get(
                            "assists"
                        )
                    ),
                "Market Value":
                    format_money(
                        history_row.get(
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

        market_chart = (
            player_matches[
                [
                    "season_year_numeric",
                    "market_value_in_eur",
                ]
            ]
            .dropna(
                subset=[
                    "season_year_numeric"
                ]
            )
            .drop_duplicates(
                "season_year_numeric"
            )
            .sort_values(
                "season_year_numeric"
            )
            .set_index(
                "season_year_numeric"
            )[
                "market_value_in_eur"
            ]
        )

        if len(market_chart) >= 2:

            st.markdown(
                "#### Market value trajectory"
            )

            st.line_chart(
                market_chart,
                width="stretch",
            )

    else:

        st.info(
            "No historical records are available."
        )

with tab_data:

    display_columns = [
        "name",
        "position",
        "season",
        "season_year_numeric",
        "appearances",
        "starts",
        "minutes_played",
        "goals",
        "assists",
        "market_value_in_eur",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in row.index
    ]

    raw_view = (
        player_matches[
            available_columns
        ]
        .sort_values(
            "season_year_numeric",
            ascending=False,
        )
        .copy()
    )

    st.dataframe(
        raw_view,
        width="stretch",
        hide_index=True,
    )

# ============================================================
# AI FORECAST CONTROLS
# ============================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">
                AI Forecast
            </div>

            <div class="section-subtitle">
                Run the saved TransferLens models against the selected
                player-season. No player-specific values are hardcoded.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

predict_col, market_col = (
    st.columns(2)
)

with predict_col:

    run_transfer = st.button(
        "🎯 Predict Transfer Likelihood",
        type="primary",
        width="stretch",
        key="run_transfer",
    )

with market_col:

    run_market = st.button(
        "💰 Predict Next-Season Value",
        width="stretch",
        key="run_market",
    )

# ============================================================
# TRANSFER PREDICTION
# ============================================================

if run_transfer:

    try:

        (
            probability,
            X_transfer,
        ) = predict_transfer(
            row
        )

    except Exception as exc:

        st.error(
            "Transfer prediction failed. "
            "The saved model and feature metadata "
            "could not be reconciled with this player-season."
        )

        st.exception(exc)
        st.stop()

    pct = (
        probability * 100
    )

    band = likelihood_band(
        probability
    )

    if (
        probability
        >= transfer_threshold
    ):

        status = (
            "LIKELY TO BE TRANSFERRED"
        )

        explanation = (
            "The predicted probability is at or above "
            "the model's configured decision threshold."
        )

    else:

        status = (
            "NOT LIKELY TO BE TRANSFERRED"
        )

        explanation = (
            "The predicted probability is below "
            "the model's configured decision threshold."
        )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-badge">
                TRANSFER MODEL OUTPUT
            </div>

            <div class="result-title">
                {escape(status)}
            </div>

            <div class="probability">
                {pct:.1f}%
            </div>

            <div class="result-band">
                {escape(band)}
            </div>

            <div class="result-foot">
                Decision threshold:
                {transfer_threshold:.3f}
                <br>
                {escape(explanation)}
                <br>
                This is a model probability,
                not confirmation of an actual transfer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        probability,
        text=(
            f"Transfer probability: "
            f"{pct:.1f}%"
        ),
    )

    with st.expander(
        "Model input used for this prediction"
    ):

        input_preview = (
            X_transfer.T
            .reset_index()
        )

        input_preview.columns = [
            "Feature",
            "Value",
        ]

        st.dataframe(
            input_preview,
            width="stretch",
            hide_index=True,
        )

    try:

        importances = (
            transfer_model
            .get_feature_importance()
        )

        if (
            len(importances)
            == len(transfer_features)
        ):

            importance_df = pd.DataFrame(
                {
                    "Feature":
                        transfer_features,
                    "Importance":
                        importances,
                }
            ).sort_values(
                "Importance",
                ascending=False,
            )

            st.markdown(
                "#### Model feature importance"
            )

            st.bar_chart(
                importance_df
                .head(12)
                .set_index(
                    "Feature"
                )[
                    "Importance"
                ],
                width="stretch",
            )

    except Exception:
        pass

# ============================================================
# MARKET VALUE PREDICTION
# ============================================================

if run_market:

    if market_model is None:

        st.warning(
            "Market-value model unavailable. "
            "TransferLens found no usable "
            "`market_value_model.joblib`."
        )

    else:

        try:

            result = (
                predict_market_value(
                    row,
                    selected_season,
                )
            )

        except Exception as exc:

            result = None

            st.error(
                "Market-value prediction failed. "
                "The saved market model's feature structure "
                "does not match the available project data."
            )

            st.exception(exc)

        if result is not None:

            predicted_value = (
                result[
                    "predicted_value"
                ]
            )

            current_value = (
                result[
                    "current_value"
                ]
            )

            previous_value = (
                result[
                    "previous_value"
                ]
            )

            next_season = (
                result[
                    "next_season"
                ]
            )

            change_pct = (
                (
                    (
                        predicted_value
                        - current_value
                    )
                    / current_value
                )
                * 100
                if current_value > 0
                else np.nan
            )

            sign = (
                "+"
                if change_pct >= 0
                else ""
            )

            st.markdown(
                f"""
                <div class="market-card">
                    <div class="stat-label">
                        PREDICTED NEXT-SEASON MARKET VALUE
                    </div>

                    <div class="market-value">
                        {escape(
                            format_money(
                                predicted_value
                            )
                        )}
                    </div>

                    <div class="market-change">
                        {sign}{change_pct:.1f}%
                        estimated change from
                        {escape(
                            selected_season_label
                        )}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            cols = st.columns(3)

            details = [
                (
                    "Current Value",
                    format_money(
                        current_value
                    ),
                ),
                (
                    "Previous Value",
                    (
                        format_money(
                            previous_value
                        )
                        if previous_value > 0
                        else "N/A"
                    ),
                ),
                (
                    "Projected Season",
                    next_season,
                ),
            ]

            for col, (
                label,
                value,
            ) in zip(
                cols,
                details,
            ):

                with col:

                    st.markdown(
                        f"""
                        <div class="stat-card">
                            <div class="stat-label">
                                {escape(label)}
                            </div>

                            <div class="stat-value">
                                {escape(str(value))}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown(
                """
                <div class="insight-card">
                    <div class="insight-title">
                        Market-value interpretation
                    </div>

                    <div class="insight-text">
                        This projection is generated by the saved
                        market-value model using the selected season
                        as the starting point. It is an estimate,
                        not a guaranteed future valuation.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            market_history = (
                player_matches[
                    [
                        "season_year_numeric",
                        "market_value_in_eur",
                    ]
                ]
                .dropna(
                    subset=[
                        "season_year_numeric"
                    ]
                )
                .drop_duplicates(
                    "season_year_numeric"
                )
                .sort_values(
                    "season_year_numeric"
                )
                .set_index(
                    "season_year_numeric"
                )[
                    "market_value_in_eur"
                ]
            )

            if len(
                market_history
            ) >= 2:

                st.markdown(
                    "#### Historical market value"
                )

                st.line_chart(
                    market_history,
                    width="stretch",
                )

            with st.expander(
                "Market model input"
            ):

                preview = (
                    result["X"]
                    .T
                    .reset_index()
                )

                preview.columns = [
                    "Feature",
                    "Value",
                ]

                st.dataframe(
                    preview,
                    width="stretch",
                    hide_index=True,
                )

# ============================================================
# DATA / MODEL HEALTH
# ============================================================

with st.expander(
    "🔧 Data & model diagnostics"
):

    diagnostic_cols = (
        st.columns(4)
    )

    diagnostics = [
        (
            "Dataset",
            (
                "Ready"
                if DATA_PATH
                else "Missing"
            ),
        ),
        (
            "Transfer model",
            (
                "Ready"
                if TRANSFER_MODEL_PATH
                else "Missing"
            ),
        ),
        (
            "Transfer features",
            str(
                len(
                    transfer_features
                )
            ),
        ),
        (
            "Market model",
            (
                "Ready"
                if market_model
                is not None
                else "Unavailable"
            ),
        ),
    ]

    for col, (
        label,
        value,
    ) in zip(
        diagnostic_cols,
        diagnostics,
    ):

        with col:

            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">
                        {escape(label)}
                    </div>

                    <div class="stat-value">
                        {escape(value)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        f"Dataset: {DATA_PATH}"
    )

    st.caption(
        "Transfer model: "
        f"{TRANSFER_MODEL_PATH}"
    )

    if MARKET_MODEL_PATH:

        st.caption(
            "Market model: "
            f"{MARKET_MODEL_PATH}"
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        TransferLens · Dataset-driven football analytics ·
        Saved machine-learning models
    </div>
    """,
    unsafe_allow_html=True,
)
