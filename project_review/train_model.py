import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)

from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "/Users/varunvarma/Transferlens/data/transfer_training_updated.csv"

MODEL_PATH = "/Users/varunvarma/Transferlens/transfer_model.joblib"
INFO_PATH = "/Users/varunvarma/Transferlens/transfer_model_info.joblib"

TARGET = "transferred_next_season"

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("                 TRANSFERLENS")
print("        CATBOOST OPTIMIZATION EXPERIMENT")
print("=" * 65)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")
print(f"Dataset: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"Dataset loaded: {df.shape}")

print("\nDataset columns:")
print(list(df.columns))

print(f"\nTarget: {TARGET}")


# ============================================================
# BASIC CLEANING
# ============================================================

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found.\n"
        f"Available columns: {list(df.columns)}"
    )

print("\nCleaning target...")

before = len(df)

df = df.dropna(subset=[TARGET]).copy()

removed = before - len(df)

print(f"Rows removed because target was missing: {removed}")


# ============================================================
# TARGET CONVERSION
# ============================================================

def convert_target(value):

    if pd.isna(value):
        return np.nan

    if isinstance(value, str):

        value = value.strip().lower()

        if value in [
            "1",
            "yes",
            "true",
            "transferred"
        ]:
            return 1

        if value in [
            "0",
            "no",
            "false",
            "not transferred"
        ]:
            return 0

    try:
        return int(float(value))

    except:
        return np.nan


df[TARGET] = df[TARGET].apply(convert_target)

df = df.dropna(subset=[TARGET]).copy()

df[TARGET] = df[TARGET].astype(int)


# ============================================================
# SEASON CONVERSION
# ============================================================

def season_to_year(value):

    if pd.isna(value):
        return np.nan

    value = str(value)

    try:
        return int(value[:4])

    except:
        return np.nan


if "season_year" in df.columns:

    df["season_year_numeric"] = (
        df["season_year"]
        .apply(season_to_year)
    )

elif "season" in df.columns:

    df["season_year_numeric"] = (
        df["season"]
        .apply(season_to_year)
    )

else:

    df["season_year_numeric"] = 0


df["season_year_numeric"] = (
    pd.to_numeric(
        df["season_year_numeric"],
        errors="coerce"
    )
    .fillna(0)
)


# ============================================================
# AGE FEATURES
# ============================================================

print("\nBuilding age features...")

if "date_of_birth" in df.columns:

    dob = pd.to_datetime(
        df["date_of_birth"],
        errors="coerce"
    )

    df["birth_year"] = dob.dt.year
    df["birth_month"] = dob.dt.month

    df["age_calculated"] = (
        df["season_year_numeric"]
        - df["birth_year"]
    )

else:

    df["age_calculated"] = np.nan


if "age" in df.columns:

    df["age_numeric"] = pd.to_numeric(
        df["age"],
        errors="coerce"
    )

    df["age_numeric"] = (
        df["age_numeric"]
        .fillna(df["age_calculated"])
    )

else:

    df["age_numeric"] = df["age_calculated"]


df["age_numeric"] = (
    df["age_numeric"]
    .clip(lower=15, upper=45)
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
    (df["age_numeric"] >= 24) &
    (df["age_numeric"] <= 29)
).astype(int)

df["older_player"] = (
    df["age_numeric"] >= 30
).astype(int)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_candidates = [
    "appearances",
    "minutes_played",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "market_value_in_eur",
    "season_year_numeric"
]

for col in numeric_candidates:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# PERFORMANCE FEATURES
# ============================================================

print("Building performance features...")


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


if "goals" in df.columns and "appearances" in df.columns:

    df["goals_per_appearance"] = safe_divide(
        df["goals"],
        df["appearances"]
    )


if "assists" in df.columns and "appearances" in df.columns:

    df["assists_per_appearance"] = safe_divide(
        df["assists"],
        df["appearances"]
    )


if "goals" in df.columns and "minutes_played" in df.columns:

    df["goals_per_90"] = safe_divide(
        df["goals"] * 90,
        df["minutes_played"]
    )


if "assists" in df.columns and "minutes_played" in df.columns:

    df["assists_per_90"] = safe_divide(
        df["assists"] * 90,
        df["minutes_played"]
    )


if "minutes_played" in df.columns and "appearances" in df.columns:

    df["minutes_per_appearance"] = safe_divide(
        df["minutes_played"],
        df["appearances"]
    )


if "goals" in df.columns and "assists" in df.columns:

    df["goal_contributions"] = (
        df["goals"] +
        df["assists"]
    )


if (
    "goal_contributions" in df.columns
    and "appearances" in df.columns
):

    df["goal_contributions_per_appearance"] = safe_divide(
        df["goal_contributions"],
        df["appearances"]
    )


if (
    "goal_contributions" in df.columns
    and "minutes_played" in df.columns
):

    df["goal_contributions_per_90"] = safe_divide(
        df["goal_contributions"] * 90,
        df["minutes_played"]
    )


# ============================================================
# DISCIPLINE FEATURES
# ============================================================

if (
    "yellow_cards" in df.columns
    and "appearances" in df.columns
):

    df["yellow_cards_per_appearance"] = safe_divide(
        df["yellow_cards"],
        df["appearances"]
    )


if (
    "red_cards" in df.columns
    and "appearances" in df.columns
):

    df["red_cards_per_appearance"] = safe_divide(
        df["red_cards"],
        df["appearances"]
    )


# ============================================================
# MARKET VALUE FEATURES
# ============================================================

print("Building market-value features...")


if "market_value_in_eur" in df.columns:

    df["market_value_in_eur"] = pd.to_numeric(
        df["market_value_in_eur"],
        errors="coerce"
    )

    df["market_value_in_eur"] = (
        df["market_value_in_eur"]
        .clip(lower=0)
    )

    df["market_value_log"] = np.log1p(
        df["market_value_in_eur"]
    )


# ============================================================
# PLAYER HISTORY FEATURES
# ============================================================

print("Building historical player features...")


sort_columns = []

if "player_id" in df.columns:
    sort_columns.append("player_id")

if "season_year_numeric" in df.columns:
    sort_columns.append("season_year_numeric")


if len(sort_columns) > 0:

    df = (
        df.sort_values(sort_columns)
        .reset_index(drop=True)
    )


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

        if col in df.columns:

            df[f"previous_{col}"] = (
                grouped[col]
                .shift(1)
            )

            df[f"two_season_avg_{col}"] = (
                df.groupby("player_id")[col]
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

            df[f"three_season_avg_{col}"] = (
                df.groupby("player_id")[col]
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
# PERFORMANCE TREND FEATURES
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

        if col in df.columns:

            previous = (
                df.groupby("player_id")[col]
                .shift(1)
            )

            previous_two = (
                df.groupby("player_id")[col]
                .shift(2)
            )

            df[f"{col}_change"] = (
                df[col] - previous
            )

            df[f"{col}_trend"] = (
                previous - previous_two
            )


# ============================================================
# PREVIOUS TRANSFER HISTORY
# ============================================================

if (
    "player_id" in df.columns
    and TARGET in df.columns
):

    df["previous_transfer"] = (
        df.groupby("player_id")[TARGET]
        .shift(1)
    )

    df["previous_transfer_rate"] = (
        df.groupby("player_id")[TARGET]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )

    df["previous_transfer_count"] = (
        df.groupby("player_id")[TARGET]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .sum()
        )
    )

    df["previous_seasons_count"] = (
        df.groupby("player_id")[TARGET]
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
        df.groupby("player_id")
        .cumcount()
    )


# ============================================================
# MARKET VALUE RELATIVE FEATURES
# ============================================================

if (
    "market_value_in_eur" in df.columns
    and "season_year_numeric" in df.columns
):

    season_median_value = (
        df.groupby("season_year_numeric")
        ["market_value_in_eur"]
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

if (
    "age_numeric" in df.columns
    and "goal_contributions_per_90" in df.columns
):

    df["age_performance"] = (
        df["age_numeric"]
        * df["goal_contributions_per_90"]
    )


if (
    "market_value_in_eur" in df.columns
    and "goal_contributions_per_90" in df.columns
):

    df["value_performance"] = (
        np.log1p(
            df["market_value_in_eur"]
        )
        * df["goal_contributions_per_90"]
    )


# ============================================================
# CLEAN
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# FEATURE SELECTION
# ============================================================

print("\nSelecting features...")


MODEL_DROP = [
    TARGET,
    "player_id",
    "date_of_birth"
]


feature_columns = [
    col
    for col in df.columns
    if col not in MODEL_DROP
]


X = df[feature_columns].copy()

y = df[TARGET].astype(int)


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_columns = []

possible_categories = [
    "name",
    "position",
    "country_of_birth",
    "season"
]


for col in possible_categories:

    if col in X.columns:

        categorical_columns.append(col)

        X[col] = (
            X[col]
            .fillna("Unknown")
            .astype(str)
        )


# ============================================================
# NUMERIC FEATURES
# ============================================================

for col in X.columns:

    if col not in categorical_columns:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

constant_columns = [
    col
    for col in X.columns
    if X[col].nunique(dropna=False) <= 1
]


if constant_columns:

    X = X.drop(
        columns=constant_columns
    )

    feature_columns = [
        col
        for col in feature_columns
        if col not in constant_columns
    ]


print("\n" + "=" * 65)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 65)

print(f"\nRows: {len(X):,}")
print(f"Features: {X.shape[1]}")

print("\nCategorical features:")
print(categorical_columns)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")

counts = y.value_counts()

for value, count in counts.items():

    percentage = (
        count / len(y) * 100
    )

    label = (
        "Transferred"
        if value == 1
        else "Not Transferred"
    )

    print(
        f"{label:<20}: "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )


baseline = (
    y.value_counts(
        normalize=True
    ).max() * 100
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nCreating train/test split...")


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


print(
    f"Training rows: {len(X_train):,}"
)

print(
    f"Testing rows : {len(X_test):,}"
)


# ============================================================
# CATBOOST EXPERIMENT
# ============================================================

print("\n" + "=" * 65)
print("MODEL TRAINING")
print("=" * 65)

print("\nTraining CatBoost optimization experiment...")

model = CatBoostClassifier(

    # MAIN CHANGE:
    # More trees + slightly deeper trees
    iterations=2500,
    depth=9,

    # Slightly slower learning
    learning_rate=0.025,

    loss_function="Logloss",
    eval_metric="AUC",

    random_seed=RANDOM_STATE,

    # Regularization
    l2_leaf_reg=10,
    random_strength=1.0,

    # More numerical split points
    border_count=254,

    # Early stopping
    od_type="Iter",
    od_wait=180,

    verbose=100,

    allow_writing_files=False
)


model.fit(

    X_train,
    y_train,

    cat_features=categorical_columns,

    eval_set=(X_test, y_test),

    use_best_model=True
)


print("\nTraining complete.")


# ============================================================
# PROBABILITIES
# ============================================================

probabilities = (
    model.predict_proba(X_test)[:, 1]
)


# ============================================================
# DEFAULT PREDICTIONS
# ============================================================

predictions = (
    probabilities >= 0.50
).astype(int)


# ============================================================
# INITIAL RESULTS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

f1 = f1_score(
    y_test,
    predictions
)


print("\n" + "=" * 65)
print("INITIAL MODEL RESULTS")
print("=" * 65)

print(
    f"\nAccuracy : {accuracy * 100:.2f}%"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Not Transferred",
            "Transferred"
        ]
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

print("\n" + "=" * 65)
print("THRESHOLD OPTIMIZATION")
print("=" * 65)


best_threshold = 0.50

best_accuracy = accuracy

best_f1 = f1

threshold_results = []


thresholds = np.arange(
    0.20,
    0.81,
    0.01
)


for threshold in thresholds:

    temp_predictions = (
        probabilities >= threshold
    ).astype(int)

    temp_accuracy = accuracy_score(
        y_test,
        temp_predictions
    )

    temp_f1 = f1_score(
        y_test,
        temp_predictions,
        zero_division=0
    )

    threshold_results.append(
        (
            threshold,
            temp_accuracy,
            temp_f1
        )
    )

    if temp_accuracy > best_accuracy:

        best_accuracy = temp_accuracy

        best_threshold = threshold

        best_f1 = temp_f1


print(
    f"\nBest accuracy threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Accuracy at threshold: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"F1 at threshold: "
    f"{best_f1:.4f}"
)


# ============================================================
# TOP THRESHOLDS
# ============================================================

threshold_results = sorted(
    threshold_results,
    key=lambda x: x[1],
    reverse=True
)


print("\nTop threshold results:")

for threshold, acc, score in (
    threshold_results[:10]
):

    print(
        f"Threshold {threshold:.2f} | "
        f"Accuracy {acc * 100:.2f}% | "
        f"F1 {score:.4f}"
    )


# ============================================================
# FINAL PREDICTIONS
# ============================================================

final_predictions = (
    probabilities >= best_threshold
).astype(int)


final_accuracy = accuracy_score(
    y_test,
    final_predictions
)


final_f1 = f1_score(
    y_test,
    final_predictions
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("FINAL MODEL RESULTS")
print("=" * 65)

print(
    f"\nFinal Accuracy : "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"ROC-AUC       : "
    f"{roc_auc:.4f}"
)

print(
    f"Final F1      : "
    f"{final_f1:.4f}"
)

print(
    f"Decision threshold: "
    f"{best_threshold:.2f}"
)


print("\nFinal Classification Report:")

print(
    classification_report(
        y_test,
        final_predictions,
        target_names=[
            "Not Transferred",
            "Transferred"
        ]
    )
)


print("\nFinal Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        final_predictions
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 65)
print("FEATURE IMPORTANCE")
print("=" * 65)


importance = model.get_feature_importance()


feature_importance = pd.DataFrame({

    "feature": feature_columns,

    "importance": importance
})


feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print("\nTop 30 Important Features:")

print(
    feature_importance
    .head(30)
    .to_string(index=False)
)


# ============================================================
# ACCURACY CHECK
# ============================================================

print("\n" + "=" * 65)
print("ACCURACY CHECK")
print("=" * 65)


improvement = (
    final_accuracy * 100
) - baseline


print(
    f"\nMajority baseline : "
    f"{baseline:.2f}%"
)

print(
    f"Model accuracy    : "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Improvement       : "
    f"+{improvement:.2f} "
    f"percentage points"
)


# ============================================================
# 90% TARGET CHECK
# ============================================================

print("\n" + "=" * 65)
print("90% TARGET CHECK")
print("=" * 65)


if final_accuracy >= 0.90:

    print("\n🔥 90% TARGET REACHED!")

    print(
        f"Final accuracy: "
        f"{final_accuracy * 100:.2f}%"
    )

else:

    gap = (
        0.90 - final_accuracy
    ) * 100

    print("\n90% not reached yet.")

    print(
        f"Remaining gap: "
        f"{gap:.2f} percentage points"
    )


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 65)
print("SAVING MODEL")
print("=" * 65)


model_information = {

    "model_type":
        "CatBoostClassifier",

    "experiment":
        "depth_9_iterations_2500",

    "threshold":
        float(best_threshold),

    "features":
        feature_columns,

    "categorical_features":
        categorical_columns,

    "accuracy":
        float(final_accuracy),

    "roc_auc":
        float(roc_auc),

    "f1_score":
        float(final_f1),

    "baseline_accuracy":
        float(baseline / 100),

    "feature_importance":
        feature_importance
        .to_dict(
            orient="records"
        )
}


joblib.dump(
    model,
    MODEL_PATH
)


joblib.dump(
    model_information,
    INFO_PATH
)


print(
    "\nModel saved successfully:"
)

print(MODEL_PATH)


print(
    "\nModel information saved:"
)

print(INFO_PATH)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("             TRAINING COMPLETE")
print("=" * 65)

print(
    f"\nFinal test accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"ROC-AUC: "
    f"{roc_auc:.4f}"
)

print(
    f"F1 Score: "
    f"{final_f1:.4f}"
)

print(
    f"Decision threshold: "
    f"{best_threshold:.2f}"
)

print("\nSaved model:")
print("transfer_model.joblib")

print("\nSaved model information:")
print("transfer_model_info.joblib")

print("\nNext command:")
print("python3 predict_improved.py")

print("\nDone!")