import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


SEASON = "2025-26"

DATA_PATH = "data/processed/ml_dataset_2025_26.csv"
MODEL_PATH = "models/best_clean_transfer_model.pkl"
RESULT_PATH = "data/processed/clean_model_comparison_2025_26.csv"


print("=" * 60)
print("TRANSFERLENS - CLEAN MODEL COMPARISON")
print(f"Season: {SEASON}")
print("=" * 60)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Players loaded: {len(df)}")


# ---------------------------------------------------------
# REMOVE DATA LEAKAGE
# ---------------------------------------------------------

print("\nRemoving leakage features...")

leakage_features = [
    "transferred",
    "transfer_count",
    "total_transfer_fee"
]

for col in leakage_features:
    if col in df.columns:
        df = df.drop(columns=[col])

print("Removed:")
for col in leakage_features:
    print(f"- {col}")


# ---------------------------------------------------------
# TARGET
# ---------------------------------------------------------

# Recreate target from original transfer information.
# Players with transfer records have target = 1.

original_df = pd.read_csv(DATA_PATH)

target = original_df["transferred"].astype(int)

X = df.copy()
y = target


# Remove player name because it should not be used
# as a predictive feature.

if "player_name" in X.columns:
    X = X.drop(columns=["player_name"])


# max_market_value may contain information from after the
# prediction point, so remove it for a cleaner experiment.

if "max_market_value" in X.columns:
    X = X.drop(columns=["max_market_value"])


# Remove season because every row belongs to the same season.

if "season" in X.columns:
    X = X.drop(columns=["season"])


# ---------------------------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------------------------

X = X.copy()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


for col in numerical_features:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_features:
    X[col] = X[col].fillna("Unknown")


print("\nCategorical features:")
for col in categorical_features:
    print(f"- {col}")

print("\nNumerical features:")
for col in numerical_features:
    print(f"- {col}")


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training players: {len(X_train)}")
print(f"Testing players: {len(X_test)}")


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ---------------------------------------------------------
# TRAIN AND EVALUATE
# ---------------------------------------------------------

results = []

trained_models = {}


for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"TRAINING: {name}")
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    print("Training complete.")

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[name] = pipeline


# ---------------------------------------------------------
# COMPARISON
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)


print("\n")
print("=" * 75)
print("CLEAN MODEL COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ---------------------------------------------------------
# SELECT BEST MODEL
# ---------------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Model: {best_model_name}")
print(
    f"ROC-AUC: "
    f"{results_df.iloc[0]['ROC-AUC']:.4f}"
)


# ---------------------------------------------------------
# SAVE BEST MODEL
# ---------------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nBest model saved to:")
print(MODEL_PATH)


# ---------------------------------------------------------
# SAVE COMPARISON
# ---------------------------------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)

results_df.to_csv(
    RESULT_PATH,
    index=False
)

print("\nComparison results saved to:")
print(RESULT_PATH)


print("\n" + "=" * 60)
print("CLEAN MODEL COMPARISON COMPLETE")
print("=" * 60)