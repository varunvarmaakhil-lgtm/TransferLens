import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import joblib


# ============================================================
# TRANSFERLENS - MODEL COMPARISON
# ============================================================

print("=" * 60)
print("TRANSFERLENS - MODEL COMPARISON")
print("Season: 2025-26")
print("=" * 60)


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

DATA_PATH = "data/processed/ml_dataset_2025_26.csv"

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Players loaded: {len(df)}")


# ------------------------------------------------------------
# 2. DEFINE TARGET
# ------------------------------------------------------------

TARGET = "transferred"

# Remove columns that should NOT be used as model inputs
DROP_COLUMNS = [
    "transferred",
    "player_name",
    "season"
]

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]


# ------------------------------------------------------------
# 3. IDENTIFY FEATURE TYPES
# ------------------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical features:")
for col in categorical_features:
    print(f" - {col}")

print("\nNumerical features:")
for col in numerical_features:
    print(f" - {col}")


# ------------------------------------------------------------
# 4. PREPROCESSING
# ------------------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 6. DEFINE MODELS
# ------------------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ------------------------------------------------------------
# 7. TRAIN AND EVALUATE
# ------------------------------------------------------------

results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"TRAINING: {name}")
    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    print("Training complete.")

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probabilities
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
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


# ------------------------------------------------------------
# 8. COMPARISON TABLE
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)

print("\n")
print("=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format
        }
    )
)


# ------------------------------------------------------------
# 9. SELECT BEST MODEL
# ------------------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Model: {best_model_name}")
print(
    f"ROC-AUC: "
    f"{results_df.iloc[0]['ROC-AUC']:.4f}"
)


# ------------------------------------------------------------
# 10. SAVE BEST MODEL
# ------------------------------------------------------------

MODEL_PATH = "models/best_transfer_prediction_model.pkl"

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nBest model saved to:")
print(MODEL_PATH)


# ------------------------------------------------------------
# 11. SAVE COMPARISON RESULTS
# ------------------------------------------------------------

RESULTS_PATH = "data/processed/model_comparison_2025_26.csv"

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print("\nComparison results saved to:")
print(RESULTS_PATH)


print("\n" + "=" * 60)
print("MODEL COMPARISON COMPLETE")
print("=" * 60)