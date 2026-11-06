import os
import glob
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("=" * 60)
print("TRANSFERLENS — BEST MODEL EVALUATION")
print("Season: 2025-26")
print("=" * 60)

# ---------------------------------------------------------
# 1. LOAD MODEL
# ---------------------------------------------------------

MODEL_PATH = "models/best_tuned_transfer_model.pkl"

print("\nLoading best model...")

model = joblib.load(MODEL_PATH)

print("Best model loaded successfully.")
print("Model:", type(model).__name__)

# ---------------------------------------------------------
# 2. FIND DATASET
# ---------------------------------------------------------

print("\nLoading dataset...")

csv_files = glob.glob("data/processed/*.csv") + glob.glob("data/*.csv")

# Ignore previously generated comparison files
csv_files = [
    f for f in csv_files
    if "comparison" not in f.lower()
]

if not csv_files:
    raise FileNotFoundError(
        "No CSV dataset found inside data/ or data/processed/"
    )

# Prefer a file containing player data
dataset_path = None

for f in csv_files:
    try:
        temp = pd.read_csv(f)
        if "transferred" in temp.columns:
            dataset_path = f
            break
    except Exception:
        continue

if dataset_path is None:
    raise ValueError(
        "Could not find a dataset containing the 'transferred' column."
    )

df = pd.read_csv(dataset_path)

print("Dataset:", dataset_path)
print("Players loaded:", len(df))

# ---------------------------------------------------------
# 3. PREPARE DATA
# ---------------------------------------------------------

print("\nPreparing evaluation data...")

TARGET = "transferred"

# Remove leakage / identifier columns
DROP_COLUMNS = [
    "transferred",
    "transfer_count",
    "total_transfer_fee",
    "player_name"
]

existing_drop = [
    col for col in DROP_COLUMNS
    if col in df.columns
]

X = df.drop(columns=existing_drop)
y = df[TARGET]

# Make sure target is numeric
y = pd.to_numeric(y, errors="coerce")

valid_rows = y.notna()

X = X.loc[valid_rows]
y = y.loc[valid_rows].astype(int)

print("Features:", len(X.columns))
print("Target distribution:")
print(y.value_counts())

# ---------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------------

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training players:", len(X_train))
print("Testing players:", len(X_test))

# ---------------------------------------------------------
# 5. PREDICTIONS
# ---------------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)[:, 1]
else:
    y_prob = None

# ---------------------------------------------------------
# 6. METRICS
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

if y_prob is not None:
    roc_auc = roc_auc_score(y_test, y_prob)
else:
    roc_auc = 0.0

print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# ---------------------------------------------------------
# 7. CONFUSION MATRIX
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)

print("\n                 Predicted")
print("                 0      1")
print(f"Actual 0      {cm[0][0]:5d}  {cm[0][1]:5d}")
print(f"Actual 1      {cm[1][0]:5d}  {cm[1][1]:5d}")

# ---------------------------------------------------------
# 8. CLASSIFICATION REPORT
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Transferred",
            "Transferred"
        ],
        zero_division=0
    )
)

# ---------------------------------------------------------
# 9. FEATURE IMPORTANCE
# ---------------------------------------------------------

print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

# Find the actual classifier inside a Pipeline
classifier = model

if hasattr(model, "named_steps"):
    for name, step in reversed(model.named_steps.items()):
        if hasattr(step, "feature_importances_"):
            classifier = step
            break

if hasattr(classifier, "feature_importances_"):

    importances = classifier.feature_importances_

    # Try to obtain transformed feature names
    feature_names = None

    if hasattr(model, "named_steps"):

        for name, step in model.named_steps.items():

            if hasattr(step, "get_feature_names_out"):

                try:
                    feature_names = step.get_feature_names_out()
                except Exception:
                    pass

    if feature_names is None:
        feature_names = np.array(X.columns)

    # Make lengths match
    if len(feature_names) != len(importances):
        feature_names = np.array(
            [f"Feature_{i}" for i in range(len(importances))]
        )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    print("\nTop 15 most important features:\n")

    print(
        importance_df.head(15).to_string(index=False)
    )

    # Save feature importance
    os.makedirs("data/processed", exist_ok=True)

    importance_df.to_csv(
        "data/processed/feature_importance_2025_26.csv",
        index=False
    )

    print(
        "\nFeature importance saved to:"
        "\ndata/processed/feature_importance_2025_26.csv"
    )

else:
    print("\nFeature importance is not available for this model.")

# ---------------------------------------------------------
# 10. SAVE EVALUATION RESULTS
# ---------------------------------------------------------

results = pd.DataFrame([{
    "model": "Gradient Boosting",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc
}])

os.makedirs("data/processed", exist_ok=True)

results.to_csv(
    "data/processed/best_model_evaluation_2025_26.csv",
    index=False
)

print("\nEvaluation results saved to:")
print("data/processed/best_model_evaluation_2025_26.csv")

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETE")
print("=" * 60)