import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# TRANSFERLENS - MODEL TUNING
# ============================================================

print("=" * 60)
print("TRANSFERLENS - MODEL TUNING")
print("Season: 2025-26")
print("=" * 60)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

DATA_PATH = "data/processed/ml_dataset_2025_26.csv"

df = pd.read_csv(DATA_PATH)

print(f"Players loaded: {len(df)}")


# ============================================================
# EXTRACT TARGET FIRST
# ============================================================

TARGET = "transferred"

if TARGET not in df.columns:
    raise ValueError(
        "Target column 'transferred' not found in dataset."
    )

y = df[TARGET]


# ============================================================
# CREATE FEATURES
# ============================================================

X = df.drop(columns=[TARGET])


# ============================================================
# REMOVE LEAKAGE FEATURES
# ============================================================

print("\nRemoving leakage features...")

LEAKAGE_FEATURES = [
    "transfer_count",
    "total_transfer_fee"
]

for feature in LEAKAGE_FEATURES:

    if feature in X.columns:
        X = X.drop(columns=[feature])
        print(f"- {feature}")

print("- transferred (target)")


# ============================================================
# REMOVE PLAYER NAME
# ============================================================

if "player_name" in X.columns:
    X = X.drop(columns=["player_name"])
    print("- player_name (identifier)")


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()


print("\nCategorical features:")

for feature in categorical_features:
    print(f"- {feature}")


print("\nNumerical features:")

for feature in numerical_features:
    print(f"- {feature}")


# ============================================================
# PREPROCESSING
# ============================================================

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
    ("numerical", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

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


# ============================================================
# MODELS AND PARAMETER GRIDS
# ============================================================

models = {

    "Logistic Regression": (
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),
        {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__class_weight": [None, "balanced"],
            "model__solver": ["liblinear", "lbfgs"]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),
        {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__class_weight": [None, "balanced"]
        }
    ),

    "Extra Trees": (
        ExtraTreesClassifier(
            random_state=42,
            n_jobs=-1
        ),
        {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__class_weight": [None, "balanced"]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(
            random_state=42
        ),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 5]
        }
    )
}


# ============================================================
# MODEL TUNING
# ============================================================

results = []

best_model = None
best_model_name = None
best_roc_auc = -1


for model_name, (model, param_grid) in models.items():

    print("\n" + "=" * 60)
    print(f"TUNING: {model_name}")
    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("Searching for best parameters...")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(search.best_params_)

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    best_pipeline = search.best_estimator_

    y_pred = best_pipeline.predict(X_test)

    y_probability = best_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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


    print("\nTUNED MODEL RESULTS")
    print("-" * 40)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")


    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if roc_auc > best_roc_auc:

        best_roc_auc = roc_auc
        best_model = best_pipeline
        best_model_name = model_name


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("TUNED MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("BEST TUNED MODEL")
print("=" * 60)

print(f"Model: {best_model_name}")
print(f"ROC-AUC: {best_roc_auc:.4f}")


# ============================================================
# SAVE BEST MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/best_tuned_transfer_model.pkl"

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nBest model saved to:")
print(MODEL_PATH)


# ============================================================
# SAVE COMPARISON RESULTS
# ============================================================

os.makedirs("data/processed", exist_ok=True)

RESULT_PATH = "data/processed/tuned_model_comparison_2025_26.csv"

results_df.to_csv(
    RESULT_PATH,
    index=False
)

print("\nTuning results saved to:")
print(RESULT_PATH)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("MODEL TUNING COMPLETE")
print("=" * 60)