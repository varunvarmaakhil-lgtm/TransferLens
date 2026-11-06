import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import joblib


# ============================================================
# TRANSFERLENS - ML MODEL TRAINING
# Season: 2025-26
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "ml_dataset_2025_26.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "transfer_prediction_model.pkl"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TRANSFERLENS - ML MODEL TRAINING")
    print("Season: 2025-26")
    print("=" * 60)


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading ML dataset...")

    if not DATA_FILE.exists():
        print("\nERROR: ML dataset not found.")
        print(DATA_FILE)
        return

    df = pd.read_csv(DATA_FILE)

    print(f"Players loaded: {len(df)}")


    # --------------------------------------------------------
    # Check target
    # --------------------------------------------------------

    target = "transferred"

    if target not in df.columns:
        print(f"\nERROR: Target column '{target}' not found.")
        return


    # --------------------------------------------------------
    # Remove columns that should NOT be used for prediction
    # --------------------------------------------------------

    columns_to_drop = [
        "player_name",
        "season",
        "transferred"
    ]

    # These fields describe past transfer history.
    # We remove them to reduce data leakage.
    leakage_columns = [
        "transfer_count",
        "total_transfer_fee"
    ]

    columns_to_drop.extend(leakage_columns)

    columns_to_drop = [
        column for column in columns_to_drop
        if column in df.columns
    ]

    X = df.drop(columns=columns_to_drop)
    y = df[target]


    # --------------------------------------------------------
    # Display target distribution
    # --------------------------------------------------------

    print("\nTarget distribution:")

    print(
        y.value_counts()
        .rename({
            0: "Not Transferred",
            1: "Transferred"
        })
    )


    # --------------------------------------------------------
    # Identify categorical and numerical columns
    # --------------------------------------------------------

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()


    print("\nCategorical features:")

    for column in categorical_columns:
        print(f" - {column}")


    print("\nNumerical features:")

    for column in numerical_columns:
        print(f" - {column}")


    # --------------------------------------------------------
    # Train / Test split
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )


    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )


    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )


    # --------------------------------------------------------
    # Create ML model
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )


    # --------------------------------------------------------
    # Complete pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining Random Forest model...")

    pipeline.fit(
        X_train,
        y_train
    )

    print("Training complete.")


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nEvaluating model...")

    y_pred = pipeline.predict(X_test)

    y_probability = pipeline.predict_proba(X_test)[:, 1]


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_probability
    )


    print("\n" + "=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Not Transferred",
                "Transferred"
            ]
        )
    )


    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_FILE
    )


    print("\n" + "=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(f"\nSaved to:")
    print(MODEL_FILE)


    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print("\nTraining completed successfully.")
    print("Next step: build the player prediction system.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()