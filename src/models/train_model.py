
import json
from pathlib import Path

import joblib
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features.features import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    engineer_features,
    get_model_features
)


DATA_PATH = Path("data/raw/transactions.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def select_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    # Prefer a threshold with at least 30% precision.
    eligible = [
        i for i, p in enumerate(precision[:-1])
        if p >= 0.30
    ]

    if eligible:
        best_index = max(
            eligible,
            key=lambda i: recall[i]
        )
        return float(thresholds[best_index])

    return 0.50


def main():

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["transaction_timestamp"]
    )

    df = engineer_features(df)

    # Chronological split to avoid random temporal leakage.
    df = df.sort_values(
        "transaction_timestamp"
    ).reset_index(drop=True)

    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    validation_df = df.iloc[train_end:validation_end]
    test_df = df.iloc[validation_end:]

    X_train = get_model_features(train_df)
    y_train = train_df[TARGET]

    X_validation = get_model_features(validation_df)
    y_validation = validation_df[TARGET]

    X_test = get_model_features(test_df)
    y_test = test_df[TARGET]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer,
         CATEGORICAL_FEATURES)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("Training model...")

    model.fit(X_train, y_train)

    feature_names = (
        model.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importance = (
        model.named_steps["classifier"]
        .feature_importances_
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(
        "importance",
        ascending=False
    )

    importance_df.to_csv(
        "reports/feature_importance.csv",
        index=False
    )

    top_features = importance_df.head(10)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.gca().invert_yaxis()

    plt.title("Top 10 Fraud Detection Features")
    plt.xlabel("Feature Importance")

    plt.tight_layout()

    plt.savefig(
        "reports/figures/feature_importance.png",
        dpi=150
    )

    plt.show()

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    threshold = select_threshold(
        y_validation,
        validation_probabilities
    )

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    metrics = {
        "threshold": threshold,
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "average_precision": average_precision_score(
            y_test, test_probabilities
        ),
        "roc_auc": roc_auc_score(
            y_test, test_probabilities
        )
    }

    joblib.dump(
        model,
        MODEL_DIR / "fraud_detection_model.joblib"
    )

    with open(
        MODEL_DIR / "threshold.json", "w"
    ) as file:
        json.dump({"threshold": threshold}, file, indent=4)

    with open(
        REPORT_DIR / "model_metrics.json", "w"
    ) as file:
        json.dump(metrics, file, indent=4)

    predictions_df = test_df[
        ["transaction_id", "customer_id",
         "transaction_timestamp", "amount", "channel"]
    ].copy()

    predictions_df["actual_fraud"] = y_test.values
    predictions_df["fraud_probability"] = test_probabilities
    predictions_df["predicted_fraud"] = predictions

    predictions_df.to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False
    )

    print("\nModel Evaluation")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()
