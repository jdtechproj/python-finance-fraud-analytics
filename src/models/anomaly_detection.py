
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features.features import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    engineer_features,
    get_model_features
)


def detect_anomalies(df):

    df = engineer_features(df)

    X = get_model_features(df)

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

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("detector", IsolationForest(
            n_estimators=200,
            contamination=0.02,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X)

    anomaly_labels = pipeline.predict(X)

    results = df.copy()

    results["anomaly_flag"] = (
        anomaly_labels == -1
    ).astype(int)

    results["anomaly_score"] = (
        -pipeline.decision_function(X)
    )

    return results


if __name__ == "__main__":

    df = pd.read_csv("data/raw/transactions.csv")

    results = detect_anomalies(df)

    print(
        "Anomalies identified:",
        results["anomaly_flag"].sum()
    )

    results.to_csv(
        "reports/anomaly_results.csv",
        index=False
    )
