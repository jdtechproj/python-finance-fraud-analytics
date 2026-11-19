
import pandas as pd


NUMERIC_FEATURES = [
    "amount",
    "distance_from_home_km",
    "account_age_days",
    "transactions_last_1h",
    "transactions_last_24h",
    "transaction_hour",
    "transaction_day",
    "is_international",
    "high_value_transaction",
    "new_account",
    "device_change"
]

CATEGORICAL_FEATURES = [
    "merchant_category",
    "channel",
    "country"
]

TARGET = "is_fraud"


def engineer_features(df):
    df = df.copy()

    df["transaction_timestamp"] = pd.to_datetime(
        df["transaction_timestamp"]
    )

    df["transaction_hour"] = (
        df["transaction_timestamp"].dt.hour
    )

    df["transaction_day"] = (
        df["transaction_timestamp"].dt.dayofweek
    )

    df["is_international"] = (
        df["country"] != df["home_country"]
    ).astype(int)

    df["high_value_transaction"] = (
        df["amount"] > 1000
    ).astype(int)

    df["new_account"] = (
        df["account_age_days"] < 90
    ).astype(int)

    return df


def get_model_features(df):
    return df[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ]
