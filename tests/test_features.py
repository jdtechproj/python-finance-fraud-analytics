import pandas as pd

from src.features.features import engineer_features


def test_international_transaction():

    df = pd.DataFrame({
        "transaction_timestamp": [
            "2026-01-01 10:00:00"
        ],
        "country": ["UK"],
        "home_country": ["USA"],
        "amount": [500],
        "account_age_days": [100],
        "device_change": [0]
    })

    result = engineer_features(df)

    assert result.iloc[0]["is_international"] == 1


def test_high_value_transaction():

    df = pd.DataFrame({
        "transaction_timestamp": [
            "2026-01-01 10:00:00"
        ],
        "country": ["USA"],
        "home_country": ["USA"],
        "amount": [2500],
        "account_age_days": [100],
        "device_change": [0]
    })

    result = engineer_features(df)

    assert result.iloc[0]["high_value_transaction"] == 1