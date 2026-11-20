import pandas as pd

from src.models.risk_rules import calculate_rule_risk


def test_high_risk_transaction():

    df = pd.DataFrame({
        "amount": [5000],
        "is_international": [1],
        "device_change": [1],
        "transactions_last_1h": [5],
        "distance_from_home_km": [500]
    })

    result = calculate_rule_risk(df)

    assert result.iloc[0]["rule_score"] == 100

    assert (
        result.iloc[0]["rule_risk_category"]
        == "High"
    )