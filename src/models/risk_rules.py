import pandas as pd


def calculate_rule_risk(df):
    df = df.copy()

    df["rule_score"] = 0

    df.loc[
        df["amount"] > 3000,
        "rule_score"
    ] += 30

    df.loc[
        df["is_international"] == 1,
        "rule_score"
    ] += 20

    df.loc[
        df["device_change"] == 1,
        "rule_score"
    ] += 20

    df.loc[
        df["transactions_last_1h"] >= 4,
        "rule_score"
    ] += 15

    df.loc[
        df["distance_from_home_km"] > 300,
        "rule_score"
    ] += 15

    df["rule_risk_category"] = pd.cut(
        df["rule_score"],
        bins=[-1, 29, 59, 100],
        labels=["Low", "Medium", "High"]
    )

    return df