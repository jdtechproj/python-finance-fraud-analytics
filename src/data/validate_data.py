import pandas as pd


REQUIRED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "transaction_timestamp",
    "amount",
    "merchant_category",
    "channel",
    "country",
    "home_country",
    "distance_from_home_km",
    "account_age_days",
    "transactions_last_1h",
    "transactions_last_24h",
    "device_change",
    "is_fraud"
]


def validate_transactions(df):
    errors = []

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        errors.append(
            f"Missing columns: {missing_columns}"
        )

    if df["transaction_id"].duplicated().any():
        errors.append("Duplicate transaction IDs found")

    if df["amount"].isna().any():
        errors.append("Missing transaction amounts found")

    if (df["amount"] <= 0).any():
        errors.append("Non-positive transaction amounts found")

    if not df["is_fraud"].isin([0, 1]).all():
        errors.append("Invalid fraud labels found")

    return errors


if __name__ == "__main__":
    df = pd.read_csv("data/raw/transactions.csv")

    errors = validate_transactions(df)

    if errors:
        for error in errors:
            print(error)
    else:
        print("Data validation passed")