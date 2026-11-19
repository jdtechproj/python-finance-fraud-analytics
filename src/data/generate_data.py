
import numpy as np
import pandas as pd
from pathlib import Path


def generate_transactions(
    number_of_transactions=30000,
    number_of_customers=3000,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    customers = [
        f"CUST{i:05d}"
        for i in range(1, number_of_customers + 1)
    ]

    start_date = pd.Timestamp("2026-01-01")

    timestamps = (
        start_date
        + pd.to_timedelta(
            rng.integers(0, 90 * 24 * 60, number_of_transactions),
            unit="m"
        )
    )

    customer_ids = rng.choice(
        customers,
        size=number_of_transactions
    )

    channels = rng.choice(
        ["Mobile", "Web", "ATM", "POS"],
        size=number_of_transactions,
        p=[0.35, 0.25, 0.10, 0.30]
    )

    merchants = rng.choice(
        [
            "Grocery",
            "Travel",
            "Electronics",
            "Restaurant",
            "Gas",
            "Retail",
            "Online"
        ],
        size=number_of_transactions
    )

    home_countries = np.full(
        number_of_transactions, "USA"
    )

    countries = rng.choice(
        ["USA", "USA", "USA", "UK", "Canada", "Germany"],
        size=number_of_transactions
    )

    amounts = rng.lognormal(
        mean=3.5,
        sigma=1.1,
        size=number_of_transactions
    ).round(2)

    hours = pd.Series(timestamps).dt.hour.to_numpy()

    distance = rng.exponential(
        scale=30,
        size=number_of_transactions
    ).round(2)

    international = (countries != home_countries).astype(int)

    device_change = rng.binomial(
        1, 0.08, number_of_transactions
    )

    account_age = rng.integers(
        10, 4000, number_of_transactions
    )

    velocity_1h = rng.poisson(
        1.2, number_of_transactions
    )

    velocity_24h = (
        velocity_1h
        + rng.poisson(4, number_of_transactions)
    )

    # Create fraud risk signals
    risk_score = (
        -4.0
        + 0.8 * international
        + 1.1 * device_change
        + 0.7 * (amounts > 1000)
        + 0.8 * (velocity_1h >= 4)
        + 0.5 * (velocity_24h >= 10)
        + 0.5 * (distance > 300)
        + 0.5 * (hours <= 5)
    )

    fraud_probability = 1 / (1 + np.exp(-risk_score))

    is_fraud = rng.binomial(
        1, fraud_probability
    )

    df = pd.DataFrame({
        "transaction_id": [
            f"TXN{i:07d}"
            for i in range(1, number_of_transactions + 1)
        ],
        "customer_id": customer_ids,
        "transaction_timestamp": timestamps,
        "amount": amounts,
        "merchant_category": merchants,
        "channel": channels,
        "country": countries,
        "home_country": home_countries,
        "distance_from_home_km": distance,
        "account_age_days": account_age,
        "transactions_last_1h": velocity_1h,
        "transactions_last_24h": velocity_24h,
        "device_change": device_change,
        "is_fraud": is_fraud
    })

    return df.sort_values(
        "transaction_timestamp"
    ).reset_index(drop=True)


if __name__ == "__main__":
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    transactions = generate_transactions()

    output_file = output_dir / "transactions.csv"

    transactions.to_csv(output_file, index=False)

    print(f"Generated {len(transactions):,} transactions")
    print(f"Fraudulent transactions: {transactions['is_fraud'].sum():,}")
    print(f"Fraud rate: {transactions['is_fraud'].mean():.2%}")
    print(f"Saved to: {output_file}")
