import pandas as pd

from src.database.database import (
    load_predictions,
    query_high_risk_transactions
)


if __name__ == "__main__":

    predictions = pd.read_csv(
        "reports/test_predictions.csv"
    )

    load_predictions(predictions)

    high_risk = query_high_risk_transactions()

    print(high_risk.head(20))