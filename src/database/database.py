
import pandas as pd
from sqlalchemy import create_engine, text


def get_engine():

    return create_engine(
        "sqlite:///data/finance_analytics.db"
    )


def load_predictions(df):

    engine = get_engine()

    df.to_sql(
        "fraud_predictions",
        engine,
        if_exists="replace",
        index=False
    )

    print("Predictions loaded successfully")


def query_high_risk_transactions():

    engine = get_engine()

    sql = text("""
        SELECT
            transaction_id,
            customer_id,
            amount,
            fraud_probability,
            predicted_fraud
        FROM fraud_predictions
        WHERE fraud_probability >= :threshold
        ORDER BY fraud_probability DESC
    """)

    with engine.connect() as connection:

        results = pd.read_sql(
            sql,
            connection,
            params={"threshold": 0.60}
        )

    return results
