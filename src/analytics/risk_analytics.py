
import pandas as pd


def classify_risk(probability):

    if probability >= 0.80:
        return "Critical"

    elif probability >= 0.60:
        return "High"

    elif probability >= 0.30:
        return "Medium"

    return "Low"


def generate_risk_analytics(predictions):

    df = predictions.copy()

    df["risk_category"] = (
        df["fraud_probability"]
        .apply(classify_risk)
    )

    summary = (
        df.groupby("risk_category")
        .agg(
            transaction_count=("transaction_id", "count"),
            total_transaction_value=("amount", "sum"),
            average_probability=("fraud_probability", "mean")
        )
        .reset_index()
    )

    summary["total_transaction_value"] = (
        summary["total_transaction_value"].round(2)
    )

    return df, summary


def generate_channel_analytics(predictions):

    return (
        predictions.groupby("channel")
        .agg(
            transaction_count=("transaction_id", "count"),
            transaction_value=("amount", "sum"),
            average_fraud_probability=(
                "fraud_probability", "mean"
            )
        )
        .reset_index()
    )


if __name__ == "__main__":

    predictions = pd.read_csv(
        "reports/test_predictions.csv"
    )

    # For this demonstration, the trained model
    # has generated fraud probabilities.
    risk_results, summary = generate_risk_analytics(
        predictions
    )

    risk_results.to_csv(
        "reports/risk_assessment.csv",
        index=False
    )

    summary.to_csv(
        "reports/risk_summary.csv",
        index=False
    )

    print(summary)
