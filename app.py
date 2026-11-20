
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


st.set_page_config(
    page_title="Financial Fraud Analytics",
    page_icon="💳",
    layout="wide"
)

st.title("Financial Risk & Fraud Detection")
st.caption(
    "Synthetic banking transaction analytics and "
    "machine learning demonstration"
)

prediction_file = Path("reports/test_predictions.csv")

if not prediction_file.exists():
    st.error(
        "Prediction results not found. "
        "Run the training pipeline first."
    )
    st.stop()

df = pd.read_csv(prediction_file)

df["risk_category"] = pd.cut(
    df["fraud_probability"],
    bins=[-0.001, 0.30, 0.60, 0.80, 1.0],
    labels=["Low", "Medium", "High", "Critical"]
)

total_transactions = len(df)

flagged = df[
    df["predicted_fraud"] == 1
]

flagged_count = len(flagged)

flagged_value = flagged["amount"].sum()

high_risk = df[
    df["fraud_probability"] >= 0.60
]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Transactions Analyzed",
    f"{total_transactions:,}"
)

col2.metric(
    "Flagged Transactions",
    f"{flagged_count:,}"
)

col3.metric(
    "Flagged Transaction Value",
    f"${flagged_value:,.2f}"
)

col4.metric(
    "High/Critical Risk",
    f"{len(high_risk):,}"
)

st.divider()

left, right = st.columns(2)

with left:

    risk_distribution = (
        df["risk_category"]
        .value_counts()
        .reset_index()
    )

    risk_distribution.columns = [
        "Risk Category", "Transactions"
    ]

    fig = px.pie(
        risk_distribution,
        names="Risk Category",
        values="Transactions",
        title="Transaction Risk Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:

    fig = px.histogram(
        df,
        x="fraud_probability",
        nbins=30,
        title="Fraud Probability Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

st.subheader("Transactions Requiring Review")

review_transactions = (
    df[df["fraud_probability"] >= 0.60]
    .sort_values(
        "fraud_probability",
        ascending=False
    )
)

st.dataframe(
    review_transactions,
    use_container_width=True
)

st.download_button(
    "Download Review Transactions",
    data=review_transactions.to_csv(index=False),
    file_name="high_risk_transactions.csv",
    mime="text/csv"
)
