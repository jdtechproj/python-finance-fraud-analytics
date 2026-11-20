
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    classification_report
)


def evaluate_predictions(file_path):

    df = pd.read_csv(file_path)

    actual = df["actual_fraud"]
    predicted = df["predicted_fraud"]
    probability = df["fraud_probability"]

    print("\nClassification Report")

    print(
        classification_report(
            actual,
            predicted,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(actual, predicted)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"]
    )

    display.plot()

    plt.title("Fraud Detection Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        "reports/figures/confusion_matrix.png",
        dpi=150
    )

    plt.show()

    # Precision-recall curve
    precision, recall, thresholds = (
        precision_recall_curve(actual, probability)
    )

    plt.figure(figsize=(8, 5))

    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Fraud Detection Precision-Recall Curve")

    plt.tight_layout()

    plt.savefig(
        "reports/figures/precision_recall_curve.png",
        dpi=150
    )

    plt.show()


if __name__ == "__main__":

    evaluate_predictions(
        "reports/test_predictions.csv"
    )
