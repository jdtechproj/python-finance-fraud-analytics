# python-finance-fraud-analytics
Python finance project


# Financial Risk & Fraud Detection Platform

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Testing](https://img.shields.io/badge/Testing-Pytest-green)
![Analytics](https://img.shields.io/badge/Analytics-Financial%20Risk-purple)

## 1. Project Overview

An end-to-end Python-based financial analytics and fraud
detection platform designed to identify suspicious banking
transactions, evaluate financial risk, and generate
actionable analytics.

The solution demonstrates Python engineering, data
processing, feature engineering, machine learning,
anomaly detection, and financial analytics.

All transaction data is synthetic and generated for
demonstration purposes.

## 2. Business Objectives

- Identify potentially fraudulent transactions.
- Detect unusual transaction behavior.
- Calculate financial transaction risk scores.
- Evaluate machine learning performance.
- Analyze potential financial exposure.
- Provide interactive fraud monitoring analytics.

## 3. Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- SQLAlchemy
- SQLite
- Pytest
- Git / GitHub

## 4. Architecture

```mermaid
flowchart TD
    A[Synthetic Banking Data] --> B[Data Ingestion]
    B --> C[Data Validation]
    C --> D[Feature Engineering]
    D --> E[Rule-Based Detection]
    D --> F[Random Forest]
    D --> G[Isolation Forest]
    E --> H[Risk Analytics]
    F --> H
    G --> H
    H --> I[SQLite Database]
    H --> J[Streamlit Dashboard]
    H --> K[Model Evaluation]
```

## 5. Project Structure

```text
data/
models/
notebooks/
reports/
src/
    data/
    features/
    models/
    analytics/
    database/
tests/
docs/
app.py
requirements.txt
README.md
```

## 6. Machine Learning Approach

### Supervised Learning

Random Forest classification using historical
transaction labels.

### Unsupervised Learning

Isolation Forest for identifying anomalous
transaction behavior.

### Rule-Based Detection

Business rules for identifying suspicious
transaction patterns.

## 7. Model Evaluation

Evaluation includes:

- Precision
- Recall
- F1 Score
- ROC-AUC
- Average Precision
- Confusion Matrix
- Precision-Recall Curve

See [Model Results](docs/model-results.md).

## 8. Application Screenshots

### Financial Risk Dashboard

![Dashboard](docs/images/09-dashboard-overview.png)

### Fraud Risk Distribution

![Risk Distribution](docs/images/10-risk-distribution.png)

### Model Evaluation

![Confusion Matrix](docs/images/13-confusion-matrix.png)

### Feature Importance

![Feature Importance](docs/images/15-feature-importance.png)

## 9. How to Run the Project

### Clone repository

```bash
git clone https://github.com/jdtechproj/python-finance-fraud-analytics.git

cd python-finance-fraud-analytics
```

### Create virtual environment

```bash
python -m venv .venv
```

### Activate environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate transaction data

```bash
python -m src.data.generate_data
```

### Validate data

```bash
python -m src.data.validate_data
```

### Train fraud detection model

```bash
python -m src.models.train_model
```

### Run anomaly detection

```bash
python -m src.models.anomaly_detection
```

### Generate evaluation results

```bash
python -m src.models.evaluate_model
```

### Launch dashboard

```bash
streamlit run app.py
```

## 10. Testing

Run automated tests:

```bash
pytest -v
```

## 11. Data Disclaimer

This project uses synthetic banking transactions.
It does not contain real customer or financial
institution information.

Model outputs are illustrative and are not intended
for actual financial fraud decisions.

## 12. Future Enhancements

- Real-time transaction scoring API.
- Advanced fraud detection algorithms.
- SHAP-based model explainability.
- Model drift monitoring.
- CI/CD automation.
- Cloud-based deployment.
- Integration with Snowflake.

## 13. Author

Data Engineering and Analytics Portfolio

GitHub: jdtechproj
