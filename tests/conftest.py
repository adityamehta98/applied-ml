"""A small synthetic Telco frame with the real column names, so tests never download data."""
import numpy as np
import pandas as pd
import pytest

RAW_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]


def make_churn_df(n=400, seed=0):
    """Build a raw-looking Telco frame with a real, learnable churn signal.

    Churn depends on Contract and tenure the way it does in the real data: month-to-month
    customers with a short tenure churn much more. New customers (tenure 0) get a blank
    TotalCharges string, exactly the quirk clean() has to handle.
    """
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(20, 120, n).round(2)

    base = np.where(contract == "Month-to-month", 0.50, 0.12)
    logit = base + 0.8 * (tenure < 6) - 0.005 * tenure
    churn_prob = np.clip(logit, 0.02, 0.9)
    churn = rng.random(n) < churn_prob

    total = (tenure * monthly).round(2).astype(object)
    total[tenure == 0] = " "  # blank string for brand-new customers, like the real export

    df = pd.DataFrame({
        "customerID": [f"{i:04d}-ABCD" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": tenure,
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": contract,
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)",
             "Credit card (automatic)"], n),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": np.where(churn, "Yes", "No"),
    })
    return df[RAW_COLUMNS]


@pytest.fixture()
def raw_df():
    return make_churn_df()
