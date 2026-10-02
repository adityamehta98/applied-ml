"""A quick, honest first look at the churn data: shape, balance, blanks, leakage suspects."""
from src.data import TARGET, clean, feature_columns

LEAKAGE_SUSPECTS = ("customerID",)  # unique per row, so it can only memorize, never generalize


def summarize(df_clean):
    """Key facts a first look should surface, as a dict so a test can check them."""
    numeric, categorical = feature_columns(df_clean)
    return {
        "rows": len(df_clean),
        "churn_rate": round(float(df_clean[TARGET].mean()), 3),
        "n_numeric": len(numeric),
        "n_categorical": len(categorical),
        "missing_total_charges": int(df_clean["TotalCharges"].isna().sum()),
    }


def main(path="data/telco.csv"):
    import pandas as pd

    raw = pd.read_csv(path)
    print("leakage suspects to drop before modelling:", ", ".join(LEAKAGE_SUSPECTS))
    facts = summarize(clean(raw))
    for key, value in facts.items():
        print(f"{key:22}: {value}")


if __name__ == "__main__":
    main()
