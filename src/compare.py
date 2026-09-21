import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.dataset import load_split

MODELS = {"logreg": LogisticRegression(max_iter=2000),
          "rf": RandomForestClassifier(random_state=42),
          "xgb": XGBClassifier(eval_metric="logloss")}


def run():
    X_tr, X_te, y_tr, y_te = load_split()
    rows = []
    for name, m in MODELS.items():
        m.fit(X_tr, y_tr); p = m.predict(X_te)
        rows.append({"model": name,
                     "acc": accuracy_score(y_te, p),
                     "prec": precision_score(y_te, p),
                     "rec": recall_score(y_te, p),
                     "f1": f1_score(y_te, p)})
    print(pd.DataFrame(rows).round(3).to_markdown(index=False))


if __name__ == "__main__":
    run()