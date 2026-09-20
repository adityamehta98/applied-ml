from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from src.dataset import load_split


def run():
    X_tr, X_te, y_tr, y_te = load_split()
    models = {"logreg": LogisticRegression(max_iter=2000),
              "rf": RandomForestClassifier(random_state=42)}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        print(name, "F1:", round(f1_score(y_te, pred), 3))
        print(classification_report(y_te, pred))


if __name__ == "__main__":
    run()