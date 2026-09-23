from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from src.dataset import load_split


def build_pipeline(numeric, categorical):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer()),
                          ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("pre", pre),
                     ("model", RandomForestClassifier(random_state=42))])


def run():
    X_tr, X_te, y_tr, y_te = load_split()
    numeric = X_tr.select_dtypes("number").columns.tolist()
    categorical = [c for c in X_tr.columns if c not in numeric]
    pipe = build_pipeline(numeric, categorical)
    scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="f1")
    print("CV F1: %.3f +/- %.3f" % (scores.mean(), scores.std()))


if __name__ == "__main__":
    run()