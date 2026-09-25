import matplotlib

matplotlib.use("Agg")  # write a file, never open a window
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import load_split
from src.metrics import precision, recall
from src.threshold import threshold_for_recall

MALIGNANT = 0  # in load_breast_cancer, 0 is the class you must not miss


def main():
    X_train, X_test, y_train, y_test = load_split()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(X_train, y_train)
    score = model.predict_proba(X_test)[:, MALIGNANT]  # probability of malignant
    y = (y_test == MALIGNANT).astype(int)

    print(f"ROC-AUC          : {roc_auc_score(y, score):.3f}")
    print(f"average precision: {average_precision_score(y, score):.3f}")

    default_pred = (score >= 0.5).astype(int)
    print(f"at 0.50 threshold: recall {recall(y, default_pred):.3f}, "
          f"precision {precision(y, default_pred):.3f}")

    t = threshold_for_recall(y, score, 0.98)
    tuned_pred = (score >= t).astype(int)
    print(f"for recall >= 0.98: threshold {t:.3f}, recall {recall(y, tuned_pred):.3f}, "
          f"precision {precision(y, tuned_pred):.3f}")

    prec, rec, _ = precision_recall_curve(y, score)
    plt.plot(rec, prec)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision-recall curve (malignant)")
    plt.savefig("reports/pr_curve.png", dpi=120, bbox_inches="tight")
    print("saved reports/pr_curve.png")


if __name__ == "__main__":
    main()