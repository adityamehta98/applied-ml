"""Draw the validation curve and mark the sweet spot where validation stops improving."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import load_split
from src.overfitting import depth_curve, sweet_spot

DEPTHS = [1, 2, 3, 4, 5, 7, 10, 15, 20]


def main():
    X_train, _, y_train, _ = load_split()
    train, val = depth_curve(X_train, y_train, DEPTHS)
    best = sweet_spot(DEPTHS, val)

    for d, t, v in zip(DEPTHS, train, val):
        print(f"depth {d:2}: train F1 {t:.3f}  val F1 {v:.3f}  gap {t - v:.3f}")
    print(f"sweet spot: max_depth={best}")

    plt.plot(DEPTHS, train, marker="o", label="training F1")
    plt.plot(DEPTHS, val, marker="o", label="validation F1")
    plt.axvline(best, color="grey", linestyle="--", label=f"sweet spot (depth {best})")
    plt.xlabel("max_depth")
    plt.ylabel("F1")
    plt.title("Validation curve: a decision tree overfitting as it deepens")
    plt.legend()
    plt.savefig("reports/validation_curve.png", dpi=120, bbox_inches="tight")
    print("saved reports/validation_curve.png")


if __name__ == "__main__":
    main()
