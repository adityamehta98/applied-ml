"""How much does one split's score swing, compared with a 5-fold estimate?"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.cv import cross_val_scores, kfold_indices
from src.dataset import load_split


def main():
    X_train, _, y_train, _ = load_split()  # the test set stays sealed
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

    print("Single 80/20 splits of the training data, five different seeds:")
    for seed in range(5):
        X_a, X_b, y_a, y_b = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train)
        score = f1_score(y_b, model.fit(X_a, y_a).predict(X_b))
        print(f"  seed {seed}: F1 = {score:.3f}")

    scores = cross_val_scores(model, X_train, y_train,
                              kfold_indices(len(X_train), 5), f1_score)
    print(f"5-fold: F1 = {scores.mean():.3f} +/- {scores.std():.3f}")


if __name__ == "__main__":
    main()
