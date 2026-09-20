"""Dataset loading and splitting utilities."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_split(test_size=0.2, seed=42):
    """Load the dataset and return a reproducible, stratified split.

    Returns: X_train, X_test, y_train, y_test
    """
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=test_size,
                            random_state=seed, stratify=y)