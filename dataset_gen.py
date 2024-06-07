from sklearn.datasets import make_classification


def make_classification_dataset():
    return make_classification(n_samples=1000, n_features=20, random_state=42)
