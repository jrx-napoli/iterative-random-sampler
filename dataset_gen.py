from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def make_classification_dataset(args):
    X, y = make_classification(n_samples=args.n_samples,
                               n_classes=args.n_classes,
                               n_features=args.n_features,
                               n_informative=args.n_informative,
                               random_state=args.seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=args.seed)
    return X_train, X_test, y_train, y_test
