import numpy as np
from scipy.stats import entropy
from sklearn.base import clone
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


class IterativeSampler:
    def __init__(self, model, strategy='hard', init_sample_pool_size=0.01, step_size=100, max_iter=10, random_state=42):
        self.model_ = None
        self.model = model
        self.strategy = strategy
        self.init_sample_pool_size = init_sample_pool_size
        self.step_size = step_size
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(self.random_state)

    def _get_extensions_idx(self, predictions):
        match self.strategy:
            case 'hard':
                uncertainty = entropy(predictions.T)
                selected_indices = np.argsort(uncertainty)[-self.step_size:]

            case 'diverse':
                pool_features = predictions
                diversity = np.linalg.norm(pool_features[:, None] - pool_features, axis=2).sum(axis=0)
                selected_indices = np.argsort(diversity)[-self.step_size:]

            case 'random':
                selected_indices = np.random.choice(len(self.X_pool), size=self.step_size, replace=False)

            case _:
                raise ValueError("Unsupported criterion. Use 'hard', 'diverse' or 'random'.")

        return selected_indices

    def fit(self, X, y):
        # Split X and y into a set of initial training samples
        # and a set of remaining data to sample from
        self.X_train, self.X_pool, self.y_train, self.y_pool = train_test_split(
            X, y, train_size=self.init_sample_pool_size, random_state=self.random_state
        )
        self.model_ = clone(self.model)

        for i in range(self.max_iter):
            if len(self.X_pool) == 0:
                break

            self.model_.fit(self.X_train, self.y_train)

            if hasattr(self.model_, 'predict_proba'):
                predictions = self.model_.predict_proba(self.X_pool)
            else:
                predictions = self.model_.predict(self.X_pool).reshape(-1, 1)
            selected_indices = self._get_extensions_idx(predictions)

            # Add selected samples to training set
            # and remove them from sampling pool
            self.X_train = np.vstack([self.X_train, self.X_pool[selected_indices]])
            self.y_train = np.hstack([self.y_train, self.y_pool[selected_indices]])
            self.X_pool = np.delete(self.X_pool, selected_indices, axis=0)
            self.y_pool = np.delete(self.y_pool, selected_indices, axis=0)

            score = self.score(self.X_train, self.y_train)
            print(f'Iteration {i}/{self.max_iter}, training score: {np.round(score, 3)}')

        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X)
        else:
            raise AttributeError(f"{self.model_.__class__.__name__} does not have method 'predict_proba'")

    def score(self, X, y):
        predictions = self.predict(X)
        if hasattr(self.model_, 'predict_proba'):
            return accuracy_score(y, predictions)
        else:
            return mean_squared_error(y, predictions)

    def get_params(self):
        return {
            'model': self.model,
            'strategy': self.strategy,
            'init_sample_pool_size': self.init_sample_pool_size,
            'step_size': self.step_size,
            'max_iter': self.max_iter,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
