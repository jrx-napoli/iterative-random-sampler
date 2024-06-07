import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


class IterativeSampler:
    def __init__(self, model, criterion='hard', sample_size=100, step_size=50, max_iter=10, random_state=42):
        self.model = model
        self.criterion = criterion
        self.sample_size = sample_size
        self.step_size = step_size
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X, y):
        self.X_train, self.X_pool, self.y_train, self.y_pool = train_test_split(
            X, y, train_size=self.sample_size, random_state=self.random_state
        )
        self.model_ = clone(self.model)
        self.history_ = []

        for i in range(self.max_iter):
            self.model_.fit(self.X_train, self.y_train)
            predictions = self.model_.predict(self.X_pool)

            selected_indices = self._get_extensions_idx(predictions)

            self.X_train = np.vstack([self.X_train, self.X_pool[selected_indices]])
            self.y_train = np.hstack([self.y_train, self.y_pool[selected_indices]])
            self.X_pool = np.delete(self.X_pool, selected_indices, axis=0)
            self.y_pool = np.delete(self.y_pool, selected_indices, axis=0)

            score = self.evaluate_model(self.X_train, self.y_train)
            self.history_.append(score)

        return self

    def _get_extensions_idx(self, predictions):
        if self.criterion == 'hard':
            uncertainty = np.abs(predictions - 0.5)
            selected_indices = np.argsort(uncertainty)[:self.step_size]
        elif self.criterion == 'diverse':
            pool_features = self.model_.predict_proba(self.X_pool) \
                if hasattr(self.model_, 'predict_proba') else predictions.reshape(-1, 1)
            diversity = np.linalg.norm(pool_features[:, None] - pool_features, axis=2).sum(axis=0)
            selected_indices = np.argsort(diversity)[-self.step_size:]
        elif self.criterion == 'random':
            selected_indices = np.random.choice(len(self.X_pool), size=self.step_size, replace=False)
        else:
            raise ValueError("Unsupported criterion. Use 'hard', 'diverse' or 'random'.")

        return selected_indices

    def evaluate_model(self, X, y):
        predictions = self.model_.predict(X)
        if hasattr(self.model_, 'predict_proba'):
            return accuracy_score(y, predictions)
        else:
            return mean_squared_error(y, predictions)

    def get_history(self):
        return self.history_
