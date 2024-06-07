import numpy as np
from scipy.stats import entropy
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


class IterativeSampler(BaseEstimator):
    """
    IterativeSampler performs iterative sampling to enhance model training by
    selectively adding examples to the training set based on specified criteria.

    Parameters
    ----------
    model : object
        A scikit-learn compatible model instance (classifier or regressor).

    strategy : str, default='hard'
        The strategy used to select examples to add to the training set.
        Supported strategies are:
        - 'hard': Selects the most uncertain/challenging examples.
        - 'diverse': Selects examples that are most diverse feature-wise.
        - 'random': Selects examples randomly.

    init_sample_pool_size : int, default=100
        The initial size of the training set.

    step_size : int, default=50
        The number of examples to add to the training set in each iteration.

    max_iter : int, default=10
        The maximum number of iterations for the iterative sampling process.

    random_state : int, default=42
        The seed used by the random number generator for reproducibility.

    Methods
    -------
    fit(X, y)
        Fit the model using iterative sampling.

    predict(X)
        Predict class labels or target values for the provided data.

    predict_proba(X)
        Predict class probabilities for the provided data (for classifiers only).

    score(X, y)
        Return the score of the model on the provided test data and labels.

    get_params()
        Get parameters of this estimator wrapper.

    set_params(**params)
        Set the parameters of this estimator wrapper.
    """

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
            # The 'hard' strategy is based on selecting samples
            # which pose the highest challenge for the model. The metric
            # used is 'uncertainty' calculated as predictions entropy.
            case 'hard':
                uncertainty = entropy(predictions.T)
                selected_indices = np.argsort(uncertainty)[-self.step_size:]

            # For 'diverse' strategy we select samples that are
            # most diverse in feature space.
            case 'diverse':
                pool_features = predictions
                diversity = np.linalg.norm(pool_features[:, None] - pool_features, axis=2).sum(axis=0)
                selected_indices = np.argsort(diversity)[-self.step_size:]

            # The 'random' strategy selects samples at random.
            case 'random':
                selected_indices = np.random.choice(len(self.X_pool), size=self.step_size, replace=False)

            case _:
                raise ValueError("Unsupported criterion. Use 'hard', 'diverse' or 'random'.")

        return selected_indices

    def fit(self, X, y):
        """
        Fit the model using iterative sampling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            IterativeSampler object.
        """
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
        """
         Predict class labels or target values for the provided data.

         Parameters
         ----------
         X : array-like of shape (n_samples, n_features)
             Test samples.

         Returns
         -------
         y_pred : array of shape (n_samples,)
             Predicted class labels or target values.
         """
        return self.model_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for the provided data (for classifiers).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Predicted class probabilities.

        Raises
        ------
        AttributeError
            If the model does not have a predict_proba method.
        """
        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X)
        else:
            raise AttributeError(f"{self.model_.__class__.__name__} does not have method 'predict_proba'")

    def score(self, X, y):
        """
        Return the score of the model on the provided test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            The score of the model. Accuracy for classifiers, mean squared error for regressors.
        """
        predictions = self.predict(X)
        if hasattr(self.model_, 'predict_proba'):
            return accuracy_score(y, predictions)
        else:
            return mean_squared_error(y, predictions)

    def get_params(self):
        """
        Get parameters for this IterativeSampler.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'model': self.model,
            'strategy': self.strategy,
            'init_sample_pool_size': self.init_sample_pool_size,
            'step_size': self.step_size,
            'max_iter': self.max_iter,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            IterativeSampler parameters.

        Returns
        -------
        self : object
            IterativeSampler instance.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self
