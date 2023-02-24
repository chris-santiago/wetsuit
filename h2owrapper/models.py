from typing import List, Union

import h2o
from sklearn.base import ClassifierMixin, RegressorMixin
import pandas as pd
import numpy as np

from h2o.estimators import H2OEstimator
from sklearn.base import BaseEstimator


class BaseContainer(BaseEstimator):
    """Container class for Scikit-Learn models."""
    def __init__(self, estimator: H2OEstimator, features: List[Union[str, int]], response: Union[str, int]):
        self.estimator = estimator
        self.features = features
        self.response = response

    def fit(self, X, y) -> "BaseContainer":
        """
        Fit the estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.
        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).
        Returns
        -------
        BaseContainer
            Self.
        """
        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            frame = h2o.H2OFrame(
                np.concatenate((X, y), axis=1),
                column_names=self.features + [self.response]
            )
        elif isinstance(X, pd.DataFrame) and isinstance(y, (pd.DataFrame, pd.Series)):
            frame = h2o.H2OFrame(
                pd.concat([X, y], axis=1),
                column_names=self.features + [self.response]
            )
        else:
            raise TypeError("Expected X, y to be either type np.ndarray or pd.DataFrame")
        self.estimator.train(
            x=self.features,
            y=self.response,
            training_frame=frame
        )
        return self

    def predict(self, X):
        """
        Make predictions with fitted estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        np.ndarray
            Array of predicted values.
        """
        frame = h2o.H2OFrame(X, column_names=self.features)
        return self.estimator.predict(frame).as_data_frame()


class H2oRegressor(BaseContainer, RegressorMixin):
    pass


class H2oClassifier(BaseContainer, ClassifierMixin):
    pass
