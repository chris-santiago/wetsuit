from typing import List, Union

import h2o

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class H2oFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features: List[Union[str, int]], response: Union[str, int]):
        self.features = features
        self.response = response

    def fit(self, X, y):
        self.X_frame_ = h2o.H2OFrame(X, column_names=self.features)
        self.y_frame_ = h2o.H2OFrame(y, column_names=[self.response])
        return self

    def transform(self, X, y):
        check_is_fitted(self)
        _, _ = X, y
        return self.X_frame_, self.y_frame_

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X, y):
        check_is_fitted(self)
        return X.as_data_frame(), y.as_data_frame()
