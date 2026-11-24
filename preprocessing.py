
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        values = pd.Series(np.asarray(X).ravel())
        self.frequency_map_ = values.value_counts().to_dict()
        return self

    def transform(self, X):
        values = pd.Series(np.asarray(X).ravel())

        encoded = (
            values.map(self.frequency_map_)
            .fillna(0)
            .to_numpy()
            .reshape(-1, 1)
        )

        return encoded
