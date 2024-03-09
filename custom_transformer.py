# custom_transformers.py

# import libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
# defining class MulticolumnLabelEncoder 
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    # initializing class
    def __init__(self, columns=None):
        self.columns = columns
    # fit function
    def fit(self, X, y=None):
        return self
    # transforming
    def transform(self, X):
        # for each column in data
        for column in self.columns:
            # label encoder to convert categorical to numerical
            le = LabelEncoder()
            # fit_transforming data
            X[column] = le.fit_transform(X[column])
        return X
