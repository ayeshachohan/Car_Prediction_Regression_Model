import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Binary Encoding Transformer
class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns, mappings):
        self.columns = columns
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(self.mappings[col])
        return X_copy

# Multi-Column Label Encoding Transformer
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.encoders = {col: LabelEncoder().fit(X[col]) for col in self.columns}
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = self.encoders[col].transform(X_copy[col])
        return X_copy


# Mileage Scaler Transformer
class MileageScaler(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        self.scaler = MinMaxScaler()
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.column] = pd.to_numeric(X_copy[self.column], errors='coerce')
        X_copy.dropna(subset=[self.column], inplace=True)
        X_copy[self.column] = self.scaler.fit_transform(X_copy[[self.column]])
        return X_copy

# Column Dropper Transformer
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.column, axis=1)
