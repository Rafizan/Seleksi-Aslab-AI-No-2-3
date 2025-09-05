import numpy as np
import pandas as pd

class Scaler:
    def __init__(self, method='standard'):
        self.method = method
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None
    def fit(self, X):
        if self.method == 'standard':
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
        elif self.method == 'minmax':
            self.min_ = X.min(axis=0)
            self.max_ = X.max(axis=0)
        else:
            raise ValueError('Unknown scaling method')
    def transform(self, X):
        if self.method == 'standard':
            return (X - self.mean_) / self.std_
        elif self.method == 'minmax':
            return (X - self.min_) / (self.max_ - self.min_)
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    def inverse_transform(self, X_scaled):
        if self.method == 'standard':
            return X_scaled * self.std_ + self.mean_
        elif self.method == 'minmax':
            return X_scaled * (self.max_ - self.min_) + self.min_
        
def precision_score(y_true, y_pred, positive_class=1):
    tp = np.sum((y_pred == positive_class) & (y_true == positive_class))
    fp = np.sum((y_pred == positive_class) & (y_true != positive_class))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0