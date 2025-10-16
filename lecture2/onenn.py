# https://scikit-learn.org/stable/developers/develop.html

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.metrics import DistanceMetric

class OneNN(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self.dist = DistanceMetric.get_metric('euclidean')
    
    def fit(self, X, y):
        # Check that X and y have correct shape, set n_features_in_, etc.
        X, y = validate_data(self, X, y)
        # Store the classes seen during fit
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self
    
    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = validate_data(self, X, reset=False)
        
        # argsort returns id of sorted distances
        distance_matrix = np.argsort(self.dist.pairwise(X, self.X_), axis=1)[1, :]
        preds = self.y_[distance_matrix]
        
        return preds