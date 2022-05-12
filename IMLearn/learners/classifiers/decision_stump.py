from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART
    algorithm
    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split
    self.j_ : int
        The index of the feature by which to split the data
    self.sign_: int
        The label to predict for samples where the value of the j'th feature is
        about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_err = 1
        feature = 0
        sign = 1
        chosen_threshold = 1
        for j, s in product(range(X.shape[1]), [1, -1]):
            threshold, curr_err = self._find_threshold(X[:, j], y, s)
            if curr_err < min_err:
                min_err = curr_err
                chosen_threshold = threshold
                feature = j
                sign = s
        self.j_ = feature
        self.sign_ = sign
        self.threshold_ = chosen_threshold
        self.fitted_ = True


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign`
        whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.array([self.sign_ if x[self.j_] >= self.threshold_ else
                         -self.sign_ for x in X])

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform
        a split
        The threshold is found according to the value minimizing the
        misclassification error along this feature
        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for
        labels: ndarray of shape (n_samples,)
            The labels to compare against
        sign: int
            Predicted label assigned to values equal to or above threshold
        Returns
        -------
        thr: float
            Threshold by which to perform split
        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold
        Notes
        -----
        For every tested threshold, values strictly below threshold are
        predicted as `-sign` whereas values which equal to or above the
        threshold are predicted as `sign`
        """

        sorted_indexes = np.argsort(values)
        values_after_sort = np.take(values, sorted_indexes)
        labels_after_sort = np.take(labels, sorted_indexes)

        min_error = 1.0
        threshold = 0
        sign_labels = np.ones(values_after_sort.shape[0])
        sign_labels *= sign
        for i in range(values_after_sort.shape[0]):
            err = np.sum(np.where(sign_labels != np.sign(labels_after_sort),
                                  np.abs(labels_after_sort), 0))
            err /= len(values_after_sort)
            if err < min_error:
                min_error = err
                threshold = values_after_sort[i]
            sign_labels[i] = -sign
        return threshold, min_error

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))