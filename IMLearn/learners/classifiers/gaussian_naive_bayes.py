from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in
            `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in
            `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in
            `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, nk = np.unique(y, return_counts=True)
        self.mu_ = np.zeros((self.classes_.size, X.shape[1]))
        self.vars_ = np.zeros((self.classes_.size, X.shape[1]))
        self.pi_ = nk / X.shape[0]
        for i in range(len(self.classes_)):
            self.mu_[i] = np.mean(X[y == self.classes_[i]], axis=0)
            self.vars_[i] = np.sum((X[y == self.classes_[i]] - self.mu_[i])
                                   ** 2, axis=0) / (nk[i] - 1)
            self.pi_[i] = nk[i] / X.shape[0]
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling "
                             "`likelihood` function")

        likelihood = np.ones((X.shape[0], self.classes_.size))
        for i in range(self.classes_.size):
            cov = np.diag(self.vars_[i])
            exp = np.exp(-0.5 * np.sum((X - self.mu_[i]) @ inv(cov) * (
                    X - self.mu_[i]), axis=1))
            sqr = np.sqrt(np.power(2 * np.pi, X.shape[1]) * det(cov))
            likelihood[:, i] = (exp / sqr) * self.pi_[i]
        return likelihood


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
