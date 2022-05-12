from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in
        `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in
        `LDA.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, \
                                                      None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same
        covariance matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((self.classes_.size, X.shape[1]))
        for i in range(len(self.classes_)):
            self.mu_[i] = np.mean(X[y == self.classes_[i]], axis=0)

        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            where = np.where(self.classes_ == y[i])
            x_mu = X[i] - self.mu_[where[0]]
            self.cov_ += x_mu * x_mu.transpose()
        self.cov_ /= X.shape[0] - self.classes_.size

        self._cov_inv = inv(self.cov_)

        self.pi_ = np.zeros(self.classes_.size)
        for i in range(len(self.classes_)):
            self.pi_[i] = y[y == self.classes_[i]].size / y.size
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
        likelihoods = self.likelihood(X)
        y = np.zeros(X.shape[0])
        for i in range(y.size):
            y[i] = np.argmax(likelihoods[i])
        return y

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

        likelihood = np.zeros((X.shape[0], self.classes_.size))
        for i in range(self.classes_.size):
            exp = np.exp(-0.5 * np.sum((X - self.mu_[i]) @ self._cov_inv * (
                    X-self.mu_[i]), axis=1))
            sqr = np.sqrt(np.power(2 * np.pi, X.shape[1]) * det(self.cov_))
            likelihood[:,i] = (exp / sqr) * self.pi_[i]
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
        return misclassification_error(y, self.predict((X)))
