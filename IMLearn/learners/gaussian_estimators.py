from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased
            estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not
            been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in
            `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in
            `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated
        estimation (where estimator is either biased or unbiased). Then sets
        `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X)
        if not self.biased_:
            self.var_ = np.var(X, ddof=1)
        else:
            self.var_ = np.var(X)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted
        estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling "
                             "`pdf` function")
        exp_power = (-0.5 / self.var_) * ((X - self.mu_) ** 2)
        pdf_values = np.exp(exp_power) / np.sqrt(2 * np.pi * self.var_)
        return np.array(pdf_values)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian
        model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        sum_in_exp = 0
        for x in X:
            sum_in_exp += (x - mu) ** 2
        # for this part doesnt the log cancels the exp
        exp_power = (-0.5) * sum_in_exp / (sigma ** 2)

        # log(x*y) = log(x)+log(y)
        return exp_power + ((-X.shape[0] / 2) * (np.log(2) + np.log(np.pi) +
                                                        np.log(sigma ** 2)))


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not
            been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in
            `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in
            `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated
        estimation. Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X, rowvar=False)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted
        estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling "
                             "`pdf` function")
        sqr = np.sqrt(
            ((2 * np.pi) ** self.cov_.shape[0]) * det(self.cov_))
        sum1 = np.sum((X - self.mu_.T) @ np.linalg.inv(self.cov_) * (X -
                                                                 self.mu_.T))

        return (1 / sqr) * np.exp(-0.5 * sum1)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> \
            float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian
        model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        sum1 = np.sum((X - mu.T) @ np.linalg.inv(cov) * (X - mu.T))
        cov_log = slogdet(cov)
        logs = X.shape[1] * np.log(2 * np.pi) + cov_log[0] * cov_log[1]
        return (-0.5) * X.shape[0] * logs + (-0.5 * sum1)
