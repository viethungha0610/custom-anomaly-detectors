# Custom Multivariate Gaussian anomaly detection classifier
from sklearn.base import BaseEstimator
import numpy as np
from math import gamma, pow, pi
from scipy.stats import multivariate_normal

class MultivariateGaussian(BaseEstimator):
    """
    Anomaly detector using the Multivariate Gaussian (Normal) distribution
    """
    def __init__(self, epsilon=None):
        """
        The main hyperparameter is epsilon. The estimator will flag an observation as anomaly if p(x) < epsilon.
        """
        self.epsilon = epsilon
    
    def fit(self, train_data):
        X = np.array(train_data)
        # Calculating n-dimensional mean vector (mu)
        self.mu = np.mean(X, axis=0)
        # Calculating n-by-n covariance matrix (sigma)
        self.sigma = np.cov(X, rowvar=False)
        assert self.sigma.shape == (X.shape[1], X.shape[1]), f"The covariance matrix must be {X.shape[1]} by {X.shape[1]}"
        return self

    def predict(self, test_data):
        X_test = np.array(test_data)
        p_X = multivariate_normal.pdf(X_test, mean=self.mu, cov=self.sigma, allow_singular=True)
        assert p_X.shape == (X_test.shape[0],), f"The predicted probabilities must be a {X_test.shape[0]}-dimensional vector"
        # Calculating m-dimensional prediction vector (probability density function)
        predictions = np.where(p_X < self.epsilon, 1, 0)
        return predictions

class MultivariateTDistribution(MultivariateGaussian):
    """
    Anomaly detector using the 'fat-tail' Student's t-Distribution.
    """
    def __init__(self, epsilon=None, df=None):
        self.epsilon = epsilon
        # Degrees of freedom parameter
        self.df = df

    def predict(self, test_data):
        X_test = np.array(test_data)
        df = self.df
        n = test_data.shape[1]
        mu = self.mu
        sigma = self.sigma
        pdf_list = []
        # The following codes should be vectorised, any suggestions are welcomed
        for x_test in X_test:
            term_1_num = gamma((n+df)/2)
            term_1_denom = gamma((df/2)) * df**(n/2) * pi**(n/2) * np.linalg.det(sigma)**0.5
            term_1 = term_1_num / term_1_denom
            term_2 = (1 + 1/df * (x_test - mu).T @ np.linalg.inv(sigma) @ (x_test - mu)) ** (-1 * (n+df)/2)
            pdf_x = term_1 * term_2
            pdf_list.append(pdf_x)
        pdf_X = np.array(pdf_list)
        assert pdf_X.shape == (X_test.shape[0],), f"The predicted probabilities must be a {X_test.shape[0]}-dimensional vector"
        predictions = np.where(pdf_X < self.epsilon, 1, 0)
        return predictions