# Custom anomaly detectors
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
import numpy as np
from math import gamma, pi
from scipy.stats import multivariate_normal

class MultivariateGaussian(BaseEstimator):
    """
    Anomaly detector using the Multivariate Gaussian class
    Based on Andrew Ng's lecture
    """
    def __init__(self, epsilon=None):
        """
        Initiator
        Args:
            epsilon (float): statistical threshold for the estimator to declare an observation as anomalous. 
        """
        self.epsilon = epsilon
    
    def fit(self, train_data):
        """
        Method to 'train' the estimator, which will calculate vector mu (mean vector) and matrix Sigma (covariance matrix).
        Args:
            train_data (pandas DataFrame or array): the train data set
        Returns:
            self (callable): called instance
        """
        if type(train_data) != np.ndarray:
            X = np.array(train_data)
        else:
            pass
        # Calculating n-dimensional mean vector (mu)
        self.mu = np.mean(X, axis=0)
        # Calculating n-by-n covariance matrix (Sigma)
        self.Sigma = np.cov(X, rowvar=False)
        assert self.Sigma.shape == (X.shape[1], X.shape[1]), f"The covariance matrix must be {X.shape[1]} by {X.shape[1]}"
        return self

    def predict(self, test_data):
        """
        Method to generate predictions
        Args:
            test_data (pandas DataFrame or array): the test data set
        Returns:
            predictions (array): predictions array
        """
        if type(test_data) != np.ndarray:
            X_test = np.array(test_data)
        else:
            pass
        p_X = multivariate_normal.pdf(X_test, mean=self.mu, cov=self.Sigma, allow_singular=True)
        assert p_X.shape == (X_test.shape[0],), f"The predicted probability densities must be a {X_test.shape[0]}-dimensional vector"
        # Calculating m-dimensional prediction vector (probability density function)
        predictions = np.where(p_X < self.epsilon, 1, 0)
        return predictions

class MultivariateTDistribution(MultivariateGaussian):
    """
    Anomaly detector using the 'fat-tail' Student's t-Distribution.
    """
    def __init__(self, epsilon=None, df=None):
        """
        Initiator
        Args:
            epsilon (float): statistical threshold for the estimator to declare an observation as anomalous.
            df (float or int): degrees of freedom parameter, which regulates the 'height' of the tails of the distribution. 
        """
        self.epsilon = epsilon
        # Degrees of freedom parameter
        self.df = df

    def predict(self, test_data):
        """
        Method to generate predictions
        Args:
            test_data (pandas DataFrame or array): the test data set
        Returns:
            predictions (array): predictions array
        """
        if type(test_data) != np.ndarray:
            X_test = np.array(test_data)
        else:
            pass
        df = self.df
        n = test_data.shape[1]
        mu = self.mu
        Sigma = self.Sigma
        pdf_list = []
        # The following codes should be vectorised, any suggestions are welcomed
        for x_test in X_test:
            term_1_num = gamma((n+df)/2)
            term_1_denom = gamma((df/2)) * df**(n/2) * pi**(n/2) * np.linalg.det(Sigma)**0.5
            term_1 = term_1_num / term_1_denom
            term_2 = (1 + 1/df * (x_test - mu).T @ np.linalg.inv(Sigma) @ (x_test - mu)) ** (-1 * (n+df)/2)
            pdf_x = term_1 * term_2
            pdf_list.append(pdf_x)
        pdf_X = np.array(pdf_list)
        assert pdf_X.shape == (X_test.shape[0],), f"The predicted probability densities must be a {X_test.shape[0]}-dimensional vector"
        predictions = np.where(pdf_X < self.epsilon, 1, 0)
        return predictions

class SimpleAnomalyDetector(BaseEstimator):
    """
    Simple anomaly detector, a predicted probability density is the product of all features' probabilities.
    Based on Andrew Ng's lecture.
    """
    def __init__(self, epsilon=None):
        """
        Initiator
        Args:
            epsilon (float): statistical threshold for the estimator to declare an observation as anomalous. 
        """
        self.epsilon = epsilon

    def fit(self, train_data):
        """
        Method to 'train' the estimator, which will calculate vector mu (mean vector) and vector sigma (feature variance vector).
        Args:
            train_data (pandas DataFrame or array): the train data set
        Returns:
            self (callable): called instance
        """
        if type(train_data) != np.ndarray:
            X = np.array(train_data)
        else:
            pass
        # Calculating n-dimensional mean vector (mu)
        self.mu = np.mean(X, axis=0).reshape(-1, 1)
        # Calculating n-dimensional vector of feature variances (sigma)
        self.sigma = np.var(X, axis=0).reshape(-1, 1)
        assert self.mu.shape == (X.shape[1], 1)
        assert self.sigma.shape == (X.shape[1], 1)
        return self

    def predict(self, test_data):
        """
        Method to generate predictions
        Args:
            test_data (pandas DataFrame or array): the test data set
        Returns:
            predictions (array): predictions array
        """
        if type(test_data) != np.ndarray:
            X_test = np.array(test_data)
        else:
            pass
        mu = self.mu
        sigma = self.sigma
        term_1 = -1 * np.square(X_test - np.tile(mu, len(X_test)).T)
        term_2 = 2 * np.tile(sigma, len(X_test)).T
        term_3 = np.exp(term_1 / term_2)
        term_4 = (1 / ((2*np.pi)**0.5 * np.power(sigma, np.array(0.5))))
        term_4 = np.tile(term_4, len(X_test)).T
        term_5 = term_3 * term_4
        pdf_X = np.prod(term_5, axis=1)
        assert pdf_X.shape == (X_test.shape[0],), f"The predicted probability densities must be a {X_test.shape[0]}-dimensional vector"
        predictions = np.where(pdf_X < self.epsilon, 1, 0)
        return predictions
        

# INFLO class
class INFLO(BaseEstimator):
    def __init__(self, contamination=None):
        """
        Anomaly detector based on "Ranking Outliers Using Symmetric Neighborhood Relationship" paper (Jin, Tung, Han and Wang, 2006)
        Args:
            - contamination: expected fraction of the dataset which are outliers.
        """
        self.contamination_ = contamination
        assert (contamination > 0) & (contamination <= 0.5), "Contamination must be between 0 and 0.5"

    def fit_transform(self, data=None, k=None):
        self.k_ = k
        assert isinstance(k, int), "k must be an integer"
        assert (k<len(data)) & (k>0), "k must be smaller than the number of observations AND greater than 0."
        n_row = len(data)
        # Initial K-Nearest Neighbors search
        knn_model = NearestNeighbors(n_neighbors=self.k_)
        knn_model.fit(data)
        knn_dist_obj, knn_dist_indices = knn_model.kneighbors(data)
        obj_density = np.apply_along_axis(lambda x: 1 / np.max(x), 1, knn_dist_obj)

        # Reverse Nearest Neighbors (RNNs) search and count
        RNN = np.zeros((n_row, 1), dtype=int)
        avg_density_influ_space = np.zeros((n_row, 1), dtype=int)
        INFLO_list = []

        # Main loop - 
        for i in range(n_row):
            influ_space = np.array(np.argwhere(knn_dist_indices==i)[1:, 0])
            if len(influ_space) == 0:
                RNN[i] = k
                influ_space = knn_dist_indices[i, 1:]
            else:
                RNN[i] = len(influ_space)
                influ_space = np.unique(np.append(influ_space, knn_dist_indices[i, 1:]))
            RNN[i] = len(influ_space)
            
            sum_RNN_obs_density = 0
            for j in range(len(influ_space)):
                rnn_obs_density = obj_density[influ_space[j]]
                sum_RNN_obs_density += rnn_obs_density
            avg_density_influ_space = sum_RNN_obs_density / RNN[i]
            INFLO_ratio = avg_density_influ_space / obj_density[i]
            INFLO_list.append(INFLO_ratio)
        self.INFLO_results_ = np.array(INFLO_list).reshape(-1,)

        # Picking outliers - selecting top-n observations with highest INFLO based on contamination
        top_n = int(self.contamination_ * len(data))
        top_n_indices = self.INFLO_results_.argsort()[-top_n:][::-1]
        anomaly_detection_results = np.zeros((self.INFLO_results_.shape[0],))
        for index, item in enumerate(self.INFLO_results_):
            if index in top_n_indices:
                anomaly_detection_results[index] = int(1)
            else:
                anomaly_detection_results[index] = int(0)
        return anomaly_detection_results