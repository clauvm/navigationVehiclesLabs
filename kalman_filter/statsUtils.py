import math
import numpy as np
from numpy.linalg import inv


def gaussian_function(mu, sigma_squared, x):
    ''' f takes in a mean and squared variance, and an input x
       and returns the gaussian value.'''
    coefficient = 1.0 / math.sqrt(2.0 * math.pi * sigma_squared)
    exponential = math.exp(-0.5 * (x - mu) ** 2 / sigma_squared)
    return coefficient * exponential


# the update function
def update(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters.'''
    # Calculate the new parameters
    new_mean = (var2 * mean1 + var1 * mean2) / (var2 + var1)
    new_var = 1 / (1 / var2 + 1 / var1)

    return [new_mean, new_var]


# the motion update/predict function
def predict(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters, after motion.'''
    # Calculate the new parameters
    new_mean = mean1 + mean2
    new_var = var1 + var2

    return [new_mean, new_var]


def gauss_pdf(X, M, S):
    """
    Computes the Predictive probability (likelihood) of measurement
    :param X: measurement vector
    :param M: Mean of predictive distribution of measurement vector
    :param S: the Covariance or predictive mean of measurement vector
    :return: Gaussian probability distribution
    """
    if M.shape()[1] == 1:
        DX = X - np.tile(M, X.shape()[1])
        E = 0.5 * sum(DX * (np.dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape()[0] * np.log(2 * np.pi) + 0.5 * np.log(np.det(S))
        P = np.exp(-E)
    elif X.shape()[1] == 1:
        DX = np.tile(X, M.shape()[1]) - M
        E = 0.5 * sum(DX * (np.dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape()[0] * np.log(2 * np.pi) + 0.5 * np.log(np.det(S))
        P = np.exp(-E)
    else:
        DX = X - M
        E = 0.5 * np.dot(DX.T, np.dot(inv(S), DX))
        E = E + 0.5 * M.shape()[0] * np.log(2 * np.pi) + 0.5 * np.log(np.det(S))
        P = np.exp(-E)
    return (P[0], E[0])
