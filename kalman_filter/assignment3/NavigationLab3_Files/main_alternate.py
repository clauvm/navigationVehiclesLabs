import random
import math
from numpy.linalg import inv, det

import matplotlib.pyplot as plt
import numpy as np

from kalman_filter.assignment2.main import update_step, prediction_step
from kalman_filter.assignment2.utils import SAC, TA_AC, NESS, TA_NIS, NIS, plot_ness, plot_nis
from kalman_filter.assignment3.NavigationLab3_Files.generate_data_2D_fun import generate_data_2D_fun
from kalman_filter.statsUtils import gauss_pdf

number_of_samples = 1507

from scipy.stats import norm
from sympy import Symbol, symbols, Matrix, sin, cos, sqrt
from sympy import init_printing
from sympy.utilities.codegen import codegen

init_printing(use_latex=True)


def update_step_extended(x_hat, P_hat, Z, C, R):
    """
    Computes the posterior mean X and covariance P of the system state given a new measurement at time step k
    This is the measurement update phase or the corrector
    :param x_hat: predicted mean(x_hat)
    :param P_hat: predicted covariance (P_hat)
    :param Z: measurement vector
    :param C: measurement matrix
    :param R: covariance matrix
    :return:
    """
    IM = np.dot(C, x_hat)  # the Mean of predictive distribution of Y
    IS = R + np.dot(C, np.dot(P_hat, C.T))  # the Covariance or predictive mean of Y

    K = np.dot(P_hat, np.dot(C.T, inv(IS)))  # Kalman Gain matrix

    # h[k,0]=np.sqrt(np.dot(xhatminus[k, (0,2)].T, xhatminus[k, (0,2)]))
    # h[k,1]=np.arctan2(xhatminus[k, 1], xhatminus[k, 0])

    h1 = np.sqrt(np.dot(x_hat.T, x_hat))
    h2 = np.arctan2(x_hat[1], x_hat[0])
    h = np.array([h1, h2])

    X = x_hat + np.dot(K, (Z - h.T).T)
    P = np.dot((np.identity(4) - np.dot(K, C)), P_hat)
    LH = 0
    return (X, P, K, IM, IS, LH)


def prediction_step_extended(A, x_hat_previous, B, control_input, Q, P_previous):
    """
    Computes prediction of mean (X) and covariance(P) of the system at a specific timestep
    These are the time update equations on the predictor phase
    :param A: The transition n n × matrix (In this case is 1)
    :param x_hat_previous: The mean state estimate of the previous step (k - 1)
    :param B: The input effect matrix. Since input control is 0 this will be 0
    :param control_input: In this case is 0
    :param Q: The process noise covariance matrix
    :param P_previous: The state covariance of previous step
    :return: predicted mean(x_hat) and predicted covariance (P_hat)
    """
    x_hat = np.dot(A, x_hat_previous)
    P_hat = np.dot(A, np.dot(P_previous, A.T)) + Q
    return (x_hat, P_hat)


def HJacobian_at(x):
    """ compute Jacobian of H matrix at x """

    horiz_dist = x[0]
    altitude = x[2]
    denom = sqrt(horiz_dist ** 2 + altitude ** 2)
    return np.array([[horiz_dist / denom, 0., altitude / denom]])


def hx(x):
    """ compute measurement for slant range that
    would correspond to state x.
    """

    return (x[0] ** 2 + x[2] ** 2) ** 0.5


def single_simulation_constant_velocity_model(q=10, matched=True, piecewise=False, plot=False, model=''):
    """
    Constant (or piecewise) acceleration model eq 15 and 16
    :param piecewise: whether data should be generated with equations 15 or 16
    :param q:
    """
    """
    Constant (or piecewise) velocity white noise accelleration eq 13 and 14
    :param piecewise: whether data should be generated with equations 13 or 14
    :param q:
    """

    # intial parameters
    n_iter = 1507
    sz = (n_iter,)  # size of array

    xhat = np.zeros((n_iter, 4, 1))  # a posteri estimate of x
    P = np.zeros((n_iter, 4, 4))  # a posteri error estimate
    xhatminus = np.zeros((n_iter, 4, 1))  # a priori estimate of x
    Pminus = np.zeros((n_iter, 4, 4))  # a priori error estimate
    K = np.zeros((n_iter, 4, 2))  # gain or blending factor
    H = 1  # identity
    w = np.zeros((n_iter, 4, 1))
    C = np.zeros((n_iter, 2, 4))

    A = np.matrix([[1, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 0]])
    T = 0.5

    R = np.matrix([(10, 0), (0, 0.001)])  # estimate of measurement variance, change to see effect

    var_a = 0.001
    Q = np.matrix([[T ** 3 / 3, T ** 2 / 2, 0, 0],
                   [T ** 2 / 2, T, 0, 0],
                   [0, 0, T ** 3 / 3, T ** 2 / 2],
                   [0, 0, T ** 2 / 2, T]]) * var_a

    # intial guesses
    xhat[0] = np.matrix([0.0, 0.0, 0.0, 0]).T
    P[0] = np.linalg.inv([(1.0, 2, 2, 1), (2, 3, 3, 2), (2, 3, 1, 1), (1, 2, 1, 1)])

    rand = np.random.multivariate_normal([0, 0, 0, 0], Q, n_iter)
    w[:, :, 0] = rand

    z, x = generate_data_2D_fun(10, 10, 10, 1e-3)

    G = np.identity(4)
    C[0] = np.matrix(
        [(x[0, 0] / (x[0, 0] ** 2 + x[0, 2] ** 2) ** (1 / 2), 0, x[0, 2] / (x[0, 0] ** 2 + x[0, 2] ** 2) ** (1 / 2), 0),
         (-x[0, 2] / (x[0, 0] ** 2 + x[0, 2] ** 2), 0, x[0, 0] / (x[0, 0] ** 2 + x[0, 2] ** 2), 0)])

    nees = np.zeros((n_iter, 1))
    nis = np.zeros((n_iter, 2, 1))
    x_nees = np.zeros((n_iter, 4))
    h = np.zeros((n_iter, 2, 1))

    tanis = np.zeros((n_iter, 1))

    for k in range(1, n_iter):
        xhatminus[k] = np.dot(A, xhat[k - 1, :]) + w[k - 1]

        Pminus[k] = np.dot(A, P[k - 1]).dot(A.T) + np.dot(G, Q).dot(G.T)  # or plus Q instead of matrix dot

        C[k] = np.matrix([(x[k, 0] / (x[k, 0] ** 2 + x[k, 2] ** 2) ** (1 / 2), 0,
                           x[k, 2] / (x[k, 0] ** 2 + x[k, 2] ** 2) ** (1 / 2), 0),
                          (-x[k, 2] / (x[k, 0] ** 2 + x[k, 2] ** 2), 0, x[k, 0] / (x[k, 0] ** 2 + x[k, 2] ** 2), 0)])
        # measurement update
        inv = np.linalg.inv(np.dot(C[k], Pminus[k]).dot(C[k].T) + R)
        K[k] = np.dot(Pminus[k], C[k].T).dot(inv)

        h[k, 0] = np.sqrt(np.dot(xhatminus[k, (0, 2)].T, xhatminus[k, (0, 2)]))
        h[k, 1] = np.arctan2(xhatminus[k, 1], xhatminus[k, 0])

        xhat[k] = xhatminus[k] + np.dot(K[k], (z[k, :] - h[k, :].T).T)

        P[k] = np.dot((np.identity(4) - np.dot(K[k], C[k])), Pminus[k])

        x_ness = x - xhat[k, :, 0]
        nees[k] = np.dot(x_nees[k].T, np.linalg.inv(P[k])).dot(x_nees[k])
        nis[k, :, 0] = np.dot((z[k, 0] - np.dot(C[k], xhatminus[k])).T, inv).dot(z[k] - np.dot(C[k], xhatminus[k]))

    nees = nees / 1507
    nis = nis / 1507
    tanis = np.mean(nis)
    print('TA-NIS =', tanis)
    print(x - xhat[:, :, 0])


if __name__ == "__main__":
    single_simulation_constant_velocity_model(10, matched=True, piecewise=True,
                                              model='Single Simulation Constant velocity piecewise', plot=True)
    print("Magic begins here")
