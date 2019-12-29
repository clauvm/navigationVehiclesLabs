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

    h1 = np.sqrt(np.dot(x_hat[[0, 2]].T, x_hat[[0, 2]]).astype(float))
    h2 = np.arctan2(x_hat[1].astype(float), x_hat[0].astype(float))
    h = np.array([h1, h2]).astype(float)

    X = x_hat + np.dot(K, (Z - h.T).T)
    P = np.dot((np.identity(4) - np.dot(K, C)), P_hat)
    LH = 0
    return (X, P, K, IM, IS, LH)


def prediction_step_extended(A, x_hat_previous, B, control_input, Q, P_previous, w):
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
    x_hat = np.dot(A, x_hat_previous) + w
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


def single_simulation_constant_velocity_model(q=0.001, matched=True, piecewise=False, plot=False, model=''):
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
    numstates = 6
    # px, vx, py, vy = symbols('p_x v_x p_y v_y')
    #
    # H = Matrix([sqrt(px ** 2 + py ** 2)])
    # state = Matrix([[px, vx, py, vy]])
    # H.jacobian(state)

    x, x_vel, y = symbols('x, x_vel y')

    Ha = Matrix([sqrt(x ** 2 + y ** 2)])

    state = Matrix([x, x_vel, y])
    a = Ha.jacobian(state)
    print(a)

    T = 0.5
    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    B = np.array([[0], [0]])
    U = np.array([[0], [0]])
    w = np.zeros((number_of_samples, 4, 1))

    G = np.array([[T ** 2 / 2], [T]])
    H = np.eye(2)
    R_1 = np.array([[10]])
    R_2 = np.array([[1e-3]])
    Q_1 = np.array([[10]])
    Q_2 = np.array([[10]])

    Q_final = np.array([[(T ** 3) / 3, (T ** 2) / 2, 0, 0],
                        [(T ** 2) / 2, T, 0, 0],
                        [0, 0, (T ** 3) / 3, (T ** 2) / 2],
                        [0, 0, (T ** 2) / 2, T]], dtype=float) * q
    R = np.array([[R_1[0][0], 0], [0, R_2[0][0]]])
    normal_random_dis = np.random.multivariate_normal([0, 0, 0, 0], Q_final, number_of_samples)
    w[:, :, 0] = normal_random_dis

    Z, X_true = generate_data_2D_fun(Q_1[0][0], Q_2[0][0], R_1[0][0], R_2[0][0])

    X = np.array([[0], [0], [0], [0]])
    P = np.linalg.inv(np.array([[1.0, 2, 2, 1], [2, 3, 3, 2], [2, 3, 1, 1], [1, 2, 1, 1]]))
    filter_estimate = []
    kalman_gain = []
    ness_arr = []
    nis_arr = []
    ta_nis = []
    x_hat = []
    P_hat = []
    first_time = True
    for i in range(number_of_samples):
        (X, P) = prediction_step_extended(A, X, B, U, Q_final, P, w[i])
        x_hat.append(X.astype(float))
        P_hat.append(P)
        C = np.array([[X_true[i, 0] / (X_true[i, 0] ** 2 + X_true[i, 2] ** 2) ** (1 / 2), 0,
                       X_true[i, 2] / (X_true[i, 0] ** 2 + X_true[i, 2] ** 2) ** (1 / 2), 0],
                      [-X_true[i, 2] / (X_true[i, 0] ** 2 + X_true[i, 2] ** 2), 0,
                       X_true[i, 0] / (X_true[i, 0] ** 2 + X_true[i, 2] ** 2), 0]])
        # nis_arr.append(NIS(C, P, Z[i], X, H, R).reshape(1)[0])
        x_true_i = np.array([[X_true[i][0]], [X_true[i][1]], [X_true[i][2]], [X_true[i][3]]])
        print("i: ", i)
        ness_arr.append(NESS(x_true_i, X, P).reshape(1)[0])
        # ta_nis.append(TA_NIS(C, Z[:i + 1], x_hat, P_hat, H, R)[0][0])
        (X, P, K, IM, IS, LH) = update_step_extended(X, P, Z[i], C, R)
        kalman_gain.append(K)

    # Gains
    # position_gain = np.array(kalman_gain)[:, 0:1, :].reshape(number_of_samples)
    # velocity_gain = np.array(kalman_gain)[:, 1:2, :].reshape(number_of_samples)
    # X values
    # position_true = X_true[0, :]
    # velocity_true = X_true[1, :]
    if plot:
        # plot_kalman_gain(position_gain, "position")
        # plot_kalman_gain(velocity_gain, "velocity")
        # plot_position_velocity(position_true, velocity_true, np.array(position_estimate[0:-1]),
        #                        np.array(velocity_estimate[0:-1]), Q[0][0])
        plot_ness(ness_arr, q, matched=matched, model=model)
        plot_nis(nis_arr, q, matched=matched, model=model)
    return ness_arr, nis_arr, ta_nis, np.array(x_hat), np.array(X_true)


if __name__ == "__main__":
    ness_arr, nis_arr, ta_nis, x_hat, X_true = single_simulation_constant_velocity_model(10, matched=True,
                                                                                         piecewise=True,
                                                                                         model='Single Simulation Constant velocity piecewise',
                                                                                         plot=False)
    print("")
    print("Magic begins here")
