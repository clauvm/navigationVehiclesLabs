"""
Kalman filter implementation was based on paper "Implementation of Kalman Filter with Python Language"
Mohamed LAARAIEDH
IETR Labs, University of Rennes 1
Mohamed.laaraiedh@univ-rennes1.fr

Code was adapted to follow the notation of the course assignment
"""
import random

from numpy.linalg import inv

import matplotlib.pyplot as plt
import numpy as np

from assignment2.files.gen_data13_fun import get_generated_data_eq_13
from assignment2.files.gen_data14_fun import get_generated_data_eq_14
from assignment2.files.gen_data15_fun import get_generated_data_eq_15
from assignment2.utils import NIS, NESS, plot_ness, plot_nis, plot_kalman_gain, plot_position_velocity, \
    plot_position_velocity_acceleration
from statsUtils import gauss_pdf

number_of_samples = 100


def prediction_step(A, x_hat_previous, B, control_input, Q, P_previous):
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


def update_step(x_hat, P_hat, Z, C, R):
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
    X = x_hat + np.dot(K, (Z - IM))
    P = P_hat - np.dot(K, np.dot(IS, K.T))
    LH = gauss_pdf(Z, IM, IS)  # the Predictive probability (likelihood)
    return (X, P, K, IM, IS, LH)


# def plot_measurements_estimates(Z, filter_estimate):
#     plt.plot(Z, '+', color='r', label='measurements')
#     plt.plot(filter_estimate, 'b', color='blue', label='filter estimate')
#     plt.axhline(x, color='green', linestyle='--', label='true value')
#     plt.ylabel('voltage [V]')
#     plt.xlabel('Iteration')
#     plt.legend()
#     plt.show()


def single_simulation_constant_velocity_model(q, piecewise=False):
    """
    Constant (or piecewise) velocity white noise accelleration eq 13 and 14
    :param piecewise: whether data should be generated with equations 13 or 14
    :param q:
    """
    T = 1.0
    A = np.array([[1, T], [0, 1]])
    print("A shape: ", A.shape)
    B = np.array([[0], [0]])
    U = np.array([[0], [0]])
    print("B shape: ", B.shape)
    C = np.array([[1, 0]])
    print("C shape: ", C.shape)
    G = np.array([[T ** 2 / 2], [T]])
    print("G shape: ", G.shape)
    H = np.array([[1]])
    R = np.array([[1]])
    Q = np.array([[q]])
    Q_try = np.array([[T ** 3 / 3, T ** 2 / 2],
                      [T ** 2 / 2, T]], dtype=float) * q
    Z, X_true = get_generated_data_eq_13(Q[0][0], R[0][0]) if not piecewise else get_generated_data_eq_14(Q[0][0],
                                                                                                          R[0][0])

    X = np.array([[Z[0]], [(Z[1] - Z[0]) / T]])
    position_estimate = [X[0][0]]
    velocity_estimate = [X[1][0]]
    P = np.linalg.inv(np.array([[R[0][0], R[0][0] / T], [R[0][0] / T, (2 * R[0][0]) / (T ** 2)]]))
    filter_estimate = []
    kalman_gain = []
    ness_arr = []
    nis_arr = []
    for i in range(number_of_samples):
        (X, P) = prediction_step(A, X, B, U, Q_try, P)
        nis_arr.append(NIS(C, P, Z[i], X, H, R).reshape(1)[0])
        position_estimate.append(X[0][0])
        velocity_estimate.append(X[1][0])
        x_true_i = np.array([[X_true[0][i]], [X_true[1][i]]])
        ness_arr.append(NESS(x_true_i, X, P).reshape(1)[0])
        (X, P, K, IM, IS, LH) = update_step(X, P, Z[i].reshape(1, 1), C, R)
        kalman_gain.append(K)

    print("Kalman gain Q{0}".format(Q[0][0]))
    print(kalman_gain)
    # Gains
    position_gain = np.array(kalman_gain)[:, 0:1, :].reshape(number_of_samples)
    velocity_gain = np.array(kalman_gain)[:, 1:2, :].reshape(number_of_samples)
    plot_kalman_gain(position_gain, "position")
    plot_kalman_gain(velocity_gain, "velocity")
    # X values
    position_true = X_true[0, :]
    velocity_true = X_true[1, :]
    plot_position_velocity(position_true, velocity_true, np.array(position_estimate[0:-1]),
                           np.array(velocity_estimate[0:-1]), Q[0][0])
    plot_ness(ness_arr, q)
    plot_nis(nis_arr, q)


def single_simulation_constant_acceleration_model(q, piecewise=False):
    """
    Constant (or piecewise) acceleration model eq 15 and 16
    :param piecewise: whether data should be generated with equations 15 or 16
    :param q:
    """
    T = 1.0
    A = np.array([[1, T, T ** 2 / 2],
                  [0, 1, T],
                  [0, 0, 1]], dtype=float)
    print("A shape: ", A.shape)
    B = np.array([[0], [0], [0]])
    U = np.array([[0], [0]])
    print("B shape: ", B.shape)
    C = np.array([[1, 0, 0]])
    print("C shape: ", C.shape)

    H = np.array([[1]])
    R = np.array([[1]])
    Q = np.array([[q]])
    Q_try = np.array([[(T ** 5) / 20, (T ** 4) / 8, (T ** 3) / 6],
                      [(T ** 4) / 8, (T ** 3) / 3, (T ** 2) / 2],
                      [(T ** 3) / 6, (T ** 2) / 2, T]], dtype=float) * q
    Z, X_true = get_generated_data_eq_15(Q[0][0], R[0][0])

    X = np.array([[Z[0]], [(Z[1] - Z[0]) / T], [0]])
    position_estimate = [X[0][0]]
    velocity_estimate = [X[1][0]]
    acceleration_estimate = [X[2][0]]
    P = np.linalg.inv(np.array(
        [[R[0][0], R[0][0] / T, (2 * R[0][0]) / T ** 2],
         [R[0][0] / T, (2 * R[0][0]) / (T ** 2), (3 * R[0][0]) / (T ** 3)],
         [(2 * R[0][0]) / (T ** 2), (3 * R[0][0]) / (T ** 3), 0]
         ]))
    filter_estimate = []
    kalman_gain = []
    ness_arr = []
    nis_arr = []
    for i in range(number_of_samples):
        (X, P) = prediction_step(A, X, B, U, Q_try, P)
        nis_arr.append(NIS(C, P, Z[i], X, H, R).reshape(1)[0])
        position_estimate.append(X[0][0])
        velocity_estimate.append(X[1][0])
        acceleration_estimate.append([X[2][0]])
        x_true_i = np.array([[X_true[0][i]], [X_true[1][i]], [X_true[2][i]]])
        ness_arr.append(NESS(x_true_i, X, P).reshape(1)[0])
        (X, P, K, IM, IS, LH) = update_step(X, P, Z[i].reshape(1, 1), C, R)
        kalman_gain.append(K)

    print("Kalman gain Q{0}".format(Q[0][0]))
    print(kalman_gain)
    # Gains
    position_gain = np.array(kalman_gain)[:, 0:1, :].reshape(number_of_samples)
    velocity_gain = np.array(kalman_gain)[:, 1:2, :].reshape(number_of_samples)
    acceleration_gain = np.array(kalman_gain)[:, 2:3, :].reshape(number_of_samples)
    plot_kalman_gain(position_gain, "position")
    plot_kalman_gain(velocity_gain, "velocity")
    plot_kalman_gain(acceleration_gain, "acceleration")
    # X values
    position_true = X_true[0, :]
    velocity_true = X_true[1, :]
    acceleration_true = X_true[2, :]
    plot_position_velocity_acceleration(position_true, velocity_true, acceleration_true, np.array(position_estimate[0:-1]),
                           np.array(velocity_estimate[0:-1]), np.array(velocity_estimate[0:-1]), Q[0][0])
    plot_ness(ness_arr, q)
    plot_nis(nis_arr, q)


if __name__ == "__main__":
    number_samples_q = 1
    indices = random.sample(range(1, 10), number_samples_q)
    for q in indices:
        # single_simulation_constant_velocity_model(q, False)
        # single_simulation_constant_velocity_model(q, True)
        single_simulation_constant_acceleration_model(q)
