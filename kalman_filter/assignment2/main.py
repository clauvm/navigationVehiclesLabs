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
from assignment2.files.gen_data16_fun import get_generated_data_eq_16
from assignment2.utils import NIS, NESS, plot_ness, plot_nis, plot_kalman_gain, plot_position_velocity, \
    plot_position_velocity_acceleration, SAC, plot_error, TA_NIS, TA_AC

number_of_samples = 100
number_of_runs_monte_carlo = 50

def gauss_pdf(X, M, S):
    """
    Computes the Predictive probability (likelihood) of measurement
    :param X: measurement vector
    :param M: Mean of predictive distribution of measurement vector
    :param S: the Covariance or predictive mean of measurement vector
    :return: Gaussian probability distribution
    """
    if M.shape[1] == 1:
        DX = X - np.tile(M, X.shape[1])
        E = 0.5 * np.sum(DX * (np.dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(det(S))
        P = np.exp(-E)
    elif X.shape[1] == 1:
        DX = np.tile(X, M.shape[1]) - M
        E = 0.5 * np.sum(DX * (np.dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(det(S))
        P = np.exp(-E)
    else:
        DX = X - M
        E = 0.5 * np.dot(DX.T, np.dot(inv(S), DX))
        E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(det(S))
        P = np.exp(-E)
    return (P[0], E[0])


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


def single_simulation_constant_velocity_model(q, matched=True, piecewise=False, plot=False, model=''):
    """
    Constant (or piecewise) velocity white noise accelleration eq 13 and 14
    :param piecewise: whether data should be generated with equations 13 or 14
    :param q:
    """
    T = 1.0
    A = np.array([[1, T], [0, 1]])
    B = np.array([[0], [0]])
    U = np.array([[0], [0]])
    C = np.array([[1, 0]])
    G = np.array([[T ** 2 / 2], [T]])
    H = np.array([[1]])
    R = np.array([[1]])
    Q = np.array([[q]])
    Q_try = np.array([[T ** 3 / 3, T ** 2 / 2],
                      [T ** 2 / 2, T]], dtype=float) * q
    Q_try_2 = np.array([[T ** 3 / 3, T ** 2 / 2],
                        [T ** 2 / 2, T]], dtype=float) * 9
    Z, X_true = get_generated_data_eq_13(Q[0][0] if matched else 9,
                                         R[0][0]) if not piecewise else get_generated_data_eq_14(
        Q[0][0] if matched else 9,
        R[0][0])

    X = np.array([[Z[0]], [(Z[1] - Z[0]) / T]])
    position_estimate = [X[0][0]]
    velocity_estimate = [X[1][0]]
    P = np.linalg.inv(np.array([[R[0][0], R[0][0] / T], [R[0][0] / T, (2 * R[0][0]) / (T ** 2)]]))
    filter_estimate = []
    kalman_gain = []
    ness_arr = []
    nis_arr = []
    sac_arr = []
    aux_1 = []
    aux_2 = []
    ta_nis = []
    ta_ac = []
    x_hat = []
    P_hat = []
    first_time = True
    for i in range(number_of_samples):
        (X, P) = prediction_step(A, X, B, U, Q_try if matched else Q_try, P)
        x_hat.append(X)
        P_hat.append(P)
        nis_arr.append(NIS(C, P, Z[i], X, H, R).reshape(1)[0])
        position_estimate.append(X[0][0])
        velocity_estimate.append(X[1][0])
        if first_time:
            aux_1.append(X)
            aux_2.append(Z[i])
            first_time = False
        else:
            r1, r2, r3 = SAC(C, aux_2[0], Z[i], aux_1[0], X)
            if 2 <= len(x_hat) <= len(Z) - 1:
                tac = TA_AC(C, Z[:len(x_hat)], x_hat)
                ta_ac.append(tac)

            sac_arr.append([r1, r2, r3])
            aux_1 = [X]
            aux_2 = [Z[i]]

        x_true_i = np.array([[X_true[0][i]], [X_true[1][i]]])
        ness_arr.append(NESS(x_true_i, X, P).reshape(1)[0])
        ta_nis.append(TA_NIS(C, Z[:i + 1], x_hat, P_hat, H, R)[0][0])
        (X, P, K, IM, IS, LH) = update_step(X, P, Z[i].reshape(1, 1), C, R)
        kalman_gain.append(K)

    # print("Kalman gain Q{0}".format(Q[0][0]))
    # print(kalman_gain)
    # Gains
    position_gain = np.array(kalman_gain)[:, 0:1, :].reshape(number_of_samples)
    velocity_gain = np.array(kalman_gain)[:, 1:2, :].reshape(number_of_samples)
    # X values
    position_true = X_true[0, :]
    velocity_true = X_true[1, :]
    if plot:
        # plot_kalman_gain(position_gain, "position")
        # plot_kalman_gain(velocity_gain, "velocity")
        # plot_position_velocity(position_true, velocity_true, np.array(position_estimate[0:-1]),
        #                        np.array(velocity_estimate[0:-1]), Q[0][0])
        plot_ness(ness_arr, q, matched=matched, model=model)
        plot_nis(nis_arr, q, matched=matched, model=model)
    return ness_arr, nis_arr, sac_arr, ta_nis, ta_ac


def single_simulation_constant_acceleration_model(q, matched=True, piecewise=False, plot=False, model=''):
    """
    Constant (or piecewise) acceleration model eq 15 and 16
    :param piecewise: whether data should be generated with equations 15 or 16
    :param q:
    """
    T = 1.0
    A = np.array([[1, T, T ** 2 / 2],
                  [0, 1, T],
                  [0, 0, 1]], dtype=float)
    B = np.array([[0], [0], [0]])
    U = np.array([[0], [0]])
    C = np.array([[1, 0, 0]])

    H = np.array([[1]])
    R = np.array([[1]])
    Q = np.array([[q]])
    Q_try = np.array([[(T ** 5) / 20, (T ** 4) / 8, (T ** 3) / 6],
                      [(T ** 4) / 8, (T ** 3) / 3, (T ** 2) / 2],
                      [(T ** 3) / 6, (T ** 2) / 2, T]], dtype=float) * q
    Q_try_2 = np.array([[(T ** 5) / 20, (T ** 4) / 8, (T ** 3) / 6],
                        [(T ** 4) / 8, (T ** 3) / 3, (T ** 2) / 2],
                        [(T ** 3) / 6, (T ** 2) / 2, T]], dtype=float) * 9
    Z, X_true = get_generated_data_eq_15(Q[0][0] if matched else 9,
                                         R[0][0]) if not piecewise else get_generated_data_eq_16(
        Q[0][0] if matched else 9,
        R[0][0])

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
    sac_arr = []
    nis_arr = []
    aux_1 = []
    aux_2 = []
    ta_nis = []
    ta_ac = []
    x_hat = []
    P_hat = []
    first_time = True

    for i in range(number_of_samples):
        (X, P) = prediction_step(A, X, B, U, Q_try, P)
        x_hat.append(X)
        P_hat.append(P)
        nis_arr.append(NIS(C, P, Z[i], X, H, R).reshape(1)[0])
        position_estimate.append(X[0][0])
        velocity_estimate.append(X[1][0])
        acceleration_estimate.append([X[2][0]])
        if first_time:
            aux_1.append(X)
            aux_2.append(Z[i])
            first_time = False
        else:
            r1, r2, r3 = SAC(C, aux_2[0], Z[i], aux_1[0], X)
            if 2 <= len(x_hat) <= len(Z) - 1:
                tac = TA_AC(C, Z[:len(x_hat)], x_hat)
                ta_ac.append(tac)
            sac_arr.append([r1, r2, r3])
            aux_1 = [X]
            aux_2 = [Z[i]]
        x_true_i = np.array([[X_true[0][i]], [X_true[1][i]], [X_true[2][i]]])
        ness_arr.append(NESS(x_true_i, X, P).reshape(1)[0])
        ta_nis.append(TA_NIS(C, Z[:i + 1], x_hat, P_hat, H, R)[0][0])
        (X, P, K, IM, IS, LH) = update_step(X, P, Z[i].reshape(1, 1), C, R)
        kalman_gain.append(K)

    # print(kalman_gain)
    # Gains
    position_gain = np.array(kalman_gain)[:, 0:1, :].reshape(number_of_samples)
    velocity_gain = np.array(kalman_gain)[:, 1:2, :].reshape(number_of_samples)
    acceleration_gain = np.array(kalman_gain)[:, 2:3, :].reshape(number_of_samples)

    # X values
    position_true = X_true[0, :]
    velocity_true = X_true[1, :]
    acceleration_true = X_true[2, :]
    if plot:
        # plot_kalman_gain(position_gain, "position")
        # plot_kalman_gain(velocity_gain, "velocity")
        # plot_kalman_gain(acceleration_gain, "acceleration")
        # plot_position_velocity_acceleration(position_true, velocity_true, acceleration_true,
        #                                     np.array(position_estimate[0:-1]),
        #                                     np.array(velocity_estimate[0:-1]), np.array(velocity_estimate[0:-1]),
        #                                     Q[0][0])
        plot_ness(ness_arr, q, matched=matched, model=model)
        plot_nis(nis_arr, q, matched=matched, model=model)
    return ness_arr, nis_arr, sac_arr, ta_nis, ta_ac


def get_res_pj(pj):
    res = np.sum(pj, axis=0)
    return res[0] * ((res[1] * res[2]) ** (-1 / 2))


def get_sac(sac_matrix):
    result = []
    for pj in sac_matrix:
        result.append(get_res_pj(pj))
    return result


def monte_carlo_simulation_constant_velocity_model(q, isVeloConstant=True, number_of_runs=50, matched=True,
                                                   piecewise=False,
                                                   model_name=''):
    ness_matrix = []
    nis_matrix = []
    sac_matrix = []

    for i in range(number_of_runs):
        ness_arr_constant_velo, nis_arr_constant_velo, sac_constant_velo, ta_nis, ta_ac = single_simulation_constant_velocity_model(
            q, matched=matched,
            piecewise=piecewise) if isVeloConstant else single_simulation_constant_acceleration_model(
            q, matched=matched, piecewise=piecewise)
        ness_matrix.append(ness_arr_constant_velo)
        nis_matrix.append(nis_arr_constant_velo)
        sac_matrix.append(sac_constant_velo)
    mean_ness = np.mean(ness_matrix, axis=0)
    mean_nis = np.mean(nis_matrix, axis=0)
    sac_np = np.array(sac_matrix).reshape(99, 50, 3)
    sac_result = np.array(get_sac(sac_np))

    if isVeloConstant:
        plot_error(mean_ness, q, "Time", "NEES", "{1} NEES Q={0}".format(q, model_name), 0, 4, 1.5, 2.6)
        plot_error(mean_nis, q, "Time", "NIS", "{1} NIS Q={0}".format(q, model_name), 0, 2, 0.65, 1.43)
        plot_error(sac_result, q, "Time", "SAC", "{1} SAC Q={0}".format(q, model_name), -0.5, 1.5, -0.277, 0.277)
    else:
        plot_error(mean_ness, q, "Time", "NEES", "{1} NEES Q={0}".format(q, model_name), 2, 4, 2.36, 3.7)
        plot_error(mean_nis, q, "Time", "NIS", "{1} NIS Q={0}".format(q, model_name), 0, 2, 0.65, 1.43)
        plot_error(sac_result, q, "Time", "SAC", "{1} SAC Q={0}".format(q, model_name), -0.5, 1.5, -0.277, 0.277)

    # plot_ness(mean_ness, q, 0, 4, 1.5, 2.6)
    # plot_nis(mean_nis, q, 0, 2, 0.65, 1.43)
    return mean_ness, mean_nis


def real_time_test_simulation_constant_velocity_model(q, isVeloConstant=True, number_of_runs=1, matched=True,
                                                      piecewise=False,
                                                      model_name=''):
    ta_nis_matrix = []

    for i in range(number_of_runs):
        ness_arr_constant_velo, nis_arr_constant_velo, sac_constant_velo, ta_nis, ta_ac = single_simulation_constant_velocity_model(
            q, matched=matched,
            piecewise=piecewise) if isVeloConstant else single_simulation_constant_acceleration_model(
            q, matched=matched, piecewise=piecewise)

        ta_nis_matrix.append(ta_nis)

    np_tanis = np.array(ta_nis)
    np_tac = np.array(ta_ac)
    print("ta_nis {0}: {1}".format(model_name, ta_nis[len(ta_nis) - 1]))
    print("ta_ac {0}: {1}".format(model_name, ta_ac[len(ta_ac) - 1]))

    if isVeloConstant:
        plot_error(np_tanis, q, "Time", "TA_NIS", "{1} TANIS Q={0}".format(q, model_name), 0, 2, 0.74, 1.3,
                   True)
        plot_error(np_tac, q, "Time", "TA_AC", "{1} TA_AC Q={0}".format(q, model_name), -1, 1, -0.196, 0.196,
                   True)
    else:
        plot_error(np_tanis, q, "Time", "TA_NIS", "{1} TA_NIS Q={0}".format(q, model_name), 0, 5, 1.6, 2.16,
                   True)
        plot_error(np_tac, q, "Time", "TA_AC", "{1} TA_AC Q={0}".format(q, model_name), -1, 1, -0.196, 0.196,
                   True)
    return np_tanis


if __name__ == "__main__":
    number_samples_q = 3
    indices = random.sample(range(2, 10), number_samples_q)
    indices_f = np.concatenate((np.array([1]), np.array(indices)), axis=None)

    matched_v = [True, False]
    for matched in matched_v:
        print("MATCHED MODEL = {0} ... ".format(matched))
        for q in indices_f:
            print("VALUE OF q={0}".format(q))
            # Single Simulations
            print("Computing Single Simulation matched model...")
            print("--------------------")
            print("Computing Single Simulation constant velocity eq 13...")

            single_simulation_constant_velocity_model(q, matched=matched,
                                                      piecewise=False, model='Single Simulation Constant velocity',
                                                      plot=True)
            print("Computing Single Simulation constant velocity piecewise eq 14...")
            single_simulation_constant_velocity_model(q, matched=matched, piecewise=True,
                                                      model='Single Simulation Constant velocity piecewise', plot=True)
            print("Computing Single Simulation constant acceleration eq 15...")
            single_simulation_constant_acceleration_model(q, matched=matched, piecewise=False,
                                                          model='Single Simulation Constant acceleration', plot=True)
            print("Computing Single Simulation constant acceleration piecewise eq 16...")
            single_simulation_constant_acceleration_model(q, matched=matched, piecewise=True,
                                                          model='Single Simulation Constant acceleration piecewise',
                                                          plot=True)
            print("--------------------")

            print("Computing Multiple Simulation (Monte Carlo) matched model...")
            print("--------------------")
            # Multiple Simulations (Monte Carlo)
            # Constant velocity piecewise false montecarlo
            print("Computing Multiple Simulation constant velocity eq 13...")
            monte_carlo_simulation_constant_velocity_model(q, True, number_of_runs_monte_carlo, matched, False,
                                                           'Matched={0} Monte Carlo Constant Velocity'.format(matched))
            print("Computing Multiple Simulation constant velocity piecewise eq 14...")
            # Constant velocity piecewise true montecarlo
            monte_carlo_simulation_constant_velocity_model(q, True, number_of_runs_monte_carlo, matched, True,
                                                           'Matched={0} Monte Carlo Constant Velocity piecewise'.format(
                                                               matched))

            # Constant acceleration piecewise false montecarlo
            print("Computing Multiple Simulation constant acceleration eq 15...")
            monte_carlo_simulation_constant_velocity_model(q, False, number_of_runs_monte_carlo, matched, False,
                                                           'Matched={0} Monte Carlo Constant Acceleration'.format(
                                                               matched))
            print("Computing Multiple Simulation constant acceleration piecewise eq 16...")
            monte_carlo_simulation_constant_velocity_model(q, False, number_of_runs_monte_carlo, matched, True,
                                                           'Matched={0} Monte Carlo Constant Acceleration piecewise'.format(
                                                               matched))

            # Real time test
            # Real time test constant velocity piece wise False
            print("Computing Real Time Test constant velocity...")
            real_time_test_simulation_constant_velocity_model(q, True, 1, matched, False,
                                                              'Matched={0} RealTime Constant Velocity'.format(matched))
            # Real time test constant velocity piece wise True
            print("Computing Real Time Test piecewise constant velocity...")
            real_time_test_simulation_constant_velocity_model(q, True, 1, matched, True,
                                                              'Matched={0} RealTime Constant Velocity piecewise'.format(
                                                                  matched))
            # Real time test constant acceleration piece wise False
            print("Computing Real Time Test constant acceleration...")
            real_time_test_simulation_constant_velocity_model(q, False, 1, matched, False,
                                                              'Matched={0} RealTime Constant Acceleration'.format(
                                                                  matched))
            # Real time test constant acceleration piece wise True
            print("Computing Real Time Test constant acceleration piecewise...")
            real_time_test_simulation_constant_velocity_model(q, False, 1, matched, True,
                                                              'Matched={0} RealTime Constant Acceleration'.format(
                                                                  matched))
