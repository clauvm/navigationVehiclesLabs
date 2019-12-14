import matplotlib.pyplot as plt
import numpy as np


def get_ise_error(measured, estimated):
    return (measured - estimated) ** 2


def get_mean_square_error(measured, estimated, n):
    # return ((measured - estimated) ** 2).mean()
    suma = 0
    for i in range(n):
        suma += ((measured[i] - estimated[i]) ** 2)
    return suma / n


def get_mean_square_error2(measured, estimated, n):
    suma = 0
    for i in range(n):
        suma += ((measured - estimated) ** 2)
    return suma / n


def plot_errors(ise_error, mse_error):
    plt.plot(ise_error, 'b', color='blue', label='ise')
    plt.plot(mse_error, 'b', color='red', label='mse')
    plt.ylabel('Spread Error [V^2]')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()


def plot_covariance(covariance):
    plt.plot(covariance, 'b', color='blue')
    plt.ylabel('Covariance')
    plt.xlabel('Iteration')
    plt.title("Estimate error covariance (P)")
    plt.legend()
    plt.show()


def plot_kalman_gain(kalman_gain, title=""):
    plt.plot(kalman_gain, 'b', color='blue')
    plt.ylabel('gain')
    plt.ylim((0, 1))
    plt.xlabel('Time')
    plt.title("Kalman Gain (K) {0}".format(title))
    plt.legend()
    plt.show()


def plot_ness(ness, q, y_bottom_lim=0, y_top_lim=10, interval_bottom=6, interval_top=6):
    a = len(list(filter(lambda x: x > 6, ness)))
    print("greater than acceptable interval")
    print(a)
    plt.plot(ness, 'b', color='blue')
    plt.ylabel('NEES')
    plt.ylim((y_bottom_lim, y_top_lim))
    if interval_bottom:
        plt.axhline(y=interval_bottom)
    if interval_top:
        plt.axhline(y=interval_top)
    plt.xlabel('Time')
    plt.title("NEES ERROR Q={0}".format(q))
    plt.legend()
    plt.show()


def plot_nis(nis, q, y_bottom_lim=0, y_top_lim=10, interval_bottom=6, interval_top=6):
    a = len(list(filter(lambda x: x > 6, nis)))
    print("a")
    print(a)
    plt.plot(nis, 'b', color='blue')
    plt.ylabel('NIS')
    plt.ylim((y_bottom_lim, y_top_lim))
    if interval_bottom:
        plt.axhline(y=interval_bottom)
    if interval_top:
        plt.axhline(y=interval_top)
    plt.xlabel('Time')
    plt.title("NIS ERROR Q={0}".format(q))
    plt.legend()
    plt.show()


def plot_error(error, q, x_label, y_label, title, y_bottom_lim=0, y_top_lim=10, interval_bottom=6, interval_top=6,
               lines=True):
    # a = len(list(filter(lambda x: x > 6, error)))
    # print("greater than acceptable interval")
    # print(a)
    plt.plot(error, 'b', color='blue')
    plt.ylabel(y_label)
    plt.ylim((y_bottom_lim, y_top_lim))
    if lines:
        if interval_bottom:
            plt.axhline(y=interval_bottom)
        if interval_top:
            plt.axhline(y=interval_top)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_position_velocity(position_true, velocity_true, position_estimate, velocity_estimate, Q):
    plt.plot(position_true, velocity_true, 'b', color='blue')
    plt.plot(position_estimate, velocity_estimate, '--', color='red')
    plt.ylabel('Velocity (m/s)')
    # plt.ylim((0, 1))
    plt.xlabel('Position x[m]')
    plt.title("State trajectory Q={0}".format(Q))
    plt.legend()
    plt.show()


def plot_position_velocity_acceleration(position_true, velocity_true, acceleration_true, position_estimate,
                                        velocity_estimate, acceleration_estimate, Q):
    plt.plot(position_true, velocity_true, acceleration_true, 'b', color='blue')
    plt.plot(position_estimate, velocity_estimate, acceleration_estimate, '--', color='red')
    plt.ylabel('Velocity (m/s)')
    # plt.ylim((0, 1))
    plt.xlabel('Position x[m]')
    plt.title("State trajectory Q={0}".format(Q))
    plt.legend()
    plt.show()


def NESS(x, x_hat, p):
    x_tilde = x - x_hat
    p_times_x_tilde = np.matmul(np.linalg.inv(p), x_tilde)
    # return np.matmul(x_tilde.T, p_times_x_tilde)
    return np.linalg.multi_dot([x_tilde.T, np.linalg.inv(p), x_tilde])


def NIS(C, P, Z, x_hat, H, R):
    z_tilde = Z - np.matmul(C, x_hat)
    s = np.linalg.multi_dot([C, P, C.T]) + np.linalg.multi_dot([H, R, H.T])
    return np.linalg.multi_dot([z_tilde.T, np.linalg.inv(s), z_tilde])


def SAC(C, Z_k, Z_j, x_hat_k, x_hat_j):
    z_tilde_k = Z_k - np.matmul(C, x_hat_k)
    z_tilde_j = Z_j - np.matmul(C, x_hat_j)
    sum1 = np.matmul(z_tilde_k.T, z_tilde_j)[0][0]
    sum2 = np.matmul(z_tilde_k.T, z_tilde_k)[0][0]
    sum3 = np.matmul(z_tilde_j.T, z_tilde_j)[0][0]
    return sum1, sum2, sum3


def TA_NIS(C, Z, x_hat, P, H, R):
    sum = 0
    K = len(Z)
    for k in range(K):
        z_tilde = Z[k] - np.matmul(C, x_hat[k])
        s_k = np.linalg.multi_dot([C, P[k], C.T]) + np.linalg.multi_dot([H, R, H.T])
        sum += np.linalg.multi_dot([z_tilde.T, np.linalg.inv(s_k), z_tilde])
    return sum / K


def TA_AC(C, Z, x):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    K = len(Z)
    for k in range(K):
        if k + 1 < K:
            z_tilde_k = Z[k] - np.matmul(C, x[k])
            z_tilde_j = Z[k + 1] - np.matmul(C, x[k + 1])
            sum1 += np.matmul(z_tilde_k.T, z_tilde_j)[0][0]
            sum2 += np.matmul(z_tilde_k.T, z_tilde_k)[0][0]
            sum3 += np.matmul(z_tilde_j.T, z_tilde_j)[0][0]

    return sum1 * ((sum2 * sum3) ** (-1 / 2))
