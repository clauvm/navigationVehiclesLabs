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


def plot_kalman_gain(kalman_gain):
    plt.plot(kalman_gain, 'b', color='blue')
    plt.ylabel('gain')
    plt.ylim((0, 1))
    plt.xlabel('Time')
    plt.title("Kalman Gain (K)")
    plt.legend()
    plt.show()


def plot_ness(ness, q):
    a = len(list(filter(lambda x: x > 6, ness)))
    print("greater than acceptable interval")
    print(a)
    plt.plot(ness, 'b', color='blue')
    plt.ylabel('NEES')
    plt.ylim((0, 10))
    plt.axhline(y=6)
    plt.xlabel('Time')
    plt.title("NEES ERROR Q={0}".format(q))
    plt.legend()
    plt.show()


def plot_nis(nis, q):
    a = len(list(filter(lambda x: x > 6, nis)))
    print("a")
    print(a)
    plt.plot(nis, 'b', color='blue')
    plt.ylabel('NIS')
    plt.ylim((0, 10))
    plt.axhline(y=6)
    plt.xlabel('Time')
    plt.title("NIS ERROR Q={0}".format(q))
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


def NESS(x, x_hat, p):
    x_tilde = x - x_hat
    p_times_x_tilde = np.matmul(np.linalg.inv(p), x_tilde)
    # return np.matmul(x_tilde.T, p_times_x_tilde)
    return np.linalg.multi_dot([x_tilde.T, np.linalg.inv(p), x_tilde])


def NIS(C, P, Z, x_hat, H, R):
    z_tilde = Z - np.matmul(C, x_hat)
    s = np.linalg.multi_dot([C, P, C.T]) + np.linalg.multi_dot([H, R, H.T])
    return np.linalg.multi_dot([z_tilde.T, np.linalg.inv(s), z_tilde])
