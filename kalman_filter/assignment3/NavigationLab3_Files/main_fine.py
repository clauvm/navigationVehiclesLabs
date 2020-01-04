import numpy as np

from numpy.linalg import inv, det
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2


def generate_data_2D_fun_fil(Q1, Q2, R1, R2):
    nSegments = 5
    points = np.array([[200, -100],
                       [100, 100],
                       [100, 300],
                       [-200, 300],
                       [-200, -200],
                       [0, 0]], dtype=float)

    dp = np.diff(points, axis=0)
    dist = dp ** 2

    dist = np.round(np.sqrt(dist[:, 0] + dist[:, 1]))  # distance
    ang = np.arctan2(dp[:, 1], dp[:, 0])  # orientation
    ang = np.array([ang]).T

    NumberOfDataPoints = int(np.sum(dist))
    print("Number Of DATA Points")
    print(NumberOfDataPoints)

    T = 0.5  # [s] Sampling time interval

    v_set = 2 * np.hstack((np.cos(ang), np.sin(ang)))

    idx = 0
    v = np.kron(np.ones((int(dist[idx]), 1)), v_set[idx, :])
    for idx in range(1, nSegments):
        v = np.vstack((v, np.kron(np.ones((int(dist[idx]), 1)), v_set[idx, :])))

    # ==motion generation====================================================

    A = np.array([[1, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0]], dtype=float)

    B = np.array([[T, 0],
                  [1, 0],
                  [0, T],
                  [0, 1]], dtype=float)

    G = np.array([[T ** 2 / 2, 0],
                  [T, 0],
                  [0, T ** 2 / 2],
                  [0, T]], dtype=float)

    w_x = np.random.normal(0.0, np.sqrt(Q1), NumberOfDataPoints)  # noise in x-direction
    w_y = np.random.normal(0.0, np.sqrt(Q2), NumberOfDataPoints)  # noise in y-direction

    w = np.hstack((np.array([w_x]).T, np.array([w_x]).T))
    x = np.zeros((NumberOfDataPoints, 4))
    x[0, :] = [200, 0, -100, 0]
    for idx in range(1, int(NumberOfDataPoints)):
        x[idx, :] = np.dot(A, np.array(x[idx - 1, :])) + np.dot(B, v[idx, :]) + np.dot(G, w[idx, :])

    true_data = x  # 2D data: [px; vx; py; vy]

    # ==measurement generation===============================================
    position = x[:, (0, 2)]  # 2D position data

    # distance and orientation with respect to the origin
    z = np.zeros((NumberOfDataPoints, 2))
    for idx in range(0, int(NumberOfDataPoints)):
        z[idx, 0] = np.sqrt(np.dot(position[idx, :], position[idx, :]))
        z[idx, 1] = np.arctan2(position[idx, 1], position[idx, 0])

    # unwrap radian phases by changing absolute jumps greater than pi to their 2*pi complement
    z[:, 1] = np.unwrap(z[:, 1])

    v_meas = np.vstack(
        (np.random.normal(0.0, np.sqrt(R1), NumberOfDataPoints),
         np.random.normal(0.0, np.sqrt(R2), NumberOfDataPoints))).T
    z_exact = z
    z = z + v_meas  # add measurement noise

    # == plots ============================
    f1 = plt.figure()
    plt.plot(x[:, 0], x[:, 2], label='linear')
    plt.xlabel('x-axis [m]')
    plt.ylabel('y-axis [m]')
    plt.savefig('xy.pdf')
    f1.show()

    xlab = [[' '], ['Time step [s]']]
    ylab = [['r [m]'], ['$\theta$ [rad]']]
    f2 = plt.figure()
    for idx in range(0, 2):
        plt.subplot(2, 1, idx + 1)
        line_z, = plt.plot(z[:, idx], label='linear')
        line_ze, = plt.plot(z_exact[:, idx], label='linear')
        plt.xlabel(xlab[idx])
        plt.ylabel(ylab[idx])
        plt.legend([line_z, line_ze], ['Measured', 'Exact'], fancybox=True, framealpha=0.0, loc='lower center', ncol=2)
        # leg.get_frame().set_linewidth(0.0)
    plt.savefig('r_th.pdf')
    f2.show()
    return z, x


def NESS_fil(x, x_hat, p):
    x_tilde = x - x_hat
    # p_times_x_tilde = np.matmul(np.linalg.inv(p), x_tilde)
    # return np.matmul(x_tilde.T, p_times_x_tilde)
    return np.linalg.multi_dot([x_tilde.T, np.linalg.inv(p), x_tilde])


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


def update_step_extended(x_hat, P_hat, Z, C, R):
    """
    Computes the posterior mean X and covariance P of the system state given a new measurement at time step k
    This is the measurement update phase or the corrector for extended kalman filter
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

    h1 = np.sqrt(np.dot(x_hat[[0, 2]].T, x_hat[[0, 2]]).astype(float))  # Compute derivatives
    h2 = np.arctan2(x_hat[1].astype(float), x_hat[0].astype(float))  # Compute derivatives
    h = np.array([h1, h2]).astype(float)

    X = x_hat + np.dot(K, (Z - h.T).T)
    P = np.dot((np.identity(4) - np.dot(K, C)), P_hat)
    LH = 0
    return (X, P, K, IM, IS, LH)


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
    number_of_samples = 1507
    R_1 = np.array([[10]])
    R_2 = np.array([[1e-3]])
    Q_1 = np.array([[10]])
    Q_2 = np.array([[10]])
    nis_array = np.zeros((number_of_samples, 2, 1))
    tanis = np.zeros((number_of_samples, 1))

    T = 0.5

    Q_final = np.array([[(T ** 3) / 3, (T ** 2) / 2, 0, 0],
                        [(T ** 2) / 2, T, 0, 0],
                        [0, 0, (T ** 3) / 3, (T ** 2) / 2],
                        [0, 0, (T ** 2) / 2, T]], dtype=float) * q

    Z, X_true = generate_data_2D_fun_fil(Q_1[0][0], Q_2[0][0], R_1[0][0], R_2[0][0])
    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    B = np.array([[0], [0]])
    U = np.array([[0], [0]])
    w = np.zeros((number_of_samples, 4, 1))
    R = np.array([[R_1[0][0], 0], [0, R_2[0][0]]])
    x_hat = []
    P_hat = []
    ness_arr = []
    kalman_gain = []
    normal_random_dis = np.random.multivariate_normal([0, 0, 0, 0], Q_final, number_of_samples)
    w[:, :, 0] = normal_random_dis
    X = np.array([[0], [0], [0], [0]])
    P = np.linalg.inv(np.array([[1.0, 2, 2, 1], [2, 3, 3, 2], [2, 3, 1, 1], [1, 2, 1, 1]]))

    for i in range(number_of_samples):
        (X, P) = prediction_step_extended(A, X, B, U, Q_final, P, w[i])
        x_hat.append(X.astype(float))
        P_hat.append(P)
        C = np.array([[X_true[i, 0] / (X_true[i, 0] ** 2 + X_true[i, 2] ** 2) ** (1 / 2), 0,
                       X_true[i, 2] / (X_true[i, 0] ** 2 + X_true[i, 2] ** 2) ** (1 / 2), 0],
                      [-X_true[i, 2] / (X_true[i, 0] ** 2 + X_true[i, 2] ** 2), 0,
                       X_true[i, 0] / (X_true[i, 0] ** 2 + X_true[i, 2] ** 2), 0]])
        x_true_i = np.array([[X_true[i][0]], [X_true[i][1]], [X_true[i][2]], [X_true[i][3]]])
        print("i: ", i)
        ness_arr.append(NESS_fil(x_true_i, X, P).reshape(1)[0])
        inv_mess = np.linalg.inv(np.dot(C, P).dot(C.T) + R)
        nis_array[i, :, 0] = np.dot((Z[i, 0] - np.dot(C, x_hat[0])).T, inv_mess).dot(Z[i] - np.dot(C, x_hat[0]))
        (X, P, K, IM, IS, LH) = update_step_extended(X, P, Z[i], C, R)
        kalman_gain.append(K)
    nis_array = nis_array / 1507
    tanis_array = np.mean(nis_array)
    return np.array(ness_arr), np.array(nis_array), np.array(tanis_array), np.array(x_hat), np.array(X_true), np.array(
        kalman_gain), Z


def plot_relation_val_and_xhat_in_p(val, X_true):
    plt.figure()
    plt.plot(val[:, 0], color='b', label='x_predicted')
    plt.plot(X_true[:, 0], color='r', linestyle='dashed', label='x_true')

    plt.legend()
    plt.xlabel('Step')
    plt.title('x_predicted vs x_true in position x')
    plt.show()


def plot_relation_val_and_xhat_in_v(val, X_true):
    plt.figure()
    plt.plot(val[:, 1], color='b', label='x_predicted')
    plt.plot(X_true[:, 1], color='r', linestyle='dashed', label='x_true')

    plt.legend()
    plt.xlabel('Step')
    plt.title('x_predicted vs x_true in vel x')
    plt.show()


def plot_relation_val_and_xhat_in_py(val, X_true):
    plt.figure()
    plt.plot(val[:, 2], color='b', label='x_predicted')
    plt.plot(X_true[:, 2], color='r', linestyle='dashed', label='x_true')

    plt.legend()
    plt.xlabel('Step')
    plt.title('x_predicted vs x_true in position y')
    plt.show()


def plot_relation_val_and_xhat_in_vy(val, X_true):
    plt.figure()
    plt.plot(val[:, 3], color='b', label='x_predicted')
    plt.plot(X_true[:, 3], color='r', linestyle='dashed', label='x_true')

    plt.legend()
    plt.xlabel('Step')
    plt.title('x_predicted vs x_true in vel y')
    plt.show()


def plot_kalman_gain(kalman_gain):
    plt.figure()
    plt.plot(kalman_gain[:, 0, 0], label='gain')
    plt.title('Q=0', fontweight='bold')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('K1')
    plt.setp(plt.gca(), 'ylim', [0, 1.8])
    plt.show()

    plt.figure()
    plt.plot(kalman_gain[:, 1, 0], label='gain')
    plt.title('Q=0', fontweight='bold')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('K2')
    plt.setp(plt.gca(), 'ylim', [-0.2, 1])
    plt.show()

    plt.figure()
    plt.plot(kalman_gain[:, 2, 0], label='gain')
    plt.title('Q=0', fontweight='bold')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('K3')
    plt.setp(plt.gca(), 'ylim', [0, 1])
    plt.show()


def plot_ness(ness, number_of_samples):
    chi_squared = np.zeros(number_of_samples)
    chi_squared = chi_squared + chi2.ppf(0.95, df=4)
    plt.figure()
    plt.plot(chi_squared, color='b', linestyle='dashed', label='chi-squared')
    plt.plot(ness, color='r', label='NEES')
    plt.legend()
    plt.title('NEES for Q=0', fontweight='bold')
    plt.xlabel('Step')
    plt.ylabel('NEES values')
    plt.show()

def plot_nis(nis):
    plt.figure()
    plt.plot(nis[:,1],color='r', label='NIS')
    #plt.plot(chi2, color='r', linestyle='dashed',label='chi-squared')
    plt.legend()
    plt.title('NIS for Q=0', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('NIS values')
    plt.show()


if __name__ == "__main__":
    number_of_samples = 1507
    ness_arr, nis_array, tanis_array, x_hat, X_true, kalman_gain, Z = single_simulation_constant_velocity_model(0.001,
                                                                                                                matched=True,
                                                                                                                piecewise=True,
                                                                                                                model='Single Simulation Constant velocity piecewise',
                                                                                                                plot=False)
    print("Kalman gain")
    print(kalman_gain)

    # x_hat vs X_true
    plot_relation_val_and_xhat_in_p(x_hat, X_true)
    plot_relation_val_and_xhat_in_v(x_hat, X_true)
    plot_relation_val_and_xhat_in_py(x_hat, X_true)
    plot_relation_val_and_xhat_in_vy(x_hat, X_true)
    #
    plot_kalman_gain(kalman_gain)
    plot_ness(ness_arr, number_of_samples)
    plot_nis(nis_array)


