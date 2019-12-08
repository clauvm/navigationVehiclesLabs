"""
Kalman filter implementation was based on paper "Implementation of Kalman Filter with Python Language"
Mohamed LAARAIEDH
IETR Labs, University of Rennes 1
Mohamed.laaraiedh@univ-rennes1.fr

Code was adapted to follow the notation of the course assignment
"""
from numpy.linalg import inv


import matplotlib.pyplot as plt
import numpy as np

from kalman_filter.statsUtils import gauss_pdf

number_of_samples = 50
x = 0.26578


def process_equation(previous_state, A, B, G, control_input, process_noise):
    """
    Desbribe the process equation
    :param previous_state:
    :param A:
    :param B:
    :param G:
    :param control_input:
    :param process_noise:
    :return:
    """
    return A * previous_state + B * control_input + G * process_noise


def generate_data():
    """
    Generates data for the measurements, 50 records subject to a noise that follows a normal
    probability distribution mu = 0 and standard deviation of 0.1, this function also plots
    the data
    :return: measurements
    """
    # We will choose a scalar constant

    mu = 0
    sigma = 0.1
    error = np.random.normal(mu, sigma, number_of_samples)
    measurements = error + x
    mean = np.mean(measurements)
    print("measurements")
    print(measurements)
    print("np mean")
    print(mean)
    plt.plot(measurements, '+', color='r', label='measurements')
    plt.axhline(mean, color='black', linestyle='--', label='mean value of measurements')
    plt.ylabel('voltage [V]')
    plt.xlabel('Measurement')
    plt.legend()
    plt.show()
    return measurements


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
    x_hat = np.dot(A, x_hat_previous) + np.dot(B, control_input)
    P_hat = np.dot(A, np.dot(P_previous, A.T)) + Q
    return (x_hat, P_hat)


def update_step(x_hat, P_hat, Z, C, R):
    """
    Computes the posterior mean X and covariance P of the system state given a new measurement at time step k
    This is the measurement update phase or the corrector
    :param x_hat: predicted mean(x_hat)
    :param P_hat: predicted covariance (P_hat)
    :param Y: measurement vector
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


def plot_measurements_estimates(Z, filter_estimate):
    plt.plot(Z, '+', color='r', label='measurements')
    plt.plot(filter_estimate, 'b', color='blue', label='filter estimate')
    plt.axhline(x, color='green', linestyle='--', label='true value')
    plt.ylabel('voltage [V]')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()


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
    plt.xlabel('Iteration')
    plt.title("Kalman Gain (K)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    """
    We want to estimate a scalar x, we can take measurements of that constant
    but these measurements are corrupted by a noise of 0.01
    """
    noise = 0.01
    """
    Our process equation normally is:
    x(k+1)= Ax(k)+Bu(k)+Gw(k),
    but in this case it will be:
    x(k+1)=x(k)+w(k)
    """
    """
    Our measurement (observation) model normally is:
    z(k) = Cx(k) + Hv(k)
    but in this case, it will be
    z(k) = x(k) + v(k) 
    """
    """
    We can see that the state does not change from step to step
    therefore A = 1, there is no control input so u = 0. The noise
    measurement is of the state directly so C = 1. There is no Input Control so B and U are 0
    """
    C = np.ones((1, 1))
    print("C shape: ", C.shape)
    A = np.ones((1, 1))
    print("A shape: ", A.shape)
    B = np.zeros((1, 1))
    print("B shape: ", B.shape)
    U = np.zeros((1, 1))
    print("U shape: ", U.shape)

    """
    Regarding the noise we will set Q = 10^-5 and R = 0.001
    """
    Q = np.array([[0.00001]])
    print("Q shape: ", Q.shape)
    R = np.array([[0.001]])
    print("R shape: ", R.shape)

    """
    We can assume that the true value of the random constant follows
    a standard normal probability distribution we will make the filter guess that
    the constant is 0 so x_hat in time 0 = 0
    """

    """
    Regarding the Time update equations or the predictors,
    We will assume that the initial prediction (x_hat in time 0) is 0
    We will assume that initial error covariance (P in time 0) is 1
    """

    X = np.array([[0]])
    print("X shape: ", X.shape)
    P = np.array([[1]])
    print("P shape: ", P.shape)

    """
    In this step, we will generate the data
    """
    Z = generate_data()
    print("Z shape: ", Z.shape)

    N_iteration = 50
    filter_estimate = []
    ise_error = []
    mse_error = []
    covariance = []
    kalman_gain = [0]
    for i in range(number_of_samples):
        (X, P) = prediction_step(A, X, B, U, Q, P)
        print("Time step: ", i)
        print("X_HAT: ", X)
        print("MEASSURED", Z[i])
        filter_estimate.append(X[0][0])
        ise_error.append(get_ise_error(Z[i], X[0][0]))
        # mse_error.append(get_mean_square_error2(Z[i], X[0][0], i + 1))
        mse_error.append(get_mean_square_error(Z[:i + 1], filter_estimate, i + 1))
        (X, P, K, IM, IS, LH) = update_step(X, P, Z[i].reshape(1, 1), C, R)
        covariance.append(P[0][0])
        kalman_gain.append((K[0][0]))
    plot_measurements_estimates(Z, filter_estimate)
    plot_errors(ise_error, mse_error)
    plot_covariance(covariance)
    plot_kalman_gain(kalman_gain)
    print("Finish iterations")
