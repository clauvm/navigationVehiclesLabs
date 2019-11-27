from kalman_filter.statsUtils import update, predict, gaussian_function
import matplotlib.pyplot as plt
import numpy as np


def process_equation(previous_state, A, B, G, control_input, process_noise):
    return A * previous_state + B * control_input + G * process_noise


def generate_data():
    # We will choose a scalar constant
    x = 0.26578
    mu = 0
    sigma = 0.1
    number_of_samples = 50
    noise = np.random.normal(mu, sigma, number_of_samples)
    measurements = noise + x
    mean = np.mean(measurements)
    print("measurements")
    print(measurements)
    print("np mean")
    print(mean)
    plt.plot(measurements,'+',color='r',label='measurements')
    plt.axhline(mean, color='black', linestyle='--',label='mean value of measurements')
    plt.ylabel('voltage [V]')
    plt.xlabel('Measurement')
    plt.legend()
    plt.show()

    pass


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
    measurement is of the state directly so C = 1
    """

    """
    Regarding the noise we will set Q = 10^-5 and R = 0.001
    """
    Q = 0.00001
    R = 0.001

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

    x_hat0 = 0
    P0 = 1

    """
    In this step, we will generate the data
    """
    generate_data()

