function [x, z, A, Q, G, H] = gen_data13(var_q, R, N)

% This function generates the data for a 1D motion considering a
% Wiener process acceleration model, equation (13)

% var_q is the variance of the process noise
% R is the variance of measurement noise
% N is the number of measurement points
% z is the measured data
% x are the true values of the system states

% example of use
% [x, z] = generate_data(1e-3, 1, 100);

T = 1; %[s] Sampling time interval

x = zeros(2, N); % states are [position; speed; acceleration]
x(:,1) = [0; 10]; % state initialization, change to give your own initial values

A = [1 T; 0 1]; % Transition matrix
Q = var_q * [T^3/3 T^2/2; T^2/2 T]; % process noise covariance matrix
G = eye(size(A));
H = 1;

w = mvnrnd(zeros(2, 1), Q, N)'; % process noise

for ii = 2:N  % simulate system dynamics
    x(:, ii) = A * x(:, ii-1) + w(:, ii);
end

v = sqrt(R) * randn(1, N); % measurement noise
z = x(1, :) + v;  % position measurements assuming C = [1 0]

% save('data_lab2_eq_13.mat', 'x', 'z'); 
% csvwrite('data_lab2_eq_13.csv', [x' z']); 