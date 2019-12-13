function [x, z, A, Q, G, H] = gen_data16(var_q, R, N)

% This function generates the data for a 1D motion considering a
% Piecewise constant Wiener process acceleration model, equation (16)

% R is the variance of measurement noise
% z is the measured data
% x are the true values of the system states
% N is the number of measurement points

% example of use
% [x, z] = generate_data(1e-3, 1,100);

T = 1; %[s] Sampling time interval

x = zeros(3, N); % states are [position; speed; acceleration]
x(:,1) = [0; 10; 0]; % state initialization, change to give your own initial values

A = [1 T T^2/2; 0 1 T; 0 0 1]; % Transition matrix
G = [T^2/2; T; 1]; % Vector gain for the process noise
Q = var_q;
H = 1;

w = sqrt(var_q) * randn(1, N); % measurement noise
for ii = 2:N
    x(:, ii) = A * x(:, ii-1) + G * w(ii);
end

v = sqrt(R) * randn(1, N); % measurement noise
z = x(1, :) + v;

% save measurements.mat z 
% save trueStatesLab3.mat x_true;