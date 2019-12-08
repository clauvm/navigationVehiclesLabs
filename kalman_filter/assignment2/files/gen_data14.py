#!/usr/bin/python
'''
This function generates the data for a 1D motion considering a
Piecewise constant, white noise acceleration model, equation (14)

var_a is the variance of the process noise
R is the variance of measurement noise
z is the measured data
x are the true values of the system states

example of use
python gen_data14.py 1e-3 1
'''

import sys
import numpy as np
import matplotlib.pyplot as plt

var_a = float(sys.argv[1]) 
R = float(sys.argv[2])

N = 100 # data size
T = 1.0 # [s] Sampling time interval

x = np.zeros((2, N)) # states are [position; speed; acceleration]
x[:, 0] = [0, 10] # state initialization, change to give your own initial values

A = np.array([[1, T], 
              [0, 1]], dtype=float)  # Transition matrix

G = np.array([[T**2/2], 
              [T]], dtype=float) # Vector gain for the process noise

w = np.random.normal(0.0, np.sqrt(var_a), N) # process noise
for ii in range(1, N): # simulate system dynamics
  x[:, ii] = A.dot(x[:, ii-1]) + G.dot(w[ii]).T

v = np.random.normal(0.0, np.sqrt(R), N) # measurement noise
z = x[0, :] + v  # position measurements assuming C = [1 0 0]

f1 = plt.figure()
plt.plot(z, label='linear')
plt.xlabel('Time [s]')
plt.ylabel('Measured position')
f1.show()

f2 = plt.figure()
plt.plot(x[0,:], label='linear')
plt.xlabel('Time [s]')
plt.ylabel('True position [m]')
f2.show()

f3 = plt.figure()
plt.plot(x[1,:], label='linear')
plt.xlabel('Time [s]')
plt.ylabel('True speed [m/s]')
f3.show()
