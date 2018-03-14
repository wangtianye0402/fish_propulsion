#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import os 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data_input_folder = './removed_noise_with_butterworth/mean_exp_data/'
data_output_path = './result2.csv'
sample_rate = 20;
U = 0.1;
rho = 1e3;
L = 0.15;
c = 0.2;
S = L*c;

plt.figure(0)
plt.xlabel('St',fontsize=20)
plt.ylabel('$ \eta $',fontsize=20)

memory = []

# for working_condition in os.listdir(data_input_folder):

# 	data_input_path = data_input_folder + working_condition + '/'

data_input_path = data_input_folder

for single_period_data in os.listdir(data_input_path):

	input_temp = pd.read_csv(data_input_path+single_period_data)

	time = np.array(input_temp.time)
	angle = np.array(input_temp.angle)
	thrust = np.array(input_temp.thrust)
	torque = np.array(input_temp.torque)

	T = time[-1] / 2
	f_m = 1 / T
	A = np.max(angle) / 180 * np.pi

	St = 2 * f_m * L * np.sin(A) / U
	C_T = np.trapz(x=time, y=thrust) / (T * 2 * 0.5 * rho * (U**2) * S)
	theta_velocity = np.diff(angle) / np.diff(time) / 180 * np.pi
	eta = U * np.trapz(x=time, y=thrust) / np.trapz(x=time[0:len(time)-1], y=torque[0:len(time)-1] * theta_velocity) * (-1000)

	memory.append(np.array([f_m, A, St, C_T, eta]).tolist())

	plt.plot(St, eta, 'ro')

plt.show()

output_temp = pd.DataFrame()
memory = np.array(memory)
output_temp['f'] = memory[:,0]
output_temp['A'] = memory[:,1]
output_temp['St'] = memory[:,2]
output_temp['CT'] = memory[:,3]
output_temp['eta'] = memory[:,4]

output_temp.to_csv(path_or_buf=data_output_path, index=None)


