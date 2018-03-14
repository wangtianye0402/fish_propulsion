#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import os 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import shutil

data_input_path = './single_period_data/f40A30/'
output_path =      './temp_after_filted/f40A30/'
sample_rate = 20;
U = 0.1;
rho = 1e3;
L = 0.15;
c = 0.2;
S = L*c;

# read the period data into the memory
name_memory = []
time_memory = []
angle_memory = []
thrust_memory = []
torque_memory = []
for single_period_data in os.listdir(data_input_path):

	input_temp = pd.read_csv(data_input_path+single_period_data)

	name_memory.append(single_period_data)
	time_memory.append(np.array(input_temp.time).tolist())
	angle_memory.append(np.array(input_temp.angle).tolist())
	thrust_memory.append(np.array(input_temp.thrust).tolist())
	torque_memory.append(np.array(input_temp.time).tolist())

plt.figure(0)
plt.xlabel('St')
plt.ylabel('$ C_T $')

St_memory = []
CT_memory = []
# calculate the St, CT, and eta for each single period
for single_period_No in range(len(name_memory)):

	time = np.array(time_memory[single_period_No])
	angle = np.array(angle_memory[single_period_No])
	thrust = np.array(thrust_memory[single_period_No])
	torque = np.array(torque_memory[single_period_No])

	T = time[-1] / 2
	f_m = 1 / T
	A = np.max(angle) / 180 * np.pi

	St = 2 * f_m * L * np.sin(A) / U
	C_T = np.trapz(x=time, y=thrust) / (T * 2 * 0.5 * rho * (U**2) * S)
	theta_velocity = np.diff(angle) / np.diff(time)
	eta = U * np.trapz(x=time, y=thrust) / np.trapz(x=time[0:len(time)-1], y=torque[0:len(time)-1] * theta_velocity) * (-1000)

	St_memory.append(St)
	CT_memory.append(C_T)

	plt.plot(St, C_T, 'ro')

plt.plot(np.mean(np.array(St_memory)), np.mean(np.array(CT_memory)), 'g*')

plt.draw()

horizontal = 0.025
height = 0.5
for center_point in range(1):

	pos = list(plt.ginput(1)[0])
	plt.plot([pos[0]-horizontal/2, pos[0]+horizontal/2],[pos[1]-height/2,pos[1]-height/2],'b')
	plt.plot([pos[0]+horizontal/2, pos[0]+horizontal/2],[pos[1]-height/2,pos[1]+height/2],'b')
	plt.plot([pos[0]-horizontal/2, pos[0]+horizontal/2],[pos[1]+height/2,pos[1]+height/2],'b')
	plt.plot([pos[0]-horizontal/2, pos[0]-horizontal/2],[pos[1]-height/2,pos[1]+height/2],'b')

	for single_period_No in range(len(name_memory)):

		St = St_memory[single_period_No]
		C_T = CT_memory[single_period_No]

		if (np.abs(St-pos[0]) < horizontal/2) and (np.abs(C_T-pos[1]) < height/2):

			print(name_memory[single_period_No])

			sub_dir = output_path

			if os.path.exists(sub_dir) == False: 
				os.mkdir(sub_dir)
			
			shutil.copyfile(data_input_path+name_memory[single_period_No],
							sub_dir+name_memory[single_period_No])
			
			plt.plot(St,C_T,'yo')

	plt.draw()

plt.show()