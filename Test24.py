#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import os 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data_input_path = './single_period_data/changing_frequency_2/'
data_output_path =     './mean_exp_data/changing_frequency_2.csv'

plt.figure(0)
plt.xlabel('t[s]')
plt.ylabel('thrust[N]')

sum_num = 0
for single_period in os.listdir(data_input_path):

	sum_num += 1

	input_temp = pd.read_csv(data_input_path+single_period)

	if sum_num == 1:

		time_sum = np.array(input_temp.time)
		angle_sum = np.array(input_temp.angle)
		noise_sum = np.array(input_temp.noise)
		thrust_sum = np.array(input_temp.thrust)
		torque_sum = np.array(input_temp.torque)

	else:

		time_sum += np.array(input_temp.time)
		angle_sum += np.array(input_temp.angle)
		noise_sum += np.array(input_temp.noise)
		thrust_sum += np.array(input_temp.thrust)
		torque_sum += np.array(input_temp.torque)

	plt.plot(np.array(input_temp.time), np.array(input_temp.thrust), 'b')

output_temp = pd.DataFrame()
output_temp['time'] = time_sum / sum_num
output_temp['angle'] = angle_sum / sum_num
output_temp['noise'] = noise_sum / sum_num
output_temp['thrust'] = thrust_sum / sum_num
output_temp['torque'] = torque_sum / sum_num

output_temp.to_csv(path_or_buf=data_output_path, index=None)

plt.plot(time_sum / sum_num, thrust_sum / sum_num, 'ro')
plt.show()
