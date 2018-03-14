#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import os 
import pandas as pd
import numpy as np

# hyper-parameters
data_input_path = './removed_noise_without_butterworth/original_data/'
label_input_path = './picked_label/'
output_file_folder = './temp_pre_filted/'
sample_rate = 20

# take the experiment data into the memory
single_period_count = 0

for working_condition in os.listdir(data_input_path):

# working_condition = 'f20A30'

	for single_file in os.listdir(data_input_path+working_condition+'/'):

		single_data_file_input_path = data_input_path+working_condition+'/'+single_file
		single_label_file_input_path = label_input_path + single_file

		data_set = np.array(pd.read_csv(filepath_or_buffer=single_data_file_input_path,
										header=None, names=['time','angle','noise','thrust','torque']),
							dtype=np.float32)
		label_set = np.array(pd.read_csv(filepath_or_buffer=single_label_file_input_path,
										 header=None,
										 names=['start','end']))

		output_file_path = output_file_folder + working_condition + '/'
		if os.path.exists(output_file_path) == False:
			os.mkdir(output_file_path)

		for single_period_No in range(len(label_set)):

			single_period_count += 1

			start_point = label_set[single_period_No][0] - 1
			end_point = label_set[single_period_No][1] - 1

			t = np.linspace(1/sample_rate, 1/sample_rate*(end_point-start_point+1),
							end_point-start_point+1)

			output_file_name_temp = output_file_path+'No%d.csv'%single_period_count
			fp = open(output_file_name_temp, 'w')
			fp.write('time,angle,noise,thrust,torque\n')
			
			for time_step in range(end_point-start_point+1):

				current_point = start_point + time_step

				fp.write('%f,%f,%f,%f,%f\n'%(t[time_step], 
											data_set[current_point][1],
											data_set[current_point][2],
											data_set[current_point][3],
											data_set[current_point][4]))

			fp.close()
