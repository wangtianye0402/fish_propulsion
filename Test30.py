#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
import os
import xlrd
import pandas as pd
from lstm_preprocess import voltage2data

input_path = r'./voltage_data/'
output_path = r'./physical_data/'

for working_state in os.listdir(input_path):

	current_dir = input_path + working_state +'/'

	count = 0
	for times in os.listdir(current_dir):

		count += 1
		
		original_data = xlrd.open_workbook(current_dir+times)

		after_data = voltage2data(data_file=original_data)

		out_temp = pd.DataFrame(data=after_data, columns=['time', 'angle', 'noise', 'thrust', 'torque'])

		output_file_name = output_path + working_state + '_%d.csv' % count

		out_temp.to_csv(output_file_name, index=False, header=True)
		