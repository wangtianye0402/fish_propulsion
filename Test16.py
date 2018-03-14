#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from lstm_class import lstm_model
from lstm_preprocess import reshape_data
from matplotlib import pyplot as plt
from scipy import signal

# some hyper-parameters defined below
sample_rate               = 20
noise_model_reload_path   = './noise_model_5/model.ckpt'
varifying_data_folder     = './physical_data/'
output_folder             = './removed_noise_without_butterworth/'
input_size_1              = 1
output_size_1             = 2
time_step_1               = 20

# define the empty graph and session
graph4 = tf.Graph()
with graph4.as_default():
	sess4 = tf.Session()

# reload the noise lstm model
noise_model = lstm_model([graph4, sess4])
noise_model.reload_network(graph_load_path=noise_model_reload_path, retrain=False)

for single_file in os.listdir(varifying_data_folder):

	varifying_data_path = varifying_data_folder + single_file

	# prepare the varifying batch
	input_temp = pd.read_csv(varifying_data_path)

	batch_size_1 = input_temp.shape[0] - (time_step_1-1)

	thrust_all_part = np.array(input_temp.thrust)[time_step_1-1 : input_temp.shape[0]]
	torque_all_part = np.array(input_temp.torque)[time_step_1-1 : input_temp.shape[0]]

	single_file_in, single_file_label = reshape_data(data_sheet=input_temp, input_size=input_size_1, 
													 output_size=output_size_1, time_step=time_step_1, 
													 batch_size=batch_size_1, is_noise=True)
	network_in = np.array(single_file_in[0])

	# start predicting thrust's and torque's noise portion in the experiment data
	out_temp = noise_model.run_predict(predict_batch_in=network_in)
	out_temp = np.array(out_temp).transpose()
	thrust_noise_part = out_temp[0]
	torque_noise_part = out_temp[1]

	# take the noise part off the thrust and torque
	thrust_left_part = thrust_all_part - thrust_noise_part
	torque_left_part = torque_all_part - torque_noise_part

	# # flit high frequency part out of the experiment data
	# f_band_max = 4 / (0.5*sample_rate)
	# b,a = signal.butter(3,f_band_max,btype='low')  
	# thrust_low_part = signal.filtfilt(b,a,thrust_left_part)
	# torque_low_part = signal.filtfilt(b,a,torque_left_part)

	# # make the plot
	# t = np.linspace(0.5, 0.5*batch_size_1, batch_size_1)

	# plt.figure(0)
	# plt.plot(t, thrust_left_part, 'b')
	# plt.plot(t, thrust_low_part, 'r')

	# plt.figure(1)
	# plt.plot(t, torque_left_part, 'b')
	# plt.plot(t, torque_low_part, 'r')

	# plt.show()

	# save the file
	output_path = output_folder + single_file
	output_temp = pd.DataFrame()

	output_temp['time'] = np.array(input_temp.time)[time_step_1-1 : input_temp.shape[0]].tolist()
	output_temp['angle'] = np.array(input_temp.angle)[time_step_1-1 : input_temp.shape[0]].tolist()
	output_temp['noise'] = np.array(input_temp.noise)[time_step_1-1 : input_temp.shape[0]].tolist()
	# output_temp['thrust'] = thrust_low_part.tolist()
	# output_temp['torque'] = torque_low_part.tolist()
	output_temp['thrust'] = thrust_left_part.tolist()
	output_temp['torque'] = torque_left_part.tolist()

	output_temp.to_csv(output_path, index=False, header=False)
