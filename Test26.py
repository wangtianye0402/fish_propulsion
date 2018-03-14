#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
from lstm_class import lstm_model
from lstm_preprocess import reshape_data
from matplotlib import pyplot as plt
import os

# some hyper-parameters are defined below
angle_model_save_path = './model4/model.ckpt'
varifying_data_path = './removed_noise_without_butterworth/la/'
result_save_path = './result/change_frequency_2.csv'
input_size_1 = 1
output_size_1 = 2
time_step_1 = 50
batch_size_1 = 10
sample_rate = 20

# prepare for the varifying set, reshaping the time sequence into the lstm shape
varifying_in = []
varifying_label = []
for single_file in os.listdir(varifying_data_path):

	input_temp = pd.read_csv(varifying_data_path+single_file)

	single_file_in, single_file_label = reshape_data(data_sheet=input_temp, 
													 input_size=input_size_1, 
													 output_size=output_size_1, 
													 time_step=time_step_1, 
													 batch_size=batch_size_1, is_noise=False)

	for single_batch_No in range(len(single_file_in)):

		for singel_sample_No in range(len(single_file_in[single_batch_No])):

			varifying_in.append(np.array(single_file_in[single_batch_No][singel_sample_No]).tolist())
			varifying_label.append(np.array(single_file_label[single_batch_No][singel_sample_No]).tolist())

varifying_in = np.array(varifying_in)

# prepare the label_thrust and label_torque whose shapes are like time sequence 
label_temp = []
for sample in varifying_label:
	label_temp.append(np.array(sample[-1]).tolist())
label_temp = np.array(label_temp).transpose()
label_thrust = label_temp[0]
label_torque = label_temp[1]

# get the angle sequence coordinate to the thrust and torque sequence
output_temp_angle = []
for sample_No in range(len(varifying_in)):
	output_temp_angle.append(np.array(varifying_in[sample_No][-1]).tolist())
output_temp_angle = np.array(output_temp_angle).reshape([-1,])

# define the working graph and session, and then reload the already-trained model into the graph
graph2 = tf.Graph()
with graph2.as_default():
	sess2 = tf.Session()

angle_model = lstm_model(graph_and_sess=[graph2, sess2])
angle_model.reload_network(graph_load_path=angle_model_save_path, retrain=False)

# predict the thrust and torque using the model, and then reshape the predict_thrust and predict_torque into the time sequence like
varifying_predict = angle_model.run_predict(predict_batch_in=varifying_in)
varifying_predict = np.array(varifying_predict).transpose()
predict_thrust = varifying_predict[0]
predict_torque = varifying_predict[1]

# # draw the result plot
t = np.linspace(time_step_1/sample_rate, (len(varifying_in)+time_step_1-1)/sample_rate,
				len(varifying_in))
# plt.figure(0)
# plt.xlabel('t[s]')
# plt.ylabel('thrust[N]')
# plt.title('t-thrust relationship')
# plt.plot(t, label_thrust, 'b', label='label')
# plt.plot(t, predict_thrust, 'r', label='predict')
# plt.legend()

# plt.figure(1)
# plt.xlabel('t[s]')
# plt.ylabel('torque[N mm]')
# plt.title('t-torque relationship')
# plt.plot(t, label_torque, 'b', label='label')
# plt.plot(t, predict_torque, 'r', label='predict')
# plt.legend()
# plt.show()


# export the result
output_temp = pd.DataFrame()
output_temp['time'] = t
output_temp['angle'] = output_temp_angle
output_temp['predict_thrust'] = predict_thrust
output_temp['label_thrust'] = label_thrust
output_temp['predict_torque'] = predict_torque
output_temp['label_torque'] = label_torque
output_temp.to_csv(result_save_path, index=None)

