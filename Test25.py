#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
from lstm_class import lstm_model
from lstm_preprocess import reshape_data
import os

# some hyper-parameters defined below
data_input_path = './removed_noise_without_butterworth/mean_exp_data/'
angle_model_save_path = './model4/model.ckpt'
input_size_1 = 1
output_size_1 = 2
rnn_size_1 = 50
time_step_1 = 50
batch_size_1 = 10
learning_rate_1 = 0.000006

# get the training set from the files, and then reshape them into the lstm-correct form
network_in = []
network_label = []
for single_file in os.listdir(data_input_path):

	input_temp = pd.read_csv(data_input_path+single_file)

	single_file_in, single_file_label = reshape_data(data_sheet=input_temp, 
													 input_size=input_size_1, 
													 output_size=output_size_1, 
													 time_step=time_step_1, 
													 batch_size=batch_size_1, is_noise=False)

	for single_batch_No in range(len(single_file_in)):

		network_in.append(np.array(single_file_in[single_batch_No]).tolist())
		network_label.append(np.array(single_file_label[single_batch_No]).tolist())

network_in = np.array(network_in)
network_label = np.array(network_label)

# get the varifying batch
varify_batch_in = []
varify_batch_label = []
for batch_No in range(len(network_in)):

	for sample_No in range(len(network_in[batch_No])):

		varify_batch_in.append(network_in[batch_No][sample_No].tolist())
		varify_batch_label.append(network_label[batch_No][sample_No].tolist())

varify_batch_in = np.array(varify_batch_in)
varify_batch_label = np.array(varify_batch_label)

# define working graph and its session
graph1 = tf.Graph()
with graph1.as_default():
	sess1 = tf.Session()

# define the angle-thrust&torque lstm model
angle_model = lstm_model(graph_and_sess=[graph1, sess1])
# angle_model.first_define(input_size=input_size_1, output_size=output_size_1, 
# 						 rnn_size=rnn_size_1, time_step=time_step_1, 
# 						 learning_rate=learning_rate_1)

angle_model.reload_network(graph_load_path=angle_model_save_path, 
						   retrain=True, lr=learning_rate_1)

angle_model.save_graph(path='./graph')

# start training
current_loss = angle_model.run_loss(varifying_batch_in=varify_batch_in, 
									varifying_batch_label=varify_batch_label)
print(current_loss)

for iteration in range(10000):

	angle_model.run_train(training_set_in=network_in, training_set_label=network_label)

	if (iteration+1) % 100 == 0:

		current_loss = angle_model.run_loss(varifying_batch_in=varify_batch_in, 
											varifying_batch_label=varify_batch_label)

		print(current_loss)

	if (iteration+1) % 1000 == 0:

		angle_model.save_network(save_path=angle_model_save_path)
		

