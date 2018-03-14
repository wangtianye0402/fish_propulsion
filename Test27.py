#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from movement_class import movement
from matplotlib import pyplot as plt
from lstm_class import lstm_model

angle_model_reload_path       =       './model1/model.ckpt'
basic_frequency_1             =       0.15
sample_rate_1                 =       20
extent_preiod_1               =       10
# coeff_set                     =       [0, 0, 5, 0, 0]
lstm_model_time_step          =       50
U                             =       0.1
L                             =       0.15 

graph1 = tf.Graph()
with graph1.as_default():
	sess1 = tf.Session()
angle_model = lstm_model(graph_and_sess=[graph1, sess1])
angle_model.reload_network(graph_load_path=angle_model_reload_path, retrain=False)

optimizing_movement = movement(basic_frequency=basic_frequency_1, 
							   sample_rate=sample_rate_1, extent_preiod=extent_preiod_1, 
							   coming_flow_speed=U)
optimizing_movement.reload_predict_model(predict_model=angle_model, 
										 time_step=lstm_model_time_step)
least_angle_amplitude = 10
largest_angle_amplitude = 70
eta_step_total = 100
angle_set = np.linspace(least_angle_amplitude, largest_angle_amplitude, eta_step_total)
St = basic_frequency_1 * 2 * L * np.sin(angle_set/180*np.pi) / U

eta_set = []
for eta_step in range(eta_step_total):

	coeff_set = [0, angle_set[eta_step], 0, 0]
	eta_set.append(optimizing_movement.calculate_eta(fourier_coefficient=coeff_set))

	print('angle:%f'%angle_set[eta_step])
	print('eta:%f'%eta_set[eta_step])

plt.figure(0)
plt.title('$f=0.15$', fontsize=15)
plt.xlabel('angle', fontsize=15)
plt.ylabel('$\eta$', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.plot(angle_set, eta_set, 'ro')
plt.show()

