#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from movement_class import movement
from matplotlib import pyplot as plt
from lstm_class import lstm_model

angle_model_reload_path       =       './model5/model.ckpt'
whole_frequency_1             =       0.20
effect_proportion_1           =       0.5
effect_amplitude              =       20
sample_rate_1                 =       20
extent_preiod_1               =       10
lstm_model_time_step          =       50
U                             =       0.1
L                             =       0.15 

# prepare the fish-like movement
time = np.linspace(1/sample_rate_1, 1/whole_frequency_1*extent_preiod_1, 
				   int(sample_rate_1/whole_frequency_1)*extent_preiod_1)

movement_1 = np.zeros([int(sample_rate_1/whole_frequency_1)])
w0 = 2 * np.pi * (whole_frequency_1 / effect_proportion_1)
effect_N = int(len(movement_1)*effect_proportion_1)
for m1 in range(effect_N):
	current_t = (m1+1) / sample_rate_1
	movement_1[m1] = effect_amplitude * np.sin(w0*current_t)

total_movement = np.zeros([extent_preiod_1*len(movement_1)])
for m1 in range(extent_preiod_1):
	total_movement[m1*len(movement_1) : (m1+1)*len(movement_1)] = movement_1

# reload the force model and the movement class
graph1 = tf.Graph()
with graph1.as_default():
	sess1 = tf.Session()
angle_model = lstm_model(graph_and_sess=[graph1, sess1])
angle_model.reload_network(graph_load_path=angle_model_reload_path, retrain=False)

optimizing_movement = movement(basic_frequency=whole_frequency_1, 
							   sample_rate=sample_rate_1, extent_preiod=extent_preiod_1, 
							   coming_flow_speed=U)
optimizing_movement.reload_predict_model(predict_model=angle_model, 
										 time_step=lstm_model_time_step)

thrust, torque = optimizing_movement.angle_predict_force(angle_input=total_movement)
eta = optimizing_movement.calculate_eta_with_given_movement(given_movement=total_movement)

print('eta=%f' % eta)

plt.figure(0)
plt.xlabel('time[s]')
plt.ylabel('thrust[N]')
plt.plot(time[lstm_model_time_step-1 : len(time)], thrust)
plt.figure(1)
plt.xlabel('time[s]')
plt.ylabel('torque[N mm]')
plt.plot(time[lstm_model_time_step-1 : len(time)], torque)
plt.show()
