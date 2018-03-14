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
coeff_set                     =       [0.037759344508397026, 25.274850465426375, 1.2179354997628684, -0.2150152736577755, -1.151244754302499, 0.02820467920443135, 0.0941459149340081, 0.21759666718934245]
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

angle = optimizing_movement.generate_angle(coeff_set)
thrust, torque = optimizing_movement.angle_predict_force(angle_input=angle)
eta = optimizing_movement.calculate_eta(fourier_coefficient=coeff_set)

t_1 = np.linspace(1/sample_rate_1, 1/sample_rate_1*len(angle), len(angle))
t_2 = np.linspace(1/sample_rate_1*lstm_model_time_step, 1/sample_rate_1*len(angle), 
				  len(torque))

print('eta=%f'%eta)

plt.figure(0)
plt.plot(t_1, angle)
plt.figure(1)
plt.plot(t_2, thrust)
plt.figure(2)
plt.plot(t_2, torque)

plt.show()

