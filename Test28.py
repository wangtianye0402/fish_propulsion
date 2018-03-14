#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from movement_class import movement
from matplotlib import pyplot as plt
from lstm_class import lstm_model
from appendix import nabla

angle_model_reload_path       =       './model1/model.ckpt'
basic_frequency_1             =       0.15
sample_rate_1                 =       20
extent_preiod_1               =       10
start_coeff                   =       [0.04903099355732584, 25.24618087735042, 1.1942706549190347, -0.1510961281977027, -1.157345195392571, 0.007226642769222984, 0.11212715432736636, -0.08674787954732166]
lstm_model_time_step          =       50
U                             =       0.1
L                             =       0.15 
optimizing_step               =       0.06


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

current_point = np.array(start_coeff)
optimizing_movement.optimize(optimizint_total_step=10000, optimizing_step=optimizing_step, 
							 init_fourier_coefficient=current_point, r_k=5e-3, max_angle=70,
							 stop_epoch=10)
