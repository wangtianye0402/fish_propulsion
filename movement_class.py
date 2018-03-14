#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

def nabla(function, point, delta=1e-5):
    
    nabla_f = []
    for m1 in range(len(point)):
        point[m1] = float(point[m1])
    
    for m1 in range(len(point)):
        temp_point_1 = np.array(point).reshape(-1,)
        temp_point_1[m1] = temp_point_1[m1] + delta
        temp_point_1 = temp_point_1.tolist()
        temp_point_2 = np.array(point).reshape(-1,)
        temp_point_2[m1] = temp_point_2[m1] - delta
        temp_point_2 = temp_point_2.tolist()

        nabla_f.append((function(temp_point_1) - function(temp_point_2)) / (2*delta))
    
    return nabla_f  

class movement():

	def __init__(self, basic_frequency, sample_rate, extent_preiod, coming_flow_speed):

		self.w0 = 2 * np.pi * basic_frequency
		self.period = 1 / basic_frequency
		self.delta_t = 1 / sample_rate
		self.extent_preiod = extent_preiod
		self.total_point_number = np.floor(self.extent_preiod*self.period / self.delta_t)
		self.U = coming_flow_speed
		
		return

	def reload_predict_model(self, predict_model, time_step):

		self.angle_model = predict_model
		self.time_step = time_step
		self.total_point_number += time_step

		return

	def generate_angle(self, fourier_coefficient):

		a = []
		b = []
		N = int(len(fourier_coefficient) / 2)
		for n in range(N):
			a.append(fourier_coefficient[2*n])
			b.append(fourier_coefficient[2*n+1])

		t = np.linspace(self.delta_t, self.total_point_number*self.delta_t, 
						self.total_point_number)

		angle_temp = np.zeros(shape=[len(t)], dtype=np.float32)
		for n in range(N):
			angle_temp += a[n] * np.cos((n+1)*self.w0*t) + b[n] * np.sin((n+1)*self.w0*t)

		return angle_temp

	def angle_predict_force(self, angle_input):

		predict_set = []

		for time_step_No in range(len(angle_input)-self.time_step+1):

			predict_set.append(angle_input[time_step_No:time_step_No+self.time_step].reshape([-1,1]).tolist())

		predict_set = np.array(predict_set)

		temp_out = np.array(self.angle_model.run_predict(predict_batch_in=predict_set)).transpose()

		return temp_out[0], temp_out[1]

	def calculate_eta(self, fourier_coefficient):

		time = np.linspace(self.delta_t*self.time_step, self.extent_preiod*self.period, 
						   self.total_point_number-self.time_step+1)

		angle = self.generate_angle(fourier_coefficient=fourier_coefficient)

		thrust, torque = self.angle_predict_force(angle_input=angle)

		theta_velocity = np.diff(angle[self.time_step-1 : len(angle)]) / self.delta_t / 180 * np.pi
		eta = self.U * np.trapz(x=time, y=thrust) / np.trapz(x=time[0:len(time)-1], y=torque[0:len(time)-1] * theta_velocity) * (-1000)

		# print(angle)

		return eta

	def restriction_function(self, fourier_coefficient, max_angle, min_angle):

		temp_loss = 1 / (max_angle - np.max(self.generate_angle(fourier_coefficient)))
		temp_loss += 1 / (np.max(self.generate_angle(fourier_coefficient))-min_angle)

		# for fourier_order in range(len(min_angle)):
		# 	temp_loss += 1 / (fourier_coefficient[fourier_order] - min_angle[fourier_order])

		return temp_loss

	def optimize(self, optimizint_total_step, optimizing_step, init_fourier_coefficient,
				 max_angle, min_angle, r_k=5e-1, stop_epoch=1):

		current_point = init_fourier_coefficient

		temp_func = lambda fourier_coefficient : (self.calculate_eta(fourier_coefficient) + (-1)*r_k*self.restriction_function(fourier_coefficient, max_angle, min_angle))

		for iteration in range(optimizint_total_step):

			direction = nabla(function=temp_func, point=current_point)
			direction = np.array(direction)
			direction /= np.sqrt(np.sum(np.square(direction)))

			current_point = current_point + optimizing_step * direction

			if (iteration+1) % stop_epoch == 0:
				print('epoth:%d'%(iteration+1))
				print('current_point:', end='')
				print(current_point.tolist())
				print('eta=%f\n'%self.calculate_eta(current_point))

	def calculate_eta_with_given_movement(self, given_movement):

		thrust, torque = self.angle_predict_force(angle_input=np.array(given_movement))
		time = np.linspace(self.delta_t*self.time_step, self.delta_t*len(given_movement), 
						   len(given_movement)-self.time_step+1)

		theta_velocity = np.diff(given_movement[self.time_step-1 : len(given_movement)]) / self.delta_t / 180 * np.pi
		eta = self.U * np.trapz(x=time, y=thrust) / np.trapz(x=time[0:len(time)-1], y=(torque[0:len(time)-1] * theta_velocity)) * (-1000)

		return eta


