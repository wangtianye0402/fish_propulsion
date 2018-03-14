#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd

class lstm_model():
	"""
	This class is to define the lstm model, with the functions of predicting, training, and loss
	"""
	def __init__(self, graph_and_sess):
		"""
		This function is to initialize the lstm model, generate the weights and biases, together with the rnn cell
		"""
		self.graph = graph_and_sess[0]
		self.sess = graph_and_sess[1]

		return

	def first_define(self, input_size, output_size, rnn_size, time_step, learning_rate):
		
		self.basic_parameters = [input_size, output_size, rnn_size, time_step, learning_rate]

		with self.graph.as_default():
			
			# define basic variables
			self.input_holder = tf.placeholder(shape=[None, time_step, input_size], dtype=tf.float32, name='inputs')
			self.label_holder = tf.placeholder(shape=[None, time_step, output_size], dtype=tf.float32, name='labels')
			
			weight_in = tf.Variable(tf.random_normal([input_size, rnn_size]), dtype=tf.float32, name='weight_in')
			bias_in = tf.Variable(tf.random_normal([rnn_size]), dtype=tf.float32, name='bias_in')

			core_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)

			weight_out = tf.Variable(tf.random_normal([rnn_size, output_size]), dtype=tf.float32, name='weight_out')
			bias_out = tf.Variable(tf.random_normal([output_size]), dtype=tf.float32, name='bias_out')

			# define predict operator
			batch_size = tf.shape(self.input_holder)[0]

			input_2D = tf.reshape(self.input_holder, [-1, input_size])
			temp_input = tf.matmul(input_2D, weight_in) + bias_in
			input_3D = tf.reshape(temp_input, [-1, time_step, rnn_size])

			core_cell_init_state = core_cell.zero_state(batch_size, dtype=tf.float32)
			output_3D, final_states = tf.nn.dynamic_rnn(core_cell, input_3D, initial_state=core_cell_init_state, dtype=tf.float32)

			output_2D = tf.reshape(output_3D, [-1, rnn_size])
			temp_output = tf.matmul(output_2D, weight_out) + bias_out
			output_3D = tf.reshape(temp_output, [-1, time_step, output_size], name='predict_op')		

			# define loss function
			predict_value_1D = tf.reshape(output_3D, [-1])

			label_value_1D = tf.reshape(self.label_holder, [-1])

			loss_value = tf.reduce_mean(tf.square(predict_value_1D - label_value_1D), name='loss_op')

			# define training operator
			lr = learning_rate

			train_op_name = 'train_op_%d' % np.random.randint(1000,9999)
			training_operator = tf.train.AdamOptimizer(learning_rate=lr, name=train_op_name).minimize(loss_value)

			# add predict_op and loss_op to their collections to make us able to get back them again
			tf.add_to_collection('predict_op', output_3D)
			tf.add_to_collection('loss_op', loss_value)

			self.predict_op = output_3D
			self.loss_op = loss_value
			self.train_op = training_operator
			self.init_op = tf.global_variables_initializer()

			self.sess.run(self.init_op)

		return

	def reload_network(self, graph_load_path, retrain, lr=0):

		with self.graph.as_default():

			saver = tf.train.import_meta_graph(graph_load_path + '.meta')

			self.input_holder = self.graph.get_operation_by_name('inputs').outputs[0]
			self.label_holder = self.graph.get_operation_by_name('labels').outputs[0]

			self.predict_op = tf.get_collection('predict_op')[0]
			self.loss_op = tf.get_collection('loss_op')[0]
			
			if retrain == True:
				train_op_name = 'train_op_%d' % np.random.randint(1000,9999)
				self.train_op = tf.train.AdamOptimizer(learning_rate=lr, name=train_op_name).minimize(self.loss_op)

			self.init_op = tf.global_variables_initializer()

			self.sess.run(self.init_op)
			saver.restore(self.sess, graph_load_path)

		return

	def save_graph(self, path):

		writer = tf.summary.FileWriter(path, graph=self.graph)
		writer.close()

		return

	def save_network(self, save_path):

		with self.graph.as_default():
			saver = tf.train.Saver()
			saver.save(self.sess, save_path)
			
		return

	def run_train(self, training_set_in, training_set_label):

		with self.graph.as_default():

			for batch_No in range(len(training_set_in)):

				batch_in = np.array(training_set_in[batch_No])
				batch_label = np.array(training_set_label[batch_No])

				self.sess.run(self.train_op, feed_dict={self.input_holder : batch_in,
														self.label_holder : batch_label})
		return

	def run_loss(self, varifying_batch_in, varifying_batch_label):

		with self.graph.as_default():

			varifying_batch_in = np.array(varifying_batch_in)
			varifying_batch_label = np.array(varifying_batch_label)

			loss_value = self.sess.run(self.loss_op, feed_dict={self.input_holder : varifying_batch_in,
																self.label_holder : varifying_batch_label})

		return loss_value

	def run_predict(self, predict_batch_in):

		with self.graph.as_default():

			predict_batch_in = np.array(predict_batch_in)

			out_temp = self.sess.run(self.predict_op, feed_dict={self.input_holder : predict_batch_in})

			predict_result_dependent = []

			for sample_No in range(len(out_temp)):

				predict_result_dependent.append(np.array(out_temp[sample_No][-1]).tolist())

		return predict_result_dependent


