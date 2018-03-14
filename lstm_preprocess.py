#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# ATI由电压转成力的变化矩阵
voltage2force = np.matrix([[-0.09497, 0.11649,   0.06514, -6.62280,  -0.03482, 6.63115],
                           [-0.10880, 8.03869, -0.03203, -3.64952, 0.06715, -4.00407],
                           [-0.38108, 29.29419, 0.25600, 27.14975, -0.76612, 28.76271]])
f_sample = 20

# define the preprocessing function
def reshape_data(data_sheet, input_size, output_size, time_step, batch_size, is_noise=False):

	noise_set = np.array(data_sheet.noise).reshape([-1,])
	thrust_set = np.array(data_sheet.thrust).reshape([-1,])
	torque_set = np.array(data_sheet.torque).reshape([-1,])
	angle_set = np.array(data_sheet.angle).reshape([-1,])

	network_in = []
	single_batch = []
	for training_sample_No in range(len(noise_set)-time_step+1):

		if is_noise == True:
			single_sample = noise_set[training_sample_No : (training_sample_No+time_step)].reshape([-1,input_size])
		else:
			single_sample = angle_set[training_sample_No : (training_sample_No+time_step)].reshape([-1,input_size])

		single_batch.append(single_sample.tolist())

		if (training_sample_No+1) % batch_size == 0:

			network_in.append(np.array(single_batch).tolist())
			single_batch = []

	network_label = []
	single_batch = []
	for training_sample_No in range(len(noise_set)-time_step+1):

		single_sample = np.zeros([time_step,output_size])
		single_sample[:,0] = thrust_set[training_sample_No : (training_sample_No+time_step)]
		single_sample[:,1] = torque_set[training_sample_No : (training_sample_No+time_step)]

		single_batch.append(single_sample.tolist())

		if (training_sample_No+1) % batch_size == 0:

			network_label.append(np.array(single_batch).tolist())
			single_batch = []

	return network_in, network_label


def voltage2data(data_file):
    """
    这个函数是用来将NI测得的电压信号转化成推力、扭矩和位移信号的函数。但其中不包含减去零点和归一化的过程
    输入量是一个已经打开好了的xls表格的handle
    返回的是一个n行5列的矩阵，每一列分别是时间、推力、扭矩、位移、噪声
    """

    # 打开第1个sheet(用sheet函数)
    table = data_file.sheets()[0]
    # 获得该工作表的行数(其中的nrows属性)
    rows_number = table.nrows

    # 获取该工作表中每一行的数据,存储到voltage_set里面(利用row_value函数)
    voltage_set = []
    for row in range(rows_number - 1):
        voltage_set.append(table.row_values(row + 1))

    voltage_set = np.array(voltage_set)

    # 对ATI的力数据进行相减处理,然后存储到force_data中去
    force_data = np.zeros([rows_number - 1, 6])
    for ATI_col in range(6):
        force_data[:, ATI_col] = voltage_set[:, ATI_col] - voltage_set[:, ATI_col + 7]

    # 将ATI得到的力的电压数据乘以转换矩阵变成力数据(单位:N以及N.mm)
    force_data = voltage2force * force_data.transpose()
    force_data = force_data.transpose()

    # 获得角位移数据
    displacement = 27.9679 * voltage_set[:, 13] + 0.0144
    angle = np.arctan((displacement - 72.5) / 38) / np.pi * 180

    # 计算推力
    thrust = []
    # angle_temp = (angle.transpose()).tolist()[0]
    force_data_temp = force_data.tolist()
    for time in range(rows_number - 1):
        degree = angle[time] / 180 * np.pi
        thrust.append((-1) * force_data_temp[time][0] * np.sin(degree) - force_data_temp[time][1] * np.cos(degree))

    # 计算时间序列
    t = np.linspace(1/f_sample, 1/f_sample * (rows_number-1), (rows_number-1))

    # 计算噪声信号
    noise = voltage_set[:, 14] - voltage_set[:, 6]

    # 将处理好的时间、推力、扭矩、角位移、噪声电压放置于一个表格中
    return_data = np.zeros([rows_number-1, 5])
    return_data[:, 0] = t
    return_data[:, 1] = angle
    return_data[:, 2] = noise
    return_data[:, 3] = np.array(thrust)
    return_data[:, 4] = force_data[:, 2].reshape([-1,])

    return return_data
