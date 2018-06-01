#!/usr/bin/python
__author__ = 'ru'
# encoding: UTF-8

""" 
@version:     1.0
@function: 测试RBFInterploant库代码
@source: https://github.com/treverhines/RBF
"""

from rbf.interpolate import RBFInterpolant
#from interpolate import RBFInterpolant
import rbf.basis
# from curve_data import load_curve, compare_fit
from sklearn import preprocessing
import csv
import numpy as np
from matplotlib import pyplot as plt
from random import sample
import gc
import re


#################################################################################################
#这是读取导出的excel表格的处理，目前是将参数以关键形参的方式提供，
#
#2018/4/12 将读取策略改为，如果该组数据st标签为0，则丢弃该组数据，价值网络不对这些畸形点进行拟合
#
def ImportAllData(path: str, row_start = 2, x_column_start= 1, y_column_start=6, x_dim = 5, y_dim = 1, each_y_dim = 1):
	
	dataset = {}

	x_result = []
	y_result = []
	index = []
	flag = 0

	num_re_str = "-?\\d+(\\.)?\\d*([eE]?[\\-\\+]?\\d+)?"

	with open(path, 'r') as data:
		csvfile = csv.reader(data)

		for i, line in enumerate(csvfile):
			
			flag = 0

			if i < row_start :
				continue
			else:
				#先专门收集一下input输入向量的数据，最后呈现为 m x n矩阵， m为样本容量，n为向量长度
				line_x_data = []
				for x_index in range(x_column_start, x_column_start + x_dim):
					temp = line[x_index].split(",")
					assert( len(temp) == 1 )  #对于输入向量而言，每个维度独占一个格

					if re.match(num_re_str, temp[0]):
						line_x_data.append( float(temp[0]) )
					else:
						flag = 1
						break
				#再来收集一下样本的label，最后呈现为 m x d x s矩阵，m为样本容量，d为输出的特征数目，s为每个特征的维度        
				line_y_data = []

				for y_index in range(y_column_start, y_column_start + y_dim):
					print("fuck asshole")
					temp = line[y_index].strip().split(",")
					
					#过滤操作，如果相应的特征维度不等于目标输入的维度，则认为该次样本是无效的
					if len(temp) != each_y_dim :
						print("impoosi")
						flag = 1
						break
					
					each_target = []
					for k in range(each_y_dim):
						if re.match(num_re_str, temp[0]):
							each_target.append( float(temp[k]) * 1000 )
						else:
							flag = 1

					line_y_data.append(each_target)

				#从船舶方面获取的消息是 > 0.001的数据可以认为是计算无效，为了后续的RBF归一化处理可以不受这些畸形点影响，直接不加进去
				if flag ==0 and line_y_data[0][0] > 0.0003 * 1000 :
					flag = 1

				if flag == 0:
					print("fuck ass")
					x_result.append( line_x_data ) #x_result应该是二维矩阵， m x n,其中m为样本容量，n为向量长度
					y_result.append( line_y_data ) #y_result应该是三维矩阵， m x y_dim x each_y_dim, 其中m为样本容量， y_dim为目标特征数目， each_y_dim为特征的具体维度
					index.append( i )
				elif flag == 1:
					flag = 0

	assert( len(x_result) == len(y_result) == len(index)) #输入样本向量数目应该和样本标签数目相等。
	
	print("length is :", len(y_result))
	
	dataset['input'] = np.array(x_result, dtype=np.float64)
	dataset['label'] = np.array(y_result, dtype=np.float64)
	dataset['index'] = np.array(index, dtype = np.int32)

	print("Data read completed, length: ", len(y_result) )
	print(np.max(dataset['label']), dataset['index'][np.argmax(dataset['label'])], dataset['input'][np.argmax(dataset['label'])] )
	print(np.min(dataset['label']), dataset['index'][np.argmin(dataset['label'])], dataset['input'][np.argmin(dataset['label'])] )

	return dataset

def getSickPoints(path: str, row_start = 2, x_column_start= 1, y_column_start=6, x_dim = 5, y_dim = 1, each_y_dim = 1):
	
	dataset = {}

	x_result = []
	y_result = []
	index = []
	flag = 0

	num_re_str = "-?\\d+(\\.)?\\d*([eE]?[\\-\\+]?\\d+)?"

	with open(path, 'r') as data:
		csvfile = csv.reader(data)

		for i, line in enumerate(csvfile):
			
			flag = 0

			if i < row_start :
				continue
			else:
				#先专门收集一下input输入向量的数据，最后呈现为 m x n矩阵， m为样本容量，n为向量长度
				line_x_data = []
				for x_index in range(x_column_start, x_column_start + x_dim):
					temp = line[x_index].split(",")
					assert( len(temp) == 1 )  #对于输入向量而言，每个维度独占一个格

					if re.match(num_re_str, temp[0]):
						line_x_data.append( float(temp[0]) )
					else:
						flag = 1
						break
				#再来收集一下样本的label，最后呈现为 m x d x s矩阵，m为样本容量，d为输出的特征数目，s为每个特征的维度        
				line_y_data = []

				for y_index in range(y_column_start, y_column_start + y_dim):
					print("fuck asshole")
					temp = line[y_index].strip().split(",")
					
					#过滤操作，如果相应的特征维度不等于目标输入的维度，则认为该次样本是无效的
					if len(temp) != each_y_dim :
						print("impoosi")
						flag = 1
						break
					
					each_target = []
					for k in range(each_y_dim):
						if re.match(num_re_str, temp[0]):
							each_target.append( float(temp[k]) * 1000 )
						else:
							flag = 1

					line_y_data.append(each_target)

				#从船舶方面获取的消息是 > 0.001的数据可以认为是计算无效，为了后续的RBF归一化处理可以不受这些畸形点影响，直接不加进去
				if flag ==0 and (line_y_data[0][0] > 0.0005 * 1000 or line_y_data[0][0] < 0.0003 * 1000):
					flag = 1

				if flag == 0:
					print("fuck ass")
					x_result.append( line_x_data ) #x_result应该是二维矩阵， m x n,其中m为样本容量，n为向量长度
					y_result.append( line_y_data ) #y_result应该是三维矩阵， m x y_dim x each_y_dim, 其中m为样本容量， y_dim为目标特征数目， each_y_dim为特征的具体维度
					index.append( i )
				elif flag == 1:
					flag = 0

	assert( len(x_result) == len(y_result) == len(index)) #输入样本向量数目应该和样本标签数目相等。
	
	print("length is :", len(y_result))
	
	dataset['input'] = np.array(x_result, dtype=np.float64)
	dataset['label'] = np.array(y_result, dtype=np.float64)
	dataset['index'] = np.array(index, dtype = np.int32)

	print("Data read completed, length: ", len(y_result) )
	print(np.max(dataset['label']), dataset['index'][np.argmax(dataset['label'])], dataset['input'][np.argmax(dataset['label'])] )
	print(np.min(dataset['label']), dataset['index'][np.argmin(dataset['label'])], dataset['input'][np.argmin(dataset['label'])] )

	return dataset

def SampleData(raw_data, size):

	new_data = {'input':[], 'label':[], 'index':[] }
	print("----> total data item sum : ", len(raw_data['input']), len(raw_data['label']))
	print("----> randomly extracted data item num: ", size)

	wanted_data_index = sample(range(len(raw_data['input'])), size)
	new_data['input'] = np.array(raw_data['input'], dtype=np.float64)[wanted_data_index, :]
	new_data['label'] = np.array(raw_data['label'], dtype=np.float64)[wanted_data_index, :]
	new_data['index'] = np.array(raw_data['index'], dtype=np.int32)[wanted_data_index]

	return new_data

class MultiDimRBF(object):

	def __init__(self, inputs):
		# 由于RBF径向基函数是通过向量间距离来设计的，故而为了统一样本输入向量的各维量纲和数量级差异，需要进行预处理
		
		self.raw_input_data = inputs;
		self.inputs_scaler = preprocessing.MinMaxScaler()
		#self.inputs_scaler = preprocessing.StandardScaler() #测试采用规范化的预处理手段
		self.normalized_inputs = self.inputs_scaler.fit_transform(inputs)

		self.rbf_nets = []  # 针对多维输出的情况，为输出的y的每一维设计一个专用的RBF网络，这个是用来收集所有的RBF网络

	def add_single_output(self, target, penalty=0.1,
						  basis=rbf.basis.phs2,
						  order=2):
		'''
		该函数是针对样本点y值是多维向量设计的，每次可以只添加y的一个维度，然后训练出针对该维度的专用RBF
		:param target:  目标值当前维度dim的向量， nx1, 其中n为训练样本点数目
		:param penalty:  惩罚系数0~1之间，如果惩罚系数越强，则多项式正则项的影响效果越强，如果p很大，则当
						当前的拟合网络退化成多项式回归
		:param basis:   选择当前RBF interpolant采用的RBF核函数类型
		:param order:   设置正则约束的多项式次数
		'''

		print(self.normalized_inputs.shape, target.shape)
		# 同样道理，需要对y输出值进行归一化处理
		#target_scaler = preprocessing.MinMaxScaler()
		target_scaler = preprocessing.StandardScaler()
		# 保证处理时的y向量是列向量，归一化是按照同列进行的
		target = target_scaler.fit_transform(target.reshape(-1, 1))  

		net = RBFInterpolant(self.normalized_inputs, target.reshape(-1), penalty=penalty, basis=basis, order=order)

		self.rbf_nets.append((len(self.rbf_nets), target_scaler, net))

	# python式的仿函数风格,其中focus_dim要么缺省，要么是类似于[1,12,14]这种形式的list
	def __call__(self, inputs, focus_dim = None):  
		inputs = self.inputs_scaler.transform(inputs)  # 对于test数据也要进行处理，mxn， m为测试样本数目， n为样本输入向量的size
		result = np.zeros((len(inputs), len(self.rbf_nets)))

		#如果调用者没有指定特定的目标特性维度，那么则默认对所有的维度都感兴趣
		if focus_dim == None:
			for i, scaler, net in self.rbf_nets:
				result[:, i] = (scaler.inverse_transform(net(inputs).reshape(-1, 1))).reshape(-1)
		else:
			for i, scaler, net in self.rbf_nets:
				if i in focus_dim:
					result[:, i] = (scaler.inverse_transform(net(inputs).reshape(-1, 1))).reshape(-1)

		return result


def model_test(fig_name, raw_data, basis=rbf.basis.se, order=2, penalty=0.4, size=500, x_dim = 5, y_dim = 1, each_y_dim = 1):
	'''
	对RBF网络的的测试函数，
	raw_data为输入的数据，字典类型，内部有{'input': , 'label': ,'index': }，注意为了后续可以抽样，raw_data['input']必须是np.array
	basis:用于指定输入的核函数类型
	order:防止过拟合需要添加在正则约束的多项式，此项用于指定多项式的次数
	penalty:0~1，用于指定惩罚系数，越小，则RBF过拟合的倾向越重，越大，则多项式约束的作用越明显
	size:由于是测试函数，需要将整个数据集分为训练集和测试集，这里的size用于指定提取的数据总量大小
	x_dim:用于指明样本的输入向量的维度，为后续的数据集提供检验标准，不满足维度要求的数据则被视为畸形项，丢弃
	y_dim:用于指明样本的输出特征数目，每个特征可能也是一个向量
	each_y_dim:用于指明特征向量的维度，按理说，应该是为每个特征单独提供一个维度说明，这里偷懒，默认所有的输出特征向量的维度是相同的
	'''
	all_index_list = [temp for temp in range(len(raw_data['input']))]

	#收集训练集的数据集合
	wanted_data_index = sample(all_index_list, size)
	train_set = {'input':[], 'label':[], 'index':[] }
	train_set['input'] = np.array(raw_data['input'], dtype=np.float64)[wanted_data_index, :]
	train_set['label'] = np.array(raw_data['label'], dtype=np.float64)[wanted_data_index, :]
	train_set['index'] = np.array(raw_data['index'], dtype=np.int32)[wanted_data_index]

	for ele in wanted_data_index:
		if ele in all_index_list:
			all_index_list.remove(ele)
		else:
			pass

	#收集测试集的数据集合
	test_data_index = []
	test_data_index = sample(all_index_list, 100)
	test_set = {'input':[], 'label':[], 'index':[]}
	test_set['input'] = np.array(raw_data['input'], dtype=np.float64)[test_data_index, :]
	test_set['label'] = np.array(raw_data['label'], dtype=np.float64)[test_data_index, :]
	test_set['index'] = np.array(raw_data['index'], dtype=np.int32)[test_data_index]
	
	print("train_set input size:", train_set['input'].shape )
	print("test_set input size:", test_set['input'].shape )
	
	print("------>get ready for new RBF net train")
	assert( len(train_set['input'][0]) == x_dim )
	train_set_inputs = train_set['input']

	net = MultiDimRBF(train_set_inputs)

	#由于有y_dim个特征，每个特征又有each_y_dim个维度.从而RBF网络中有y_dim * each_y_dim个具体的RBF网络
	for i in range(y_dim):
		for j in range(each_y_dim):
			target_y_data = train_set['label'][:, i, j]
			net.add_single_output( target_y_data, penalty=penalty, basis=basis, order=order)

	print("------>check test dataset output")
	test_set_inputs = test_set['input']
	test_set_label = test_set['label']

	#RBF调用原型形式为： def __call__(self, inputs, focus_dim = None)
	test_set_predict = net( test_set_inputs ) #返回将是m x K矩阵，其中m为样本容量，K=y_dim * each_y_dim

	#删除RBF持有对象，以释放内存
	del net
	# 绘制图像,这里只画100%流量时对应的效率
	#target_dim = 0 * each_y_dim + eye_dim
	test_set_label = test_set_label[:, 0, 0].reshape(-1)
	test_set_predict = test_set_predict[:, 0].reshape(-1)

	accuracy = 0
	for i in range( len(test_set_label) ):
		real_comparison_list = get_comparison_list(test_set_label, i)
		predict_comparison_list = get_comparison_list(test_set_predict, i)
		iter_accuracy = compare_accuracy(real_comparison_list, predict_comparison_list)
		accuracy += iter_accuracy
		print( "iteration--{0}: accuracy is {1}".format(i, iter_accuracy) )
	final_accuracy = accuracy / len(test_set_label)

	print(">>>>>>>>final_accuracy", final_accuracy)
	# 绘制图像
	test_offset = (test_set_predict.reshape(-1) - test_set_label.reshape(-1)) / test_set_label.reshape(-1)
	abs_test_offset = abs(test_offset)
	
	final_data = [ np.mean(test_offset), np.std(test_offset), np.mean(abs_test_offset), np.std(abs_test_offset), final_accuracy ]

	return final_data

def compare_similarity(list_1, list_2):
	assert(len(list_1) == len(list_2))

	num = 0
	for item in list_1:
		if item in list_2:
			num += 1
	return num/len(list_1)

def compare_accuracy(list_1, list_2):
	assert(len(list_1) == len(list_2))

	num = 0
	for i in range( len(list_1) ):
		if list_1[i] == list_2[i]:
			num += 1
	return num/len(list_1)

#给定value_list，并给出当前要比较的元素index，遍历整个list得出element-wise的大小关系，1代表当前位置元素比对标index大，0代表小于等于
def get_comparison_list(valuelist, index):
	comparison_result_list = []
	
	assert( index < len(valuelist) )
	compare_point = valuelist[index]

	for i in range( len(valuelist) ):

		if valuelist[i] > compare_point:
			comparison_result_list.append(1)
		else:
			comparison_result_list.append(0)

	return comparison_result_list

if __name__ == '__main__':
	
	basis_dict = {#'exp': rbf.basis.exp,
				'wen30': rbf.basis.wen30,
				#'wen31': rbf.basis.wen31,
				#'wen32': rbf.basis.wen32
				}
	
	order_list1 = [2]
	penalty_list1 = [0.15]
	size_list1 = [200, 300, 400, 500, 600, 700, 800, 900]

	number_list = [ele for ele in range(50)]

	input_csv_path = "数据-5-02.csv" # 原始数据文件目录
	raw_data = ImportAllData(input_csv_path)  # 避免重复读取CSV文件
	#final_sickPoints = getSickPoints(input_csv_path)

	output_csv_path = "5_31_rapid_comparison.csv"
	output = open(output_csv_path, 'r+', newline='')
	csv_writer = csv.writer(output, dialect='excel')

	for kernel in basis_dict.items():
		for order in order_list1:
			for penalty in penalty_list1:
				for size in size_list1:

					hyper_param = "" + kernel[0] + "-o" + str(order) + "-p" + str(penalty) + "-size" + str(size)
					csv_writer.writerow([hyper_param, 'mean', 'std', 'abs_mean', 'abs_std','comparison_accuracy'])
					print("-->", hyper_param)

					mean_statistic = [0 for ele in range(4+1)]

					for number in number_list:

						print("***call the basis: ", kernel[0])
						fig_name = "./figure_5_31/{0}-{1}.png".format(hyper_param, number)
						print("***", fig_name)

						specific_data = []

						#def model_test(fig_name, raw_data, basis=rbf.basis.phs2, order=1, penalty=0.1, size=5000, x_dim = 18, y_dim = 2, each_y_dim = 15)
						temp = model_test(fig_name, raw_data, kernel[1], order, penalty, size)
						for index, i in enumerate(temp):
							if index < 11:
								mean_statistic[index] = mean_statistic[index] + i
							else:
								specific_data = i

						gc.collect() #显式调用垃圾回收
						print("-->one iteration end")

					# mean_statistic =np.array( mean_statistic / len(number_list) )
					print("push final mean data into csv")
					temp = ["MEAN"]
					for i in mean_statistic:
						temp.append(i / len(number_list))

					csv_writer.writerow(temp)

					del mean_statistic
					del temp
					gc.collect()  # 显式地出发调用垃圾回收'''

