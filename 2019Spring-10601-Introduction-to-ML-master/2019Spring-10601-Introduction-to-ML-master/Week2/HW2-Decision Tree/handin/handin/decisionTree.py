import math
import csv
import sys
import numpy as np




# Main class training DT for 1. part
class decisionTree:
	"""
	Involves 5 main function steps:
	<1> create_labels_binary: read label dataset as dictionary label_binary
	<2> calculation functions: margin_entropy, conditional_entropy and information_gain Y
	<3> train *keypoint*: Use information gain Y to split attrs and build tree recursively, complete recursion with leaf_datasets as new training datasets
	<4> predict: use majority vote to predict label for split_attr
	<5> printTree: print the tree as in 2.3.4
	"""
	def __init__(self, datasets, depth, max_depth, attrs, used_attrs, msg = ""):
		self.datasets = datasets
		self.attrs = attrs
		self.used_attrs = used_attrs
		self.depth = depth
		self.max_depth = max_depth
		self.msg = msg
		self.split_attr = None
		self.split_attr_values = None
		self.leaves = []
		self.labels_binary = self.create_labels_binary(datasets)
		self.majority_vote_result = help_function.majority_vote_classifier(self.labels_binary)

	# <1> read label dataset as dictionary label_binary
	def create_labels_binary(self, datasets):
		labels_value = datasets[:, -1]
		labels_binary = {}
		for value in labels_value:
			if value not in labels_binary:
				labels_binary[value] = 1
			else:
				labels_binary[value] += 1
		return labels_binary
	
	# <2> Define 2 functions in calculation entropy-
	# 1. margin_entropy
	# 2. conditional_entropy
	# 3. information_gain
	def margin_entropy(self, labels_binary):
		margin_entropy = 0
		for label in labels_binary.values():
			margin_entropy += help_function.cal_entropy(label, labels_binary)
		return margin_entropy
	
	def conditional_entropy(self, attr_label):
		N = attr_label.shape[0]
		attr_label_binary = {}
		for tup in attr_label:
			attr_value = tup[0]
			attr_label = tup[1]
			if attr_value not in attr_label_binary.keys():
				attr_label_binary[attr_value] = {}
			if attr_label in attr_label_binary[attr_value].keys():
				attr_label_binary[attr_value][attr_label] += 1
			else:
				attr_label_binary[attr_value][attr_label] = 1
		conditional_entropy = 0
		for attr_value in attr_label_binary.keys():
			P = sum(attr_label_binary[attr_value].values()) / N
			specific_entropy = 0
			for value in attr_label_binary[attr_value].values():
				specific_entropy += help_function.cal_entropy(value, attr_label_binary[attr_value])
			conditional_entropy += P * specific_entropy
		return conditional_entropy

	def information_gain(self, margin_entropy, conditional_entropy):
		return margin_entropy - conditional_entropy
	
	# <3> train
	# - return None, if DT is already fixed: margin_entropy == 0
	# - return None, if all the attributes are used: attrs == used_attrs;
	# - return None, if DT's depth is enough: depth == max_depth;
	#   else:
	#       Use information gain Y to split attrs and build tree recursively 
	def train(self):
		margin_entropy = self.margin_entropy(self.labels_binary)
		if margin_entropy == 0:
			return
		if len(self.attrs) == len(self.used_attrs) or self.depth == self.max_depth:
			return
		# find best attribute as root
		max_Y = 0
		split_attr_index = -1
		for i in range(len(self.attrs)):
			if self.attrs[i] not in self.used_attrs:
				conditional_entropy = self.conditional_entropy(self.datasets[:, [i, -1]])
				Y = self.information_gain(margin_entropy, conditional_entropy)
				if Y >= max_Y:
					split_attr_index = i
					max_Y = Y
		self.split_attr = self.attrs[split_attr_index]
		self.split_attr_values = list(set(self.datasets[:, split_attr_index]))
		# split the root into leaves
		for value in self.split_attr_values:
			leaf_datasets = []
			for dataset in self.datasets:
				if dataset[split_attr_index] == value:
					leaf_datasets.append(dataset)
			# renew the data as leaf_data
			leaf_datasets = np.array(leaf_datasets)
			msg = "{} = {}: ".format(self.split_attr, value)
			self.leaves.append(decisionTree(leaf_datasets, self.depth + 1, self.max_depth, self.attrs, self.used_attrs + [self.split_attr], msg))
		# recursion on each depth
		for leaf in self.leaves:
			leaf.train()
		return

	# <4> predict the label based on the training DT and attribute datasets 
	def predict(self, attr_dictionary_list):
		if self.split_attr == None:
			return self.majority_vote_result
		attr_value = attr_dictionary_list[self.split_attr]
		for i in range(len(self.split_attr_values)):
			if attr_value == self.split_attr_values[i]:
				return self.leaves[i].predict(attr_dictionary_list)
		return self.majority_vote_result
	
	# <5> print the tree
	def printTree(self):
		print("{}{}".format("|" * self.depth + self.msg, self.labels_binary))
		# recursion on each depth
		for leaf in self.leaves:
			leaf.printTree()
		return


class help_function:
	# function for 2. and 3. parts - read infile
	def read_dataset(csv_file):
		"""
		Input: .csv train or test dataset, which contains N sets and M attrs 
		return: stored 2 lists -
			<1> attr_dictionary_list, as a list of N*M dictionaries
				(e.g. [{attr1:A or B}, {attr2:C}, {attr1:A}, {attr2:D}, ... ])
			<2> label_list (e.g. [A, B, A, B, ... ])
		"""
		with open(csv_file, "r") as infile:
			reader = csv.reader(infile)
			header = next(reader)
			label = header[-1]
			attrs = header[:-1]
			datasets = np.array([row for row in reader])
		attr_dictionary_list = []
		label_list = []
		for dataset in datasets:
			attr_dictionary ={}
			for i in range(len(attrs)):
				attr_dictionary[attrs[i]] = dataset[i]
			attr_dictionary_list.append(attr_dictionary)
			label_list.append(dataset[-1])
		return attr_dictionary_list, label_list
	
	# function to calculate entropy for 1. part
	def cal_entropy(a, b):
		"""
		Input: int(a) and dictionary b
		Assume: a == b.values()
		Return: entropy
		"""
		if a == 0:
			return 0
		else:
			return -(a / sum(b.values())) * math.log2(a / sum(b.values()))
	
	# function for majority vote classifier for binary-dictionary {'a':n, 'b':m}
	def majority_vote_classifier(dict):
		"""
		Input: binary-dictionary d = {'a':n, 'b':m}
		Return: maximum key
		"""
		max_val = max(dict.values())
		for key in dict.keys():
			if dict[key] == max_val:
				return key
				break
	

class decisionTree_lite:
	"""
	The decisionTree_lite is basically the same DT as class decisionTree.
	But only 2 inputs (train_infile and max_depth) in order to fit the input task, so is called "lite".
	This lite one will call the complex one by decisionTree.functions so that it can finish the job.
	#Involved 4 steps:
	 <1> read_train_infile: read .csv and return with data, attrs and label
	 <2> train: same as decisionTree.train()
	 <3> predict: same as decisionTree.predict(attr_dictionary_list)
	 <4> printTree: same as decisionTree.printTree()   

	"""
	def __init__(self, train_infile, max_depth):
		self.datasets, self.attrs, self.label = self.read_train_infile(train_infile)
		self.complex = decisionTree(self.datasets, 0, max_depth, self.attrs, [])
	
	def read_train_infile(self, train_infile):
		with open(train_infile, "r") as infile:
			reader = csv.reader(infile)
			header = next(reader)
			label = header[-1]
			attrs = header[:-1]
			datasets = np.array([row for row in reader])
		return datasets, attrs, label

	def train(self):
		self.complex.train()
	
	def predict(self, attr_dictionary_list):
		return self.complex.predict(attr_dictionary_list)
	
	def printTree(self):
		self.complex.printTree()


def main():
	"""
	main function of decision tree, which contains 6 arguments:
	<train input>, <test input>, <max depth>,
	<train out>, <test out> and <metrics out>
	* 4 parts are involved:
		1. train the decision tree with train dataset (training DT)
		2. predict output for train dataset: train_label_outfile
		3. predict output for test dataset: test_label_outfile
		4. calculate error rate for train and test dataset: metrics_outfile
	"""
	train_infile = sys.argv[1]
	test_infile = sys.argv[2]
	max_depth = int(sys.argv[3])
	train_label_outfile = sys.argv[4]
	test_label_outfile = sys.argv[5]
	metrics_outfile = sys.argv[6]

	# 1. train the decision tree with train dataset (training DT)
	tree = decisionTree_lite(train_infile, max_depth)
	tree.train()
	tree.printTree()

	# 2. Read train dataset and predict output: train_label_outfile
	output_file = open(train_label_outfile, "w")
	errorNum = 0
	attr_dictionary_list, label_list = help_function.read_dataset(train_infile)
	for attr_dictionary, label in zip(attr_dictionary_list, label_list):
		prediction = tree.predict(attr_dictionary)
		if prediction != label:
			errorNum += 1
		output_file.write("{}\n".format(prediction))
	output_file.close()
	train_error_rate = errorNum / len(label_list)

	# 3. Read test dataset and predict output: test_label_outfile
	output_file = open(test_label_outfile, "w")
	errorNum = 0
	attr_dictionary_list, label_list = help_function.read_dataset(test_infile)
	for attr_dictionary, label in zip(attr_dictionary_list, label_list):
		prediction = tree.predict(attr_dictionary)
		if prediction != label:
			errorNum += 1
		output_file.write("{}\n".format(prediction))
	output_file.close()
	test_error_rate = errorNum / len(label_list)

	# 4. calculate error rate for train and test dataset: metrics_outfile
	metrics_file = open(metrics_outfile, "w")
	metrics_file.write("error(train): {}\n".format(train_error_rate))
	metrics_file.write("error(test): {}\n".format(test_error_rate))
	metrics_outfile.close()


if __name__ == "__main__":
	main()
