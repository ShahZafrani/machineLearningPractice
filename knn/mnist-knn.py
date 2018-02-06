# mnist-knn.py

import pandas as pd
import numpy as np

def knn(training_data, test_data, k):
	knn_predictions = []
	for test_case in test_data:
		dists = []
		for index, d in enumerate(training_data):
			dists.append([training_labels[index], distance(d, test_case)])
		dists_from_k = sorted(dists, key=lambda tup: tup[1])
		k_nearest = []
		for index in range(0, k):
			k_nearest.append(dists_from_k[index][0])
		# gather consensus from top k elements from list
		knn_predictions.append(get_most_common(k_nearest))
	return knn_predictions

def optimize_k_value(k_range):
	print("testing odd values up to {} for k. This may take a while, go read some xkcd comics and come back later...".format(k_range))
	k_results = []
	for kval in range(1, k_range, 2):
		accuracy = (evaluate_accuracy(knn(training_data, test_data, kval)) / float(len(test_labels))) * 100
		k_results.append(["kval: {}, accuracy: {} percent".format(kval, accuracy)])
	for result in k_results:
		print(result)

def get_most_common(array):
	digits = np.zeros(10, dtype=int)
	for num in array:
		digits[num] +=1
	return digits.argmax()

def distance(training, test):
	total = sum((training - test)**2)
	# for i in range(1, len(test)-1):
	# 	diff = training[i] - test[i]
	# 	total += diff**2
	return np.sqrt(total)

def evaluate_accuracy(knn_predictions):
	truth = np.array(test_labels) == np.array(knn_predictions)
	return sum(truth)

# read data
# training = pd.read_csv('fashion-mnist_train.csv')
training = pd.read_csv('MNIST_training.csv')
training_labels = training.iloc[:, 0]
training_data = training.drop('label', axis=1).as_matrix()
# test = pd.read_csv('fashion-mnist_test.csv')
test = pd.read_csv('MNIST_test.csv')
test_labels = test.iloc[:, 0]
test_data = test.drop('label', axis=1).as_matrix()

optimize_k_value(14)
# print(len(test_data))
# print(len(training_data))
