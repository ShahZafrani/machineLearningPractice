# mnist-knn.py

import pandas as pd
import numpy as np

def knn(training_data, test_data, k):
	cases_correct = 0
	for test_case in test_data:
		dists = []
		for d in training_data:
			dists.append([d[0], distance(d, test_case)])
		dists_from_k = sorted(dists, key=lambda tup: tup[1])
		k_nearest = []
		for index in range(0, k):
			k_nearest.append(dists_from_k[index][0])
		# gather consensus from top k elements from list
		knn_guess = get_most_common(k_nearest)
		# print("knn guess: {}, correct answer: {}".format(knn_guess, test_case[0]))
		if knn_guess == test_case[0]:
			cases_correct +=1
	# print("cases correct: {}".format(cases_correct))
	accuracy = cases_correct / float( len(test_data))
	return accuracy

def optimize_k_value(k_range):
	print("testing odd values up to {} for k. This may take a while, go read some xkcd comics and come back later...".format(k_range))
	k_results = []
	for kval in range(1, k_range, 2):
		accuracy = knn(training_data, test_data, kval)
		k_results.append(["kval: {}, accuracy: {}".format(kval, accuracy)])
	for result in k_results:
		print(result)

def get_most_common(array):
	digits = np.zeros(10, dtype=int)
	for num in array:
		digits[num] +=1
	return digits.argmax()

def distance(training, test):
	total = 0
	for i in range(1, len(test)-1):
		diff = training[i] - test[i]
		total += diff**2
	return np.sqrt(total)

# read data
training_data = pd.read_csv('MNIST_training.csv').as_matrix()
test_data = pd.read_csv('MNIST_test.csv').as_matrix()

optimize_k_value(14)
