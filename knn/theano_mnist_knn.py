# theano_mnist_knn.py
import pandas as pd
import numpy as np
from theano import function
import theano.tensor as T
import time

# Read in data
training = pd.read_csv('MNIST_training.csv')
# training = pd.read_csv('../datasets/fashion-mnist_train.csv')
training_labels = training.iloc[:, 0]
training_data = training.drop('label', axis=1).as_matrix()
test = pd.read_csv('MNIST_test.csv')
# test = pd.read_csv('../datasets/fashion-mnist_test.csv')
test_labels = test.iloc[:, 0]
test_data = test.drop('label', axis=1).as_matrix()

X, Y = T.dvectors('X', 'Y')

distance = T.sqrt(T.sum(T.sqr(X - Y))) # equivalent to code below
# diff = X - Y
# diff_squared = T.sqr(diff)
# summed = T.sum(diff_squared)
# distance = T.sqrt(summed)
tdist = function([X, Y], distance)

def get_most_common(array):
	digits = np.zeros(10, dtype=int)
	for num in array:
		digits[num[1]] +=1
	return digits.argmax()

def evaluate_accuracy(knn_predictions):
	truth = np.array(test_labels) == np.array(knn_predictions)
	return sum(truth)

def dist(training, test):
	return np.sqrt(np.sum((training - test)**2))


start_time = time.time()
predictions = []
for test_case in test_data:
    distances = []
    for index, training_case in enumerate(training_data):
        distances.append((tdist(training_case, test_case), training_labels[index]))
    distances.sort(key=lambda tup: tup[0])
    predictions.append(get_most_common(distances[0:9]))
end_time = time.time()
elapsed_time = end_time - start_time

print("K = 9, Training Set Size: {}, Test Set Size: {}".format(len(training_labels), len(test_labels)))
print("Accuracy:")
print(evaluate_accuracy(predictions))
print("--")
print(len(test_labels))
print("Theano Runtime: {}".format(elapsed_time))


n_start_time = time.time()
n_predictions = []
for test_case in test_data:
    distances = []
    for index, training_case in enumerate(training_data):
        distances.append((dist(training_case, test_case), training_labels[index]))
    distances.sort(key=lambda tup: tup[0])
    n_predictions.append(get_most_common(distances[0:9]))
n_end_time = time.time()
n_elapsed_time = n_end_time - n_start_time

# print("Accuracy:")
# print(evaluate_accuracy(n_predictions))
# print("--")
# print(len(test_labels))
print("Numpy Runtime: {}".format(n_elapsed_time))

# a = [(0,2),(0,2),(0,1),(0,1),(0,1),(0,2)]
# print(get_most_common(a[0:2]))

# print(tdist(training_data[0], test_data[0]))
# print(tdist(training_data[0], test_data[40]))
