# feed_forward_nn.py
# python 2.7.14
# 3 layer feed-forward neural network
# MNIST classier
# Shah Zafrani

from sklearn.model_selection import KFold
import numpy as np

def normalize(x):
    return x / 255.0

def sigmoid(z, derivative=False):
    if derivative is False:
        return (1.0/(1.0 + np.exp(-z)))
    else:
        # taken from https://github.com/MichalDanielDobrzanski/DeepLearningPython35/blob/master/network2.py
        return sigmoid(z)*(1-sigmoid(z))

def softmax(x, derivative=False):
    if derivative is False:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    else:
        return x


num_steps = 1 #1000
learning_rate = 1e-2
num_folds = 2 #10
input_size = 784
hidden_size = 30
output_size = 10

input_to_hidden_synapses = np.random.randn(input_size, hidden_size)
hidden_to_output_synapses = np.random.randn(hidden_size, output_size)

# print(softmax([2,2,4]), sum(softmax([2,2,4])))

mnist_data = np.genfromtxt('MNIST.csv', delimiter=',', dtype=int, skip_header=1)

labels = mnist_data[:, 0]
mnist_data = normalize(mnist_data[:, 1:])

kf = KFold(n_splits=num_folds)
kf.get_n_splits(mnist_data)

for train_index, test_index in kf.split(mnist_data):
    # print(train_index)
    for step in range(num_steps):
        for m in train_index:
            l1 = np.zeros(hidden_size)
            # assert(len(mnist_data[m]) == input_to_hidden_synapses.shape[0])
            for i in range(input_to_hidden_synapses.shape[1]):
                l1[i] = sigmoid(np.dot(mnist_data[m], input_to_hidden_synapses.T[i]))
            # print(l1)
            l2 = np.zeros(output_size)
            # assert(input_to_hidden_synapses.shape[1] == hidden_to_output_synapses.shape[0])
            for i in range(hidden_to_output_synapses.shape[1]):
                l2[i] = np.dot(l1, hidden_to_output_synapses.T[i])
            output = softmax(l2)

            prediction = list(output).index(max(output))
            if m % 100 == 0:
                print("prediction {} : actual {}".format(prediction, labels[m]))


# print(labels)
