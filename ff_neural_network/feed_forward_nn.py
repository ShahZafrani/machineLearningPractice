# feed_forward_nn.py
# python 2.7.14
# 3 layer feed-forward neural network
# MNIST classier
# Shah Zafrani

from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

def normalize(x):
    return x / 255.0

def one_hot_encode(labels):
    output = []
    for label in labels:
        l_array = np.zeros(10)
        l_array[label] = 1
        output.append(l_array)
    return output

def predict(probs):
    one_hot = np.zeros(10)
    one_hot[probs.index(max(probs))] = 1
    return one_hot

def sigmoid(z, derivative=False):
    if derivative is False:
        return (1.0/(1.0 + np.exp(-z)))
    else:
        # taken from https://github.com/MichalDanielDobrzanski/DeepLearningPython35/blob/master/network2.py
        return sigmoid(z)*(1-sigmoid(z))

# def cross_entropy_cost(labels, output):
#     return -1 * labels

def softmax(x, derivative=False):
    if derivative is False:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    else:
        return x

if __name__ == "__main__":
    num_steps = 10 #1000
    learning_rate = 1e-2
    num_mini_batches = 25
    num_folds = 2 #10
    input_size = 784
    hidden_size = 28
    output_size = 10

    # print(softmax([2,2,4]), sum(softmax([2,2,4])))

    mnist_data = np.genfromtxt('MNIST.csv', delimiter=',', dtype=int, skip_header=1)

    labels = one_hot_encode(mnist_data[:, 0])
    mnist_data = normalize(mnist_data[:, 1:])

    kf = KFold(n_splits=num_folds)
    kf.get_n_splits(mnist_data)

    for train_index, test_index in kf.split(mnist_data):

        input_to_hidden_synapses = np.random.randn(input_size, hidden_size)
        hidden_to_output_synapses = np.random.randn(hidden_size, output_size)
        output_weights = np.random.randn(output_size)

        # input_to_hidden_synapses = np.zeros(shape=(input_size, hidden_size))
        # hidden_to_output_synapses = np.zeros(shape=(hidden_size, output_size))
        # print(train_index)
        mini_batch_size = len(train_index) / num_mini_batches
        error_costs = []
        for step in range(num_steps):
            # learning_rate = learning_rate * ((num_steps - step) / num_steps)
            outputs = []
            train_labels = []
            iterator = 0
            correct = 0
            activations = []
            for m in train_index:
                activation = []
                iterator += 1
                l1 = np.zeros(hidden_size)
                # assert(len(mnist_data[m]) == input_to_hidden_synapses.shape[0])
                activation.append(mnist_data[m])
                for i in range(hidden_size):
                    l1[i] = sigmoid(np.dot(mnist_data[m], input_to_hidden_synapses.T[i]))
                activation.append(l1)
                # print(l1)
                l2 = np.zeros(output_size)
                # assert(input_to_hidden_synapses.shape[1] == hidden_to_output_synapses.shape[0])
                for i in range(output_size):
                    l2[i] = sigmoid(np.dot(l1, hidden_to_output_synapses.T[i]))
                l3 = np.zeros(output_size)
                for i in range(output_size):
                    l3[i] = np.dot(l2[i], output_weights[i])
                activation.append(l2)
                output = softmax(l3)
                activation.append(output)
                prediction = predict(list(output))
                if(np.array_equal(prediction, labels[m])):
                    correct += 1
                outputs.append(output)
                activations.append(activation)
                train_labels.append(labels[m])
                # backprop
                if iterator % mini_batch_size == 0:
                    print(correct)
                    print(len(train_labels))
                    # compute error signals
                    batch_cost = 0
                    for a in range(len(activations)):
                        mean_square_error = (np.array(train_labels[a]) - np.array(activations[a][3])) ** 2
                        batch_cost += (sum(abs(mean_square_error)))
                        hidden_signal = np.zeros(shape=(hidden_size, output_size))
                        input_signal = np.zeros(shape=(input_size, hidden_size))

                        output_weights -= (learning_rate * mean_square_error)
                        for i in range(hidden_size):
                            for j in range(output_size):
                                hidden_signal[i, j] = sigmoid(np.dot(mean_square_error[j], activations[a][1][i]), derivative=True)
                        hidden_to_output_synapses -= (learning_rate * hidden_signal)

                        for i in range(input_size):
                            for j in range(hidden_size):
                                input_signal[i] += np.dot(sum(sigmoid(hidden_signal[j], derivative=True)), activations[a][0][i])
                        input_to_hidden_synapses -= (learning_rate * input_signal)
                    iterator = 0
                    correct = 0
                    activations = []
                    outputs = []
                    train_labels = []
                    error_costs.append(batch_cost)
                    print("batch cost: {}".format(batch_cost))
        plt.title("Gradient Descent Convergence")
        plt.plot(error_costs)

    # print(labels)
