import numpy as np
# class feed_forward_neural_network (object):
#
#     # create constructor
#     def __init__(self, input, hidden, output):
#         self.input = input
#         self.hidden = hidden
#         self.output = output

def normalize(x):
    return x / 255.0

def sigmoid(z):
    return (1.0/(1.0 + np.exp(-z)))

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()


num_steps = 1000
learning_rate = 1e-2
input_layer = np.ones([784])
hidden_layer = np.ones([392])
output_layer = np.ones([10])

input_to_hidden_synapses = np.random.randn(784, 392)
hidden_to_output_synapses = np.random.randn(392, 10)

# print(softmax([2,2,4]), sum(softmax([2,2,4])))

mnist_data = np.genfromtxt('MNIST.csv', delimiter=',', dtype=int, skip_header=1)

labels = mnist_data[:, 0]
mnist_data = normalize(mnist_data[:, 1:])
l1 = np.zeros(392)
for i in range(input_to_hidden_synapses.shape[1]):
    l1[i] = sigmoid(np.dot(input_layer, input_to_hidden_synapses.T[i]))
# print(l1)
l2 = np.zeros(10)
for i in range(hidden_to_output_synapses.shape[1]):
    l2[i] = sigmoid(np.dot(hidden_layer, hidden_to_output_synapses.T[i]))
output = softmax(l2)

prediction = list(output).index(max(output))

print(prediction)


# print(labels)
