import numpy as np


def sigmoid_function(x, deriv=False):
    if deriv:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


input_data_set = np.array([[0, 0, 1],
                          [0, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])

output_data_set = np.array([[0, 0, 1, 1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
synapse_0 = 2 * np.random.random((3, 1)) - 1

for i in range(100000):
    # forward propagation
    layer_0 = input_data_set
    layer_1 = sigmoid_function(np.dot(layer_0, synapse_0))

    layer_1_error = output_data_set - layer_1

    layer_1_delta = layer_1_error * sigmoid_function(layer_1, True)

    # update weights
    synapse_0 += np.dot(layer_0.T, layer_1_delta)

print('Output after trainning:\n{}'.format(layer_1))
