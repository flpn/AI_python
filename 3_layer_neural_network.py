import numpy as np


def sigmoid_function(x, deriv=False):
    if deriv:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


input_data_set = np.array([[0, 0, 1],
                          [0, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])

output_data_set = np.array([[0, 1, 1, 0]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
synapse_0 = 2 * np.random.random((3, 4)) - 1
synapse_1 = 2 * np.random.random((4, 1)) - 1


for i in range(100000):
    # Feed forward through layers 0, 1, and 2
    layer_0 = input_data_set
    layer_1 = sigmoid_function(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid_function(np.dot(layer_1, synapse_1))

    layer_2_error = output_data_set - layer_2

    # if i % 10000 == 0:
    #     print('Error: {}'.format(np.mean(np.abs(layer_2_error))))

    layer_2_delta = layer_2_error * sigmoid_function(layer_2, True)
    layer_1_error = layer_2_delta.dot(synapse_1.T)
    layer_1_delta = layer_1_error * sigmoid_function(layer_1, True)

    synapse_1 += layer_1.T.dot(layer_2_delta)
    synapse_0 += layer_0.T.dot(layer_1_delta)

print(layer_2)
