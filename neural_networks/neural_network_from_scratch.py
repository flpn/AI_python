import random
import numpy as np


class NeuralNetwork:
    def __init__(self, neuron_quantity):
        self.layers_quantity = len(neuron_quantity)
        self.neuron_quantity = neuron_quantity
        self.biases = [np.random.randn(y, 1) for y in neuron_quantity[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        
        return a
    
    def update_mini_batch(self, mini_batch, eta):
        pass
    
    def evaluate(self, test_data):
        pass
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        n_training_data = len(training_data)
        
        if test_data:
            test_data = list(test_data)
            n_test_data = len(test_data)
        
        for i in range(epochs):
            random.shuffle(training_data)
            
            mini_batches = [training_data[j:j + mini_batch_size] for j in range(0, n_training_data, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                
            if test_data:
                print('Epoch {}: {}/{}'.format(i, self.evaluate(test_data), n_test_data))
            else:
                print('Epoch {} finalizada!'.format(i))

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
