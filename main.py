import sys
import json

import numpy as np

from app.models.neural_network import NeuralNetwork
from app.data_loaders import mnist_loader


def start():
    training_data, validation_data, test_data = (
        mnist_loader.load_data_wrapper()
    )
    net = NeuralNetwork([784, 30, 10])
    net.sgd(
        list(training_data),
        epochs=300,
        mini_batch_size=10,
        alpha=3.0,
        evaluation_data=list(test_data),
        monitor_evaluation_accuracy=True
    )

def load(filename):
    """
    Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = NeuralNetwork(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

if __name__ == '__main__':
    start()
