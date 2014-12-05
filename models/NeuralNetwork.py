__author__ = 'olavgjerde'

from numpy.random import random

class NeuralNetwork():

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [random(y, 1) for y in sizes[1:]]
        self.weights = [random(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
