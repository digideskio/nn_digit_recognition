
import json
import numpy as np
from numpy.random.mtrand import randn
from numpy.random.mtrand import shuffle

from ..math.sigmoid import sigmoid_vec, sigmoid_prime_vec
from ..math.cost_functions import CrossEntropyCost


class NeuralNetwork():

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.default_weight_initializer()

    def default_weight_initializer(self):
        """
                Initialize each weight using a Gaussian distribution with mean 0
                and standard deviation 1 over the square root of the number of
                weights connecting to the same neuron. Initialize the biases
                using a Gaussian distribution with mean 0 and standard
                deviation 1.
                Note that the first layer is assumed to be an input layer, and
                by convention we won't set any biases for those neurons, since
                biases are only ever used in computing the outputs from later
                layers.
                """
        self.biases = [randn(y, 1) for y in self.sizes[1:]]
        self.weights = [randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """
                Initialize the weights using a Gaussian distribution with mean 0
                and standard deviation 1. Initialize the biases using a
                Gaussian distribution with mean 0 and standard deviation 1.
                Note that the first layer is assumed to be an input layer, and
                by convention we won't set any biases for those neurons, since
                biases are only ever used in computing the outputs from later
                layers.
                This weight and bias initializer uses the same approach as in
                Chapter 1, and is included for purposes of comparison. It
                will usually be better to use the default weight initializer
                instead.
                """
        self.biases = [randn(y, 1) for y in self.sizes[1:]]
        self.weights = [randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feed_forward(self, a):
        """ Return the output of the network if "a" is input. """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w, a) + b)
        return a

    def sgd(
            self,
            training_data,
            epochs=30,
            mini_batch_size=10,
            alpha=3.0,
            lmbda=0.1,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False
    ):
        """
                Train the neural network using mini-batch stochastic gradient
                descent. The ``training_data`` is a list of tuples ``(x, y)``
                representing the training inputs and the desired outputs. The
                other non-optional parameters are self-explanatory, as is the
                regularization parameter ``lmbda``. The method also accepts
                ``evaluation_data``, usually either the validation or test
                data. We can monitor the cost and accuracy on either the
                evaluation data or the training data, by setting the
                appropriate flags. The method returns a tuple containing four
                lists: the (per-epoch) costs on the evaluation data, the
                accuracies on the evaluation data, the costs on the training
                data, and the accuracies on the training data. All values are
                evaluated at the end of each training epoch. So, for example,
                if we train for 30 epochs, then the first element of the tuple
                will be a 30-element list containing the cost on the
                evaluation data at the end of each epoch. Note that the lists
                are empty if the corresponding flag is not set.
                """
        n_data = None
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            shuffle(training_data)

            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, alpha, lmbda, len(training_data))

            print("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print(
                    "Accuracy on training data: {} / {}"
                    .format(accuracy, n)
                )
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(
                    "Accuracy on evaluation data: {} / {}"
                    .format(self.accuracy(evaluation_data), n_data)
                )
        return (
            evaluation_cost,
            evaluation_accuracy,
            training_cost,
            training_accuracy
        )

    def update_mini_batch(self, mini_batch, alpha, lmbda, n):
        """
                Update the network's weights and biases by applying gradient
                descent using backpropagation to a single mini batch. The
                ``mini_batch`` is a list of tuples ``(x, y)``, ``alpha`` is the
                learning rate, ``lmbda`` is the regularization parameter, and
                ``n`` is the total size of the training data set.
                """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb +dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw +dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [
            (1 - alpha * (lmbda/n))* w -(alpha / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b -( alpha /len(mini_batch) ) *nb
                       for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """
                Return a tuple ``(nabla_b, nabla_w)`` representing the
                gradient for the cost function C_x.  ``nabla_b`` and
                ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
                to ``self.biases`` and ``self.weights``.
                """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime_vec(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            spv = sigmoid_prime_vec(z)
            delta = np.dot(self.weights[- l +1].transpose(), delta) * spv
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[- l -1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """
                Return the number of test inputs for which the neural
                network outputs the correct result. Note that the neural
                network's output is assumed to be the index of whichever
                neuron in the final layer has the highest activation.
                """
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
                Return the vector of partial derivatives \partial C_x /
                \partial a for the output activations.
                """
        return output_activations - y

    def accuracy(self, data, convert=False):
        """
                Return the number of inputs in ``data`` for which the neural
                network outputs the correct result. The neural network's
                output is assumed to be the index of whichever neuron in the
                final layer has the highest activation.

                The flag ``convert`` should be set to False if the data set is
                validation or test data (the usual case), and to True if the
                data set is the training data. The need for this flag arises
                due to differences in the way the results ``y`` are
                represented in the different data sets.  In particular, it
                flags whether we need to convert between the different
                representations.  It may seem strange to use different
                representations for the different data sets.  Why not use the
                same representation for all three data sets?  It's done for
                efficiency reasons -- the program usually evaluates the cost
                on the training data and the accuracy on other data sets.
                These are different types of computations, and using different
                representations speeds things up.  More details on the
                representations can be found in
                mnist_loader.load_data_wrapper.
                """
        if convert:
            results = [(np.argmax(self.feed_forward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feed_forward(x)), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """
                Return the total cost for the data set ``data``.  The flag
                ``convert`` should be set to False if the data set is the
                training data (the usual case), and to True if the data set is
                the validation or test data.  See comments on the similar (but
                reversed) convention for the ``accuracy`` method, above.
                """
        cost = 0.0
        for x, y in data:
            a = self.feed_forward(x)
            if convert:
                y = self.vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data) ) * sum(
            np.linalg.norm(w)**2 for w in self.weights
        )
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def vectorized_result(self, j):
        """
                Return a 10-dimensional unit vector with a 1.0 in the j'th position
                and zeroes elsewhere.  This is used to convert a digit (0...9)
                into a corresponding desired output from the neural network.
                """
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e