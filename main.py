from app.models.NeuralNetwork import NeuralNetwork
from app.data_loaders import mnist_loader


def start():
    training_data, validation_data, test_data = (
        mnist_loader.load_data_wrapper()
    )
    net = NeuralNetwork([784, 30, 30, 30, 30, 30, 30, 30, 30, 10])
    net.sgd(
        list(training_data),
        epochs=30,
        mini_batch_size=10,
        alpha=3.0,
        test_data=list(test_data)
    )

if __name__ == '__main__':
    start()
