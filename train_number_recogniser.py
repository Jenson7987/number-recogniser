"""
Trains a neural network to recognise handwritten digits from MNIST image data.

Copyright (c) 2022 by Jenson Wong
"""
from logging import basicConfig, getLogger, INFO
from os.path import abspath, dirname

from mnist_loader import MnistDataLoader
from neural_net import NeuralNet

logger = getLogger(__name__)
basicConfig(level=INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

CURRENT_DIRECTORY = dirname(abspath(__file__))
WEIGHTS_AND_BIASES_FILENAME = "epoch.pickle"
WEIGHTS_AND_BIASES_FILE = f"{CURRENT_DIRECTORY}/{WEIGHTS_AND_BIASES_FILENAME}"

MNIST_DATA_FILENAME = "mnist.pkl.gz"
MNIST_DATA_FILE = f"{CURRENT_DIRECTORY}/{MNIST_DATA_FILENAME}"


if __name__ == "__main__":
    training_data = MnistDataLoader(pickled_mnist_gzip_file_path=MNIST_DATA_FILE).load().get_training_data()
    neural_network = NeuralNet(layer_sizes=[784, 30, 10], persist_to_file_path=WEIGHTS_AND_BIASES_FILE)
    neural_network.train(training_samples=training_data, epochs=30, mini_batch_size=10, learning_rate=3)
