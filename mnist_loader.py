"""
Loads the MNIST handwritten digits image data.

Copyright (c) 2022 by Jenson Wong
"""
from gzip import open
from logging import getLogger
from pickle import load

from neural_net import TrainingSample

logger = getLogger(__name__)


class MnistDataLoader:
    def __init__(self, pickled_mnist_gzip_file_path: str) -> None:
        self._pickled_mnist_gzip_file_path = pickled_mnist_gzip_file_path
        self._training_data = []
        self._validation_data = []
        self._test_data = []

    def load(self) -> "MnistDataLoader":
        """Loads MNIST data stored as a Python pickle gzip file, originally from:
        https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz

        'training_data' consists of a tuple with two entries:
        - a numpy ndarray of 50,000 images of 784 pixels (28 * 28)
        - a numpy ndarray of 50,000 values, between 0 and 9, indicating the image digit

        'validation_data' and 'test_data' are similar but contain 10,000 images each.
        """
        logger.info("Loading MNIST data: %s", self._pickled_mnist_gzip_file_path)
        with open(self._pickled_mnist_gzip_file_path, "rb") as file:
            training_data, validation_data, test_data = load(file, encoding="latin1")

        training_inputs = [d.tolist() for d in training_data[0]]
        training_results = [self.normalise_result(y) for y in training_data[1]]
        self._training_data = [TrainingSample(i, r) for i, r in zip(training_inputs, training_results)]

        validation_inputs = [d.tolist() for d in validation_data[0]]
        self._validation_data = [TrainingSample(i, r) for i, r in zip(validation_inputs, validation_data[1])]

        test_inputs = [d.tolist() for d in test_data[0]]
        self._test_data = [TrainingSample(i, r) for i, r in zip(test_inputs, test_data[1])]
        logger.info("Finished loading MNIST data: %s", self._pickled_mnist_gzip_file_path)
        return self

    def get_training_data(self) -> list[TrainingSample]:
        return self._training_data

    def get_validation_data(self) -> list[TrainingSample]:
        return self._validation_data

    def get_test_data(self) -> list[TrainingSample]:
        return self._test_data

    @staticmethod
    def normalise_result(n: int) -> list[float]:
        result = [0.0] * 10
        result[n] = 1.0
        return result
