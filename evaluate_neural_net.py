"""
Calculates the accuracy and cost function output of a neural network across training epochs.

Copyright (c) 2022 by Jenson Wong
"""
from glob import glob
from os.path import abspath, dirname, split

from mnist_loader import MnistDataLoader, TrainingDigit
from neural_net import NeuralNet


CURRENT_DIRECTORY = dirname(abspath(__file__))
PICKLED_MNIST_GZIP_FILE = f"{CURRENT_DIRECTORY}/mnist.pkl.gz"
EPOCH_WEIGHTS_AND_BIASES_FILES = f"{CURRENT_DIRECTORY}/training/epoch_*.pickle"


def evaluate_percentage_accuracy_and_cost(net: NeuralNet, test_data: list[TrainingDigit]) -> [float, float]:
    correct_predictions: int = 0
    sum_of_squared_differences: float = 0
    samples = len(test_data)
    for i in range(samples):
        result = net.classify(test_data[i].input)
        m = max(result)
        correct_predictions += int(test_data[i].output == result.index(m))
        expected_output = [0.0] * len(result)
        expected_output[test_data[i].output] = 1.0
        sum_of_squared_differences += sum(
            [(expected - actual) ** 2 for expected, actual in zip(expected_output, result)]
        )
    return correct_predictions / samples, sum_of_squared_differences / (2 * samples)


def evaluate_neural_network(epoch_weights_and_biases_file_pattern_path: str, test_data: list[TrainingDigit]) -> None:
    directory, file_pattern = split(epoch_weights_and_biases_file_pattern_path)
    csv_file_path = f"{directory}/epoch_accuracy_and_cost.csv"
    print(f"Writing output file: {csv_file_path}")
    with open(csv_file_path, "w") as csv:
        output = "epoch,accuracy,cost"
        print(output)
        csv.write(output + "\n")

        epochs = len(glob(epoch_weights_and_biases_file_pattern_path))
        for epoch in range(epochs):
            filename = file_pattern.replace("*", str(epoch))
            net = NeuralNet.load(f"{directory}/{filename}")
            percentage_accuracy, cost = evaluate_percentage_accuracy_and_cost(net, test_data)
            output = f"{epoch},{percentage_accuracy},{cost}"
            print(output)
            csv.write(output + "\n")
    print(f"Finished writing output file: {csv_file_path}")


if __name__ == "__main__":
    mnist_test_data = MnistDataLoader(PICKLED_MNIST_GZIP_FILE).load().get_test_data()
    evaluate_neural_network(EPOCH_WEIGHTS_AND_BIASES_FILES, mnist_test_data)
