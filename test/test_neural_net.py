"""
Test neural network using the XOR function.

Copyright (c) 2022 by Jenson Wong
"""
from pathlib import Path
from pytest import approx

from neural_net import NeuralNet, TrainingSample


def xor_neural_network(persist_to_file_path: str = None):
    training_samples = [
        TrainingSample(input=[0.0, 0.0], output=[0.0]),
        TrainingSample(input=[0.0, 1.0], output=[1.0]),
        TrainingSample(input=[1.0, 0.0], output=[1.0]),
        TrainingSample(input=[1.0, 1.0], output=[0.0]),
    ]
    net = NeuralNet(layer_sizes=[2, 5, 1], persist_to_file_path=persist_to_file_path)
    net.train(training_samples=training_samples, epochs=10_000, mini_batch_size=4, learning_rate=0.5)
    return net


def assert_xor_outputs_from(net: NeuralNet) -> None:
    assert net.classify([0.0, 0.0]) == approx([0.0], abs=0.1)
    assert net.classify([0.0, 1.0]) == approx([1.0], abs=0.1)
    assert net.classify([1.0, 0.0]) == approx([1.0], abs=0.1)
    assert net.classify([1.0, 1.0]) == approx([0.0], abs=0.1)


def test_neural_network_learns_xor_function():
    assert_xor_outputs_from(xor_neural_network())


def test_save_and_load_xor_neural_network(tmp_path: Path):
    directory_path = str(tmp_path.resolve())
    xor_neural_network(persist_to_file_path=directory_path + "/epoch.pickle")
    assert_xor_outputs_from(NeuralNet.load(file_path=directory_path + "/epoch_9999.pickle"))
