"""
A feedforward neural network that is trained using backpropagation and one or more hidden layers.

Copyright (c) 2022 by Jenson Wong
"""
from dataclasses import dataclass
from logging import getLogger
from math import exp
from os.path import splitext
from pickle import dump, load
from random import gauss, shuffle
from typing import Optional

logger = getLogger(__name__)


def rand() -> float:
    return gauss(mu=0, sigma=1)


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


@dataclass
class TrainingSample:
    input: list[float]
    output: list[float]


class Weight:
    def __init__(self, name: str, value: float, input_neuron: "Neuron", output_neuron: "Neuron") -> None:
        self.name = name
        self.value = value
        self.del_weight_sum = 0.0
        self.del_weight_count = 0

        input_neuron.add_output_weight(self)
        self.input_neuron = input_neuron

        output_neuron.add_input_weight(self)
        self.output_neuron = output_neuron

    def add_del_weight(self, del_weight: float) -> None:
        self.del_weight_sum += del_weight
        self.del_weight_count += 1

    def update_with_average_del_weight(self) -> None:
        self.value = self.value + (self.del_weight_sum / self.del_weight_count)
        self.del_weight_sum = 0.0
        self.del_weight_count = 0


class Neuron:
    def __init__(
        self,
        name: str,
        out: float,
        has_bias: bool = False,
        target: Optional[float] = None,
    ) -> None:
        self.name = name
        self.out = out
        self.input_weights: list[Weight] = []
        self.output_weights: list[Weight] = []
        self.bias_weight = rand() if has_bias else 0
        self.del_bias_sum = 0
        self.del_bias_count = 0
        self.target = target
        self.del_error_by_net = 0

    def add_input_weight(self, weight: Weight) -> None:
        self.input_weights.append(weight)

    def add_output_weight(self, weight: Weight) -> None:
        self.output_weights.append(weight)

    def calculate_net(self) -> float:
        net = 0
        for weight in self.input_weights:
            net += weight.value * weight.input_neuron.out
        net += self.bias_weight
        return net

    def calculate_out(self) -> float:
        net = self.calculate_net()
        self.out = sigmoid(net)
        return self.out

    def update_weights(self) -> None:
        for weight in self.output_weights:
            weight.update_with_average_del_weight()
        if self.del_bias_count > 0:
            self.bias_weight = self.bias_weight + (self.del_bias_sum / self.del_bias_count)
            self.del_bias_sum = 0
            self.del_bias_count = 0

    def add_del_bias(self, del_bias: float) -> None:
        self.del_bias_sum += del_bias
        self.del_bias_count += 1

    def __repr__(self) -> str:
        return "Neuron(name=" + self.name + ")"


class NeuralNet:
    def __init__(self, layer_sizes: list[int], persist_to_file_path: str = None) -> None:
        self._neuron_layers: list[list[Neuron]] = NeuralNet._init_neurons_and_weights(layer_sizes)
        self._persist_to_file_path = persist_to_file_path

    @staticmethod
    def _init_neurons_and_weights(layer_sizes: list[int]) -> list[list[Neuron]]:
        def name(layer_index: int, node: int) -> str:
            if layer_index == 0:
                return f"i{node}"
            if layer_index == len(layer_sizes) - 1:
                return f"o{node}"
            return f"h{layer_index}_{node}"

        neuron_layers = []
        for layer, layer_size in enumerate(layer_sizes):
            neuron_layers.append([Neuron(name(layer, i), out=0, has_bias=(layer != 0)) for i in range(layer_size)])

        for neuron_layer, next_neuron_layer in zip(neuron_layers[:-1], neuron_layers[1:]):
            for neuron in neuron_layer:
                for next_neuron in next_neuron_layer:
                    NeuralNet._create_weight_between(neuron, next_neuron)

        return neuron_layers

    @staticmethod
    def _create_weight_between(neuron1: Neuron, neuron2: Neuron) -> None:
        Weight(
            f"{neuron1.name}-{neuron2.name}",
            value=rand(),
            input_neuron=neuron1,
            output_neuron=neuron2,
        )

    def train(
        self,
        training_samples: list[TrainingSample],
        epochs: int,
        mini_batch_size: int,
        learning_rate: float,
    ) -> None:
        sample_count = len(training_samples)
        logger.info(
            "Started training: samples=%d, epochs=%d, mini_batch_size=%d, learning_rate=%f",
            sample_count,
            epochs,
            mini_batch_size,
            learning_rate,
        )

        for epoch in range(epochs):
            shuffle(training_samples)
            mini_batches = [training_samples[k : k + mini_batch_size] for k in range(0, sample_count, mini_batch_size)]
            logger.info("Mini batches = %d", len(mini_batches))

            for i, mini_batch in enumerate(mini_batches):
                logger.info("Processing mini batch %d", i)
                self.update_mini_batch(mini_batch, learning_rate)

            logger.info("Epoch %d complete", epoch)
            self.persist_to_file_if_needed(epoch)

        logger.info("Training complete")

    def update_mini_batch(self, training_samples: list[TrainingSample], learning_rate: float) -> None:
        for training_sample in training_samples:
            self._backpropagation(training_sample, learning_rate)

        for neuron_layer in self._neuron_layers[:-1]:
            for neuron in neuron_layer:
                neuron.update_weights()

    def _backpropagation(self, training_sample: TrainingSample, learning_rate: float) -> None:
        self._feedforward(training_sample.input)

        output_neurons = self._neuron_layers[-1]
        for i, neuron in enumerate(output_neurons):
            neuron.target = training_sample.output[i]

        for neuron in output_neurons:
            neuron.del_error_by_net = -(neuron.target - neuron.out) * neuron.out * (1 - neuron.out)
            for weight in neuron.input_weights:
                del_total_error_by_weight = neuron.del_error_by_net * weight.input_neuron.out
                weight.add_del_weight(-learning_rate * del_total_error_by_weight)

            del_total_error_by_bias = neuron.del_error_by_net * neuron.bias_weight
            neuron.add_del_bias(-learning_rate * del_total_error_by_bias)

        reversed_hidden_layers = reversed(self._neuron_layers[1:-1])
        for hidden_layer in reversed_hidden_layers:
            for neuron in hidden_layer:
                del_total_error_by_out = 0
                for output_weight in neuron.output_weights:
                    del_total_error_by_out += output_weight.output_neuron.del_error_by_net * output_weight.value
                neuron.del_error_by_net = del_total_error_by_out * neuron.out * (1 - neuron.out)

                for input_weight in neuron.input_weights:
                    del_total_error_by_weight = neuron.del_error_by_net * input_weight.input_neuron.out
                    input_weight.add_del_weight(-learning_rate * del_total_error_by_weight)

                neuron.add_del_bias(-learning_rate * neuron.del_error_by_net)

    def _feedforward(self, input_data: list[float]) -> None:
        input_neurons = self._neuron_layers[0]
        for i, neuron in enumerate(input_neurons):
            neuron.out = input_data[i]

        hidden_and_output_layers = self._neuron_layers[1:]
        for neuron_layer in hidden_and_output_layers:
            for neuron in neuron_layer:
                neuron.calculate_out()

    def classify(self, input_data: list[float]) -> list[float]:
        self._feedforward(input_data)
        output_neurons = self._neuron_layers[-1]
        return [neuron.out for neuron in output_neurons]

    def persist_to_file_if_needed(self, epoch: int) -> None:
        if self._persist_to_file_path:
            root, extension = splitext(self._persist_to_file_path)
            file_path = f"{root}_{epoch}{extension}"
            try:
                logger.info("Persisting weights and biases file: %s", file_path)
                with open(file_path, "wb") as file:
                    dump(self, file)
            finally:
                logger.info("Finished persisting file: %s", file_path)

    @staticmethod
    def load(file_path: str) -> "NeuralNet":
        try:
            logger.info("Loading weights and biases file: %s", file_path)
            with open(file_path, "rb") as file:
                return load(file)
        finally:
            logger.info("Finished loading file: %s", file_path)
