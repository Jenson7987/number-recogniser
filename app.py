"""
A REST service for classifying an image as a numerical digit.

Copyright (c) 2022 by Jenson Wong
"""
from base64 import decodebytes
from io import BytesIO
from logging import basicConfig, getLogger, INFO
from os.path import abspath, dirname

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image

from neural_net import NeuralNet

logger = getLogger(__name__)
basicConfig(level=INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

CURRENT_DIRECTORY = dirname(abspath(__file__))
WEIGHTS_AND_BIASES_FILENAME = "epoch_27.pickle"
WEIGHTS_AND_BIASES_FILE = f"{CURRENT_DIRECTORY}/{WEIGHTS_AND_BIASES_FILENAME}"


def normalise_pixel(pixel_level):
    # Scale and invert the pixel level from range [0, 255] to [0, 1]
    output = 1 - (pixel_level / 255)
    # Rounding small levels down to zero improves recognition
    return output if output >= 0.05 else 0


def convert_to_normalised_image(image_base64):
    image_bytes = decodebytes(bytes(image_base64, "utf-8"))
    raw_image = Image.open(BytesIO(image_bytes))
    image = raw_image.convert("L")  # Convert from colour to greyscale image
    resized_image = image.resize((28, 28), Image.LANCZOS)
    return [normalise_pixel(pixel) for pixel in resized_image.getdata()]


def create_app():
    logger.info("Starting application...")
    app = Flask(__name__)
    CORS(app)  # Allows access to this server from other domains
    app.config["CORS_HEADERS"] = "Content-Type"  # Allow JSON POST requests for browsers

    neural_network = NeuralNet.load(WEIGHTS_AND_BIASES_FILE)

    @app.route("/recognise-number", methods=["POST"])
    @cross_origin()
    def recognise_number():
        image = request.json["image"]  # Base64-encoded PNG image
        image_base64 = image[image.find(",") + 1 :]
        input_image = convert_to_normalised_image(image_base64)
        output = neural_network.classify(input_image)
        return jsonify({str(i): round(output[i] * 100, 1) for i in range(10)})

    return app


if __name__ == "__main__":
    application = create_app()
    application.run()
