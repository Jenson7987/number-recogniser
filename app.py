import os

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
import logging
from PIL import Image
import io
from neural_net import NeuralNet


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def normalise_pixel(x):
    output = 1 - (x / 255)
    return output if output >= 0.05 else 0


def bytes_to_matrix(image_bytes):
    raw_image = Image.open(io.BytesIO(image_bytes))
    image = raw_image.convert("L")
    image = image.resize((28, 28), Image.LANCZOS)
    return list(image.getdata())


def normalise_matrix(matrix):
    result = []
    for pixel in matrix:
        result.append(normalise_pixel(pixel))
    return result


def create_app() -> Flask:
    logger.info("Starting app...")

    app = Flask(__name__)
    CORS(app)
    app.config["CORS_HEADERS"] = "Content-Type"
    path = os.path.dirname(__file__)
    net = NeuralNet.load(path + "/epoch_27.pickle")

    @app.route("/recognise-number", methods=["POST"])
    @cross_origin()
    def recognise_number():
        data = request.json
        image = data["image"]
        image = image[image.find(",") + 1 :]
        image_bytes = bytes(image, "utf-8")
        matrix = bytes_to_matrix(base64.decodebytes(image_bytes))
        normalised_matrix = normalise_matrix(matrix)
        output = net.classify(normalised_matrix)
        return jsonify({str(i): round(output[i] * 100, 1) for i in range(10)})

    return app


if __name__ == "__main__":
    application = create_app()
    application.run()
