from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from random import randint
import base64
import logging
from PIL import Image
import numpy as np
import io
import mnist_loader
import network


logger = logging.getLogger(__name__)

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.info("Loading data...")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
logger.info("Loaded data.")
net = network.Network([784, 30, 10])
logger.info("Training neural network...")
net.SGD(training_data, 1, 10, 3.0)
logger.info("Trained neural network.")


@app.route("/")
def index():
    return "Hello World!"


def normalise_pixel(x):
    output = 1 - (x / 255)
    return output if output >= 0.05 else 0


def random_confidence():
    return randint(0, 100)


def bytes_to_matrix(image_bytes):
    raw_image = Image.open(io.BytesIO(image_bytes))
    image = raw_image.convert("L")
    image = image.resize((28, 28), Image.LANCZOS)
    image.save("J:/Projects/imageToSave.png")
    greyscale_matrix = np.array(image)
    return greyscale_matrix


@app.route("/recognise-number", methods=["POST"])
@cross_origin()
def recognise_number():
    data = request.json
    image = data['image']
    image = image[image.find(',') + 1:]
    image_bytes = bytes(image, "utf-8")
    matrix = bytes_to_matrix(base64.decodebytes(image_bytes))
    normalised_matrix = np.vectorize(normalise_pixel, otypes=[float])(matrix)
    normalised_matrix.resize((784, 1))
    output = net.feedforward(normalised_matrix)
    return jsonify({
        '0': int(output[0][0] * 100),
        '1': int(output[1][0] * 100),
        '2': int(output[2][0] * 100),
        '3': int(output[3][0] * 100),
        '4': int(output[4][0] * 100),
        '5': int(output[5][0] * 100),
        '6': int(output[6][0] * 100),
        '7': int(output[7][0] * 100),
        '8': int(output[8][0] * 100),
        '9': int(output[9][0] * 100),
    })


if __name__ == '__main__':
    logger.info("Starting app...")
    app.run()
