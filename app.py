"""
A REST service for classifying an image as a numerical digit.

Copyright (c) 2022 by Jenson Wong
"""
from base64 import decodebytes
from io import BytesIO
from logging import basicConfig, getLogger, INFO
from math import isqrt
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


def normalize_size_and_center(image: list[float]) -> list[float]:
    """Normalize the size of the digit and center it, similar to MNIST preprocessing."""
    rows = columns = isqrt(len(image))
    
    # Find bounding box of the digit
    min_x = max_x = min_y = max_y = None
    for row in range(rows):
        for col in range(columns):
            if image[row * columns + col] != 0:
                if min_x is None or col < min_x: min_x = col
                if max_x is None or col > max_x: max_x = col
                if min_y is None or row < min_y: min_y = row
                if max_y is None or row > max_y: max_y = row
    
    # Handle empty image
    if min_x is None:
        return image
    
    # Calculate digit dimensions
    digit_width = max_x - min_x + 1
    digit_height = max_y - min_y + 1
    
    # Target size (leave margin like MNIST - 20x20 digit in 28x28 canvas)
    target_size = 20
    
    # Calculate scale factor (preserve aspect ratio)
    scale = min(target_size / digit_width, target_size / digit_height)
    
    # Calculate new dimensions after scaling
    new_width = digit_width * scale
    new_height = digit_height * scale
    
    # Calculate centering offsets
    offset_x = (columns - new_width) / 2
    offset_y = (rows - new_height) / 2
    
    # Create new centered and scaled image
    new_image = [0.0] * len(image)
    
    for new_row in range(rows):
        for new_col in range(columns):
            # Map back to original coordinates
            orig_col = (new_col - offset_x) / scale + min_x
            orig_row = (new_row - offset_y) / scale + min_y
            
            # Check if we're within the original digit bounds
            if min_x <= orig_col <= max_x and min_y <= orig_row <= max_y:
                # Bilinear interpolation
                x0 = int(orig_col)
                x1 = min(x0 + 1, columns - 1)
                y0 = int(orig_row)
                y1 = min(y0 + 1, rows - 1)
                
                # Calculate interpolation weights
                wx = orig_col - x0
                wy = orig_row - y0
                
                # Get the 4 surrounding pixel values
                val_00 = image[y0 * columns + x0]
                val_10 = image[y0 * columns + x1]
                val_01 = image[y1 * columns + x0]
                val_11 = image[y1 * columns + x1]
                
                # Bilinear interpolation formula
                interpolated = (
                    val_00 * (1 - wx) * (1 - wy) +
                    val_10 * wx * (1 - wy) +
                    val_01 * (1 - wx) * wy +
                    val_11 * wx * wy
                )
                
                new_image[new_row * columns + new_col] = interpolated
    
    return new_image


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
        normalized_image = normalize_size_and_center(input_image)
        output = neural_network.classify(normalized_image)
        return jsonify({str(i): round(output[i] * 100, 1) for i in range(10)})

    return app


if __name__ == "__main__":
    application = create_app()
    application.run()
