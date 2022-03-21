"""
Test REST service that classifies an image as a numerical digit.

Copyright (c) 2022 by Jenson Wong
"""
from base64 import b64encode
from io import BytesIO
from os.path import abspath, dirname

from PIL import Image

from app import create_app

CURRENT_DIRECTORY = dirname(abspath(__file__))
NUMBER_2_FILE = f"{CURRENT_DIRECTORY}/2.png"


def base64_encoded_image_of_number_2():
    image = Image.open(NUMBER_2_FILE)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = bytes("data:image/jpeg;base64,", encoding="utf-8") + b64encode(buffer.getvalue())
    return image_base64.decode("utf-8")


def test_app_recognises_image_of_number_2():
    client = create_app().test_client()
    response = client.post("/recognise-number", json={"image": base64_encoded_image_of_number_2()})
    assert response.json == {
        "0": 0.0,
        "1": 0.0,
        "2": 100.0,
        "3": 0.0,
        "4": 0.0,
        "5": 0.0,
        "6": 0.0,
        "7": 0.0,
        "8": 0.0,
        "9": 0.0,
    }
