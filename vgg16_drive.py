import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import socket

class Smooth:
    def __init__(self, windowsize=50):
        self.window_size = windowsize
        self.data = np.zeros(self.window_size, dtype=np.float32)
        self.index = 0

    def __iadd__(self, x):
        self.data[self.index % self.window_size] = x
        self.index += 1
        return self

    def __str__(self):
        return str(self.data[:self.index].mean())

    def __float__(self):
        return float(self.data[:self.index].mean())

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
image_size = (128, 128)
address = ("0.0.0.0", 20160)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
smooth = Smooth(windowsize=20)

@sio.on('telemetry')
def telemetry(sid, data):
    global smooth
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    try:
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image = image.resize(image_size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        # transformed_image_array = image_array[None, :, :, :]
        transformed_image_array = image_array.reshape((1, image_size[1], image_size[0], 3))
        # This model currently assumes that the features of the model are just the images. Feel free to change this.
        steering_angle = float(model.predict(transformed_image_array, batch_size=1))/100
        # if random.random() < 0.1:
        #     steering_angle += (random.random() - 0.5)
        # The driving model currently just outputs a constant throttle. Feel free to edit this.
        throttle = 1
        smooth += steering_angle
        s.sendto(str(smooth), address)
        print("%.3f" % steering_angle, throttle)
    except Exception as e:
        print(e)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)