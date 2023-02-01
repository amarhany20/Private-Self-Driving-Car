from flask import Flask
import socketio
import eventlet
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D,Dropout,Flatten,Dense
import cv2
import pandas as pd
import random
import os
import h5py

import ntpath
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

from keras.models import model_from_json

import subprocess

import time


sio = socketio.Server()
app = Flask(__name__)

speed_limit = 35
counter = 0
start = time.time()

#My Parameters





def img_preprocess(img):
    img = img[60:135, :, :]  # trimming
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Because I will use Nvidia Model (The say it is better with yuv)
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Smoothing
    img = cv2.resize(img, (200, 66))  # this size is used by nvidia model arch
    img = img / 255

    return img
@sio.on('telemetry')
def telemtery(sid, data):

    # start = time.time()
    # Debug: print("Image: " ,data['image'])
    # Debug: print("Type: ", type(data['image']))
    # Debug: print("Time Elabsed in this function : ",(time.time() - start)*1000)
    # plt.imshow(image[0])
    # plt.show(block=False)
    # plt.close()
    # image = Image.open(BytesIO(base64.b64decode(data['image'])))
    # image = np.asarray(image)
    # image = img_preprocess(image)
    # image = np.array([image])
    # Debug : Count()
    # send_control(0,1)
    # steering_angle = (float(model.predict(image)) + float(model2.predict(image)))/2
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image,verbose = 0))
    throttle = 1.0 - speed / speed_limit
    # print('{} {} {}'.format(steering_angle, throttle - abs(steering_angle), speed))
    send_control(steering_angle,throttle - abs(steering_angle*1.5))


@sio.on('connect')
def connect(sid,myenviroment):
    print('Connected')
    # send_control(0,0)

def send_control(steering_angle,throttle):
    sio.emit('steer',data={
        'steering_angle':steering_angle.__str__(),
        'throttle': throttle.__str__()
    })
# def Count():
#     global counter
#     global start
#     print(counter)
#     counter = counter + 1
#     end = time.time()
#     print(end - start , "MS")


# def connect()
#     print('connected')
if __name__ == '__main__':
    # model = load_model('Model 2 A1A2 (Amazing in 1 and 2 but possible accident on right turn 10 Epochs).h5')
    # model = load_model("Model 5 Gucci for 2.h5")
    model = load_model("Model 4 A1A2 acceptable (will be used in case there is nothing better).h5")
    app = socketio.Middleware(sio,app)
    eventlet.wsgi.server(eventlet.listen(("",4567)),app)
    counter = 0