import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import json
import requests
from tqdm import tqdm

# print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('\nTrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
print('Test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

train_images_gr = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images_gr = test_images.reshape(test_images.shape[0], 28, 28, 1)

print('\nTrain_images.shape: {}, of {}'.format(train_images_gr.shape, train_images_gr.dtype))
print('Test_images.shape: {}, of {}'.format(test_images_gr.shape, test_images_gr.dtype))

INPUT_SHAPE = (28, 28 ,1)

def create_cnn_architecture_model1(input_shape):
    inp = keras.layers.Input(shape=input_shape)

    conv1 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),
                                activation='relu', padding='same')(inp)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                activation='relu', padding='same')(pool1)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    flat = keras.layers.Flatten()(pool2)

    hidden1 = keras.layers.Dense(256, activation='relu')(flat)
    drop1 = keras.layers.Dropout(rate=0.3)(hidden1)

    out = keras.layers.Dense(10, activation='softmax')(drop1)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_cnn_architecture_model1(input_shape=INPUT_SHAPE)
model.summary()