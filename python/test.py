from main import *

import os
import cv2
import numpy as np

ROWS = 64
COLS = 64
CHANNELS = 3

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def prepare_data(images):
    m = len(images)
    X = np.zeros((m, ROWS, COLS, CHANNELS), dtype=np.uint8)
    y = np.zeros((1, m))
    
    for i, image_file in enumerate(images):
        X[i,:] = read_image(image_file)
        if 'dog' in image_file.lower():
            y[0, i] = 1
        elif 'cat' in image_file.lower():
            y[0, i] = 0
            
    return X, y

TRAIN_DIR = 'dataset/Train_data/'
TEST_DIR = 'dataset/Test_data/'

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

train_set_x, train_set_y = prepare_data(train_images)
test_set_x, test_set_y = prepare_data(test_images)

train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], ROWS*COLS*CHANNELS).T
test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

layers_dims = [12288, 800, 10, 1] #  4-layer model
parameters = L_layer_model(train_set_x, train_set_y, layers_dims, learning_rate = 0.1, num_iterations = 2, print_cost = True)

print("train accuracy: {} %".format(100 - np.mean(np.abs(predict(train_set_x, parameters) - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(predict(test_set_x, parameters) - test_set_y)) * 100))

