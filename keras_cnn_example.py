import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras as ks
import numpy as np
np.random.seed(123) #setting a seed for the computers pseudo random generator which will reproduce results from our script
#import Sequential model type from Keras. This is a linear stack of neural network layers for feed-forward CNN
from keras.models import Sequential
#import core layers from Keras. These layers are used in any neural network
from keras.layers import Dense, Dropout, Activation, Flatten
#import CNN layers from Keras - convolutional layers to help us train on image data
from keras.layers import Convolution2D, MaxPooling2D
#import utilities to transform data later
from keras.utils import np_utils
#Loading image data from MNIST
from keras.datasets import mnist
#load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#this is the shape of the dataset
print X_train.shape
