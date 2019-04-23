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
print X_train.shape # ==> (60000, 28, 28) there are 60000 samples in the data set and each image is 28 by 28 pixels
#confirm results of shape by using matplot
from matplotlib import pyplot as plt
plt.imshow(X_train[0])
#declaring a dimension for the depth of the input image (in this case the depth is 1)
#####pre-processing##########
#to reshape input data ie. transforming data from having shape (n, width, height) to (n, depth, width, height)
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
print X_train.shape
#convert data type to float32 and normalize data values to range [0, 1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
