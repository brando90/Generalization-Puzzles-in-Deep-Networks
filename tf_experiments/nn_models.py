# Import all modules
import time
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
#from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

# Import Tensorflow with multiprocessing
import tensorflow as tf
import multiprocessing as mp

# Loading the CIFAR-10 datasets
from keras.datasets import cifar10

def base_model_4_layered_CNN():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # Train model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def base_model_6_layered_CNN():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
    model.add(Dropout(0.2))

    model.add(Conv2D(32,(3,3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024,activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # Train model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def model_convs_FCs(CHW, nb_conv_layers,nb_fc_layers, nb_conv_filters,kernels, nb_units_fcs):
    model = Sequential()
    if nb_conv_layers < 1:
        # first layer is conv
        raise ValueError(f'nb_conv_layers: {nb_conv_layers} is too low need at least 1')
    if nb_fc_layers < 1:
        # last layer is 10 neurons with softmax activation for classification
        raise ValueError(f'nb_fc_layers: {nb_fc_layers} is too low need at least 1')
    ''' make convolution layers '''
    model.add(Conv2D(nb_conv_filters[0], kernels[0], padding='same', input_shape=CHW))
    model.add(Activation('relu'))
    for i in range(1,nb_conv_layers):
        model.add(Conv2D(nb_conv_filters[i],kernels[i]))
        model.add(Activation('relu'))
    ''' make fully connected layers '''
    model.add(Flatten())
    for i in range(0,nb_fc_layers-1):
        model.add(Dense(nb_units_fcs[i]))
        model.add(Activation('relu'))
    ##
    num_classes = nb_units_fcs[-1]
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

def compile_mdl_with_sgd(model, lr, weight_decay=0,momentum=0,nesterov=False):
    ''' optimizer '''
    sgd = SGD(lr = lr, decay=weight_decay, momentum=momentum, nesterov=nesterov)
    # Configures the model for training.
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
