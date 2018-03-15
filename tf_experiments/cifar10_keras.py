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

from data_sets import load_cifar10
from nn_models import model_convs_FCs
from nn_models import compile_mdl_with_sgd
import utils

## https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/#comment-670

def main(plot):
    start_time = time.time()
    ''' experiment type '''
    expt = 'BoixNet'
    #expt = 'LiaoNet'
    ''' declare variables '''
    batch_size = 256
    num_classes = 10
    nb_epochs = 60
    lr = 0.01
    ''' load cifar '''
    standardize=True
    (x_train, y_train), (x_test, y_test) = load_cifar10(num_classes,standardize=standardize)
    ''' params for cnn'''
    if expt == 'BoixNet':
        nb_conv_layers, nb_fc_layers = 2,3
        nb_conv_filters = [32]*nb_conv_layers
        kernels = [(5,5)]*nb_conv_layers
        nb_units_fcs = [512,256,num_classes]
    elif expt == 'LiaoNet':
        nb_conv_layers, nb_fc_layers = 5,1
        nb_conv_filters = [32]*nb_conv_layers
        kernels = [(5,5)]*nb_conv_layers
        nb_units_fcs = [512,num_classes]
    CHW = x_train.shape[1:] # (3, 32, 32)
    ''' get model '''
    cnn_n = model_convs_FCs(CHW, nb_conv_layers,nb_fc_layers, nb_conv_filters,kernels, nb_units_fcs)
    cnn_n.summary()
    compile_mdl_with_sgd(cnn_n,lr, weight_decay=0,momentum=0,nesterov=False)
    ''' Fit model '''
    cnn = cnn_n.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(x_test,y_test),shuffle=True)
    seconds,minutes,hours = utils.report_times(start_time)
    print(f'\nFinished Training, hours={hours}\a')
    ''' Plots '''
    if plot:
        # Plots for training and testing process: loss and accuracy
        plt.figure(0)
        plt.plot(cnn.history['acc'],'r')
        plt.plot(cnn.history['val_acc'],'g')
        plt.xticks(np.arange(0, nb_epochs+1, 2.0))
        plt.rcParams['figure.figsize'] = (8, 6)
        plt.xlabel("Num of Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy vs Validation Accuracy")
        plt.legend(['train','validation'])


        plt.figure(1)
        plt.plot(cnn.history['loss'],'r')
        plt.plot(cnn.history['val_loss'],'g')
        plt.xticks(np.arange(0, nb_epochs+1, 2.0))
        plt.rcParams['figure.figsize'] = (8, 6)
        plt.xlabel("Num of Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss vs Validation Loss")
        plt.legend(['train','validation'])

if __name__ == '__main__':
    main(plot=True)
    print('\a')


plt.show()
