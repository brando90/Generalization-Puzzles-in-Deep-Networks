#!/usr/bin/env python
#SBATCH --mem=20000
#SBATCH --time=1-00:30
#SBATCH --mail-type=END
#SBATCH --mail-user=brando90@mit.edu
#SBATCH --array=1-5
#SBATCH --gres=gpu:1

import sys
import os

current_directory = os.getcwd() #The method getcwd() returns current working directory of a process.
sys.path.append(current_directory)

import time
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.models import load_model

import os

if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

from data_sets import load_cifar10
from nn_models import model_convs_FCs
from nn_models import compile_mdl_with_sgd
import utils

import pickle

import argparse

from pdb import set_trace as st

## inspired from: https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/#comment-670

sj, satid = 0, 0
''' Params '''
parser = argparse.ArgumentParser(description='Keras Example')
''' setup params '''
parser.add_argument("-satid", "--satid", type=int, default=0,
                    help="Slurm Array Task ID, for naming the file")

if 'SLURM_ARRAY_TASK_ID' in os.environ and 'SLURM_JOBID' in os.environ:
    satid = int(os.environ['SLURM_ARRAY_TASK_ID'])
    sj = int(os.environ['SLURM_JOBID'])

def main(plot):
    start_time = time.time()
    ''' Directory names '''
    path = '../pytorch_experiments/test_runs_flatness/keras_expt_April_19'
    filename = f'chance_plateau_{satid}'
    utils.make_and_check_dir(path)
    ''' experiment type '''
    #expt = 'BoixNet'
    #expt = 'LiaoNet'
    expt = 'GBoixNet'
    #expt = 'debug'
    ''' declare variables '''
    batch_size = 2**10
    num_classes = 10
    nb_epochs = 300
    lr = 0.1
    ''' load cifar '''
    standardize=True
    (x_train, y_train), (x_test, y_test) = load_cifar10(num_classes,standardize=standardize)
    print(x_train[0,:])
    ''' params for cnn'''
    print(f'expt = {expt}')
    if expt == 'BoixNet':
        nb_conv_layers = 2
        nb_conv_filters = [32]*nb_conv_layers
        kernels = [(5,5)]*nb_conv_layers
        nb_fc_layers = 3
        nb_units_fcs = [512,256,num_classes]
    elif expt == 'LiaoNet':
        nb_conv_layers, nb_fc_layers = 3,1
        nb_conv_filters = [32]*nb_conv_layers
        kernels = [(5,5)]*nb_conv_layers
        nb_units_fcs = [num_classes]
    elif expt == 'GBoixNet':
        cnn_filename = f'keras_net_{satid}'
        nb_conv_layers, nb_fc_layers = 1,2
        nb_conv_filters = [22]*nb_conv_layers
        kernels = [(5,5)]*nb_conv_layers
        nb_units_fcs = [30,num_classes]
    elif expt == 'debug':
        nb_conv_layers, nb_fc_layers = 1,1
        nb_conv_filters = [2]*nb_conv_layers
        kernels = [(10,10)]*nb_conv_layers
        nb_units_fcs = [2,num_classes]
    CHW = x_train.shape[1:] # (3, 32, 32)
    ''' get model '''
    cnn_n = model_convs_FCs(CHW, nb_conv_layers,nb_fc_layers, nb_conv_filters,kernels, nb_units_fcs)
    cnn_n.summary()
    compile_mdl_with_sgd(cnn_n,lr, weight_decay=0,momentum=0,nesterov=False)
    ''' Fit model '''
    cnn = cnn_n.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(x_test,y_test),shuffle=True)
    seconds,minutes,hours = utils.report_times(start_time)
    print(f'\nFinished Training, hours={hours}\a')
    ''' save history and mdl '''
    path_2_save = os.path.join(path,filename)
    print(f'path_2_save = {path_2_save}')
    print(f'does dir exist? {os.path.isdir(path)}')
    # save history
    with open(path_2_save, 'wb+') as file_pi:
        history = dict({'batch_size':batch_size,'nb_epochs':nb_epochs,'lr':lr,'expt':expt},**cnn.history)
        pickle.dump(history, file_pi)
    # save model
    cnn_n.save( os.path.join(path,cnn_filename) )
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

        plt.show()

if __name__ == '__main__':
    main(plot=False)
    print('\a')
