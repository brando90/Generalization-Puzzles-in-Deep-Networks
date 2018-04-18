import matplotlib.pyplot as plt

import pickle
import os

import numpy as np

from pdb import set_trace as st

def plot_keras(history):
    '''
    :param history: the dictionary history saved by keras.
    :return:
    '''
    nb_epochs = len(history['acc'],)
    # Plots for training and testing process: loss and accuracy
    plt.figure(0)
    plt.plot(history['acc'], 'r')
    plt.plot(history['val_acc'], 'g')
    plt.xticks(np.arange(0, nb_epochs + 1, 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train', 'validation'])

    plt.figure(1)
    plt.plot(history['loss'], 'r')
    plt.plot(history['val_loss'], 'g')
    plt.xticks(np.arange(0, nb_epochs + 1, 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train', 'validation'])

    plt.show()

def main():
    path = '../pytorch_experiments/test_runs_flatness/keras_expt'
    filename = 'chance_plateau_debug_0'
    ''' load history '''
    path_2_file = os.path.join(path,filename)
    with open(path_2_file, 'rb') as keras_hist_file:
        hist_dict = pickle.load(keras_hist_file)
    ''' plot history '''
    plot_keras(hist_dict)

if __name__ == '__main__':
    main()
    print('Done')

