import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

import pdb
from pdb import set_trace as st

def plot_loss_errors(iterations,stats_collector):
    fig=plt.figure()
    train_line, = plt.plot(iterations,stats_collector.train_losses,label='Train Loss')
    # plt.plot(iterations,stats_collector.val_losses)
    test_line, = plt.plot(iterations,stats_collector.test_losses,label='Test Loss')
    plt.legend(handles=[train_line,test_line])
    plt.title('Loss and Errors vs iterations')

####

def visualize_reconstruction(mdl,X,Y,dataset_name):
    X = Variable(torch.FloatTensor(X),requires_grad=False)
    Y_mdl = mdl(X).data.numpy()
    x_axis = X.data.numpy()
    Y_real = Y
    ##
    plt.figure()
    mdl_line, = plt.plot(x_axis,Y_mdl,label='Mdl Reconstruction')
    real_line, = plt.plot(x_axis,Y_real,label='Real Reconstruction')
    plt.legend(handles=[mdl_line,real_line])
    plt.title('Reconstructions')
