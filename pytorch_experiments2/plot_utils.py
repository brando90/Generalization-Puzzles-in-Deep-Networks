import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.autograd import Variable

import pdb
from pdb import set_trace as st

def print_gd_vs_pinv_params(mdl,c_pinv):
    '''
        Print to console the parameters of the mdl vs the vector c_pinv given
    '''
    d1,d2 = c_pinv.size()
    c_sgd = mdl.C.weight.data
    ##
    c_sgd = c_sgd.view(d1,d2)
    c_pinv = c_pinv.view(d1,d2)
    print(f'c_sgd={c_sgd}')
    print(f'c_pinv={c_pinv}')

##

def plot_loss_errors(iterations,stats_collector):
    '''
        provides a single plot of Train and Test losses vs training iterations
        (with appropriate legends and title)
    '''
    fig=plt.figure()
    train_line, = plt.plot(iterations,stats_collector.train_losses,label='Train Loss')
    # plt.plot(iterations,stats_collector.val_losses)
    test_line, = plt.plot(iterations,stats_collector.test_losses,label='Test Loss')
    plt.legend(handles=[train_line,test_line])
    plt.title('Loss and Errors vs iterations')

def plot_sgd_vs_pinv_distance_during_training(iterations,stats_collector):
    '''
        plots evolution of ||c_mdl - c_pinv||^2 as training progresses.
    '''
    fig=plt.figure()
    diff_GD_vs_PINV = stats_collector.dynamic_stats_storer['diff_GD_vs_PINV']
    sgd_vs_pinv, = plt.plot(iterations,diff_GD_vs_PINV,label='distance between (S)GD solution and minimum-norm solution')
    plt.legend(handles=[sgd_vs_pinv])
    plt.title('Distance between (S)GD solution and minimum-norm solution')

####

def visualize_reconstruction_on_data_set(mdl,X,Y,dataset_name,f_target=None,f_pinv=None):
    # TODO visualize only the on data set
    X = Variable(torch.FloatTensor(X),requires_grad=False)
    x_axis = X.data.numpy()
    Y_mdl = mdl(X).data.numpy()
    Y_real = Y
    ##
    x_axis = np.linspace(-1,1,1000)
    st()
    X = Variable(torch.FloatTensor(X),requires_grad=False)
    Y_mdl = mdl(X).data.numpy()
    ##
    plt.figure()
    mdl_line, = plt.plot(x_axis,Y_mdl,label='Model Reconstruction')
    #real_line, = plt.plot(x_axis,Y_real,label='Real Reconstruction')
    handles_4_legend = [mdl_line,real_line]
    if f_target is not None:
        #x_axis = np.linspace(-1,1,1000)
        true_line, = plt.plot(x_axis,f_target(x_axis),label='True Target Function')
        handles_4_legend.append(true_line)
    if f_target is not None:
        #x_axis = np.linspace(-1,1,1000)
        true_line, = plt.plot(x_axis,f_target(x_axis),label='Minimum norm solution')
        handles_4_legend.append(true_line)
    ''' add auxiliarly info '''
    plt.legend(handles=handles_4_legend)
    plt.title('Reconstructions')

def visualize_1D_reconstruction(lb,ub,N_denseness, f_mdl,f_target=None,f_pinv=None,X=None,Y=None,legend_data_set=None):
    '''
        Visualize 1D reconstruction of mdl. Function assumes f's already return numpy
        and are able to process as input numpy data.

        Optional arguments for plotting the data set points too
    '''
    ''' create grid and put it in right format'''
    x_axis = np.linspace(lb,ub,N_denseness)
    ''' plot reconstruction'''
    plt.figure()
    mdl_line, = plt.plot(x_axis,f_mdl(x_axis.reshape(N_denseness,1)),label='Model Reconstruction')
    handles_4_legend = [mdl_line]
    ''' plot target function'''
    if f_target is not None:
        target_line, = plt.plot(x_axis,f_target(x_axis),label='True Target Function')
        handles_4_legend.append(target_line)
    ''' plot minimum norm soln'''
    if f_pinv is not None:
        pinv_line, = plt.plot(x_axis,f_pinv(x_axis.reshape(N_denseness,1)),label='Minimum norm solution')
        handles_4_legend.append(pinv_line)
    # ''' plot data set points X and Y also'''
    # if X is not None and Y is not None and legend_data_set is not None:
    #     N,_ = X.shape
    #     data_pts = plt.scatter(X.reshape((N,)),Y,label=legend_data_set,c='r')
    #     handles_4_legend.append(data_pts)
    ''' add auxiliarly info '''
    plt.legend(handles=handles_4_legend)
    plt.title('Reconstructions')
