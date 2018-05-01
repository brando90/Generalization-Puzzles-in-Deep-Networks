import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.autograd import Variable

import data_classification as data_class

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

def plot_weight_norm_vs_iterations(iterations,w_norms,legend_hyper_params=''):
    fig=plt.figure()
    x_axis=iterations
    w_norm_line, = plt.plot(x_axis,w_norms,label='Weight Norm ||w||')
    handles_4_legend = [w_norm_line]
    ''' set up title and legend '''
    plt.legend(handles=handles_4_legend)
    plt.title(f'Weight norm vs iterations {legend_hyper_params}')

def plot_loss_classification_errors(iterations,stats_collector,legend_hyper_params=''):
    ''' set up x-axis grid as # iterations '''
    x_axis=iterations
    ''' set up figure '''
    fig=plt.figure()
    handles_4_legend = []
    ''' plot loss vs iterations of mdl'''
    #st()
    train_line, = plt.plot(x_axis,stats_collector.train_errors,label=f'Train Error (S)GD model')
    # plt.plot(x_axis,stats_collector.val_losses)
    test_line, = plt.plot(x_axis,stats_collector.test_errors,label='Test Error (S)GD model')
    handles_4_legend.extend([train_line,test_line])
    ''' set up title and legend '''
    plt.legend(handles=handles_4_legend)
    plt.title(f'Errors vs iterations {legend_hyper_params}')

def plot_loss_errors(iterations,stats_collector,test_error_pinv=None,legend_hyper_params='',plot_errors=False):
    '''
        provides a single plot of Train and Test losses vs training iterations
        (with appropriate legends and title)
    '''
    ''' set up x-axis grid as # iterations '''
    x_axis=iterations
    ''' set up figure '''
    fig=plt.figure()
    handles_4_legend = []
    ''' plot loss vs iterations of mdl'''
    train_line, = plt.plot(x_axis,stats_collector.train_losses,label='Train Loss (S)GD model')
    # plt.plot(x_axis,stats_collector.val_losses)
    test_line, = plt.plot(x_axis,stats_collector.test_losses,label='Test Loss (S)GD model')
    handles_4_legend.extend([train_line,test_line])
    if plot_errors:
        train_line, = plt.plot(x_axis,stats_collector.train_errors,label='Train Error (S)GD model')
        # plt.plot(x_axis,stats_collector.val_losses)
        test_line, = plt.plot(x_axis,stats_collector.test_errors,label='Test Error (S)GD model')
        handles_4_legend.extend([train_line,test_line])
    ''' plot losses of pinv model '''
    if test_error_pinv is not None:
        pinv_line, = plt.plot(iterations,test_error_pinv*np.ones(x_axis.shape),label='Test Loss of minimum-norm solution',linestyle='--')
        handles_4_legend.append(pinv_line)
    ''' set up title and legend '''
    plt.legend(handles=handles_4_legend)
    plt.title(f'Loss and Errors vs iterations {legend_hyper_params}')

def plot_loss_and_accuracies(stats_collector):
    '''
        provides a single plot of Train and Test losses vs training iterations
        (with appropriate legends and title)
    '''
    ''' set up x-axis grid as # iterations '''
    x_axis = range(len(stats_collector.train_losses))
    ''' Accuracies '''
    plt.figure(0)
    train_line, = plt.plot(x_axis,stats_collector.train_accs,label='Train')
    test_line, = plt.plot(x_axis,stats_collector.test_accs,label='Validation')
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend([train_line,test_line])
    ''' Losses '''
    plt.figure(1)
    train_line, = plt.plot(x_axis,stats_collector.train_losses,label='Train')
    test_line, = plt.plot(x_axis,stats_collector.test_losses,label='Validation')
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend([train_line,test_line])

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

def visualize_reconstruction_on_data_set(X,Y,dataset_name,f_mdl,f_target=None,f_pinv=None):
    # TODO visualize only the on data set
    ''' set up x-axis grid from data set '''
    x_axis = X.data.numpy()
    ''' plot reconstructions'''
    plt.figure()
    mdl_line, = plt.plot(x_axis,Y_mdl,label='Model Reconstruction')
    handles_4_legend = [mdl_line,real_line]
    ''' plot target function'''
    if f_target is not None:
        target_line, = plt.plot(x_axis,f_target(x_axis),label='True Target Function')
        handles_4_legend.append(target_line)
    ''' plot minimum norm soln'''
    if f_pinv is not None:
        pinv_line, = plt.plot(x_axis,f_pinv(x_axis.reshape(N_denseness,1)),label='Minimum norm solution')
        handles_4_legend.append(pinv_line)
    ''' plot data set points X and Y also'''
    if X is not None and Y is not None and legend_data_set is not None:
        N,_ = X.shape
        data_pts = plt.scatter(X.reshape((N,)),Y,label=legend_data_set,c='r')
        handles_4_legend.append(data_pts)
    ''' add auxiliarly info '''
    plt.legend(handles=handles_4_legend)
    plt.title('Reconstructions')

def visualize_1D_reconstruction(lb,ub,N_denseness, f_mdl,f_target=None,f_pinv=None,X=None,Y=None,legend_data_set=None):
    '''
        Visualize 1D reconstruction of mdl. Function assumes f's already return numpy
        and are able to process as input numpy data.

        Optional arguments for plotting the data set points too
    '''
    ''' create x-axis grid and put it in right format'''
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
    ''' plot data set points X and Y also'''
    if X is not None and Y is not None and legend_data_set is not None:
        N,_ = X.shape
        data_pts = plt.scatter(X.reshape((N,)),Y,label=legend_data_set,c='r')
        handles_4_legend.append(data_pts)
    ''' add auxiliarly info '''
    plt.legend(handles=handles_4_legend)
    plt.title('Reconstructions')

#### Classification plots

def visualize_classification_data_learned_planes_2D(lb,ub,N_denseness,Xtr,Ytr,f_mdl,f_target=None):
    ''' '''
    X_pos,X_neg = data_class.separte_data_by_classes(Xtr,Ytr)
    fig = plt.figure()
    ''' plot data points '''
    plt.scatter(X_pos[:,0],X_pos[:,1],marker='+')
    plt.scatter(X_neg[:,0],X_neg[:,1],marker='o')
    ''' plot hyper planes '''
    x_axis = np.linspace(lb,ub,N_denseness)
    plt.plot(x_axis,f_target(x_axis))
