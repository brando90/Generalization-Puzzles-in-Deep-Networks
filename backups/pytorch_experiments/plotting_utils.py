import numpy as np

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from models_pytorch import trig_kernel_matrix

import pdb

def plot_1D_stuff(arg):
    N_train = arg.X_train.shape[0]
    ##
    x_horizontal = np.linspace(arg.data_lb,arg.data_ub,1000).reshape(1000,1)
    X_plot = arg.poly_feat.fit_transform(x_horizontal)
    X_plot_pytorch = Variable( torch.FloatTensor(X_plot), requires_grad=False)
    ##
    fig1 = plt.figure()
    #plots objs
    p_sgd, = plt.plot(x_horizontal, [ float(f_val) for f_val in arg.mdl_sgd.forward(X_plot_pytorch).data.numpy() ])
    p_pinv, = plt.plot(x_horizontal, np.dot(X_plot,arg.c_pinv))
    p_target, = plt.plot(x_horizontal, arg.f_target(x_horizontal))
    p_data, = plt.plot(arg.X_train,arg.data.Y_train.data.numpy(),'ro')
    ## legend
    nb_terms = arg.c_pinv.shape[0]
    plt.legend(
            [p_sgd,p_pinv,p_data],
            [arg.legend_mdl,f'min norm (pinv) number of monomials={nb_terms}',f'data points = {N_train}']
        )
    ##
    plt.xlabel('x'), plt.ylabel('f(x)')
    plt.title('SGD vs minimum norm solution curves')

def plot_iter_vs_error(arg):
    ## TODO fix put into arguments better
    # start = 1
    # iterations_axis = np.arange(1,nb_iter,step=logging_freq)[start:]
    # ##
    # train_loss_list, test_loss_list, erm_lamdas = np.array(train_loss_list)[start:], np.array(test_loss_list)[start:], np.array(erm_lamdas)[start:]
    # p_train_WP_legend = f'Train error reg_lambda = {reg_lambda}'
    # p_test_WP_legend = f'Test error reg_lambda = {reg_lambda}'
    # p_erm_reg_WP_legend = f'Error+Regularization, reg_lambda_WP = {reg_lambda}'
    # ##
    # fig1 = plt.figure()
    # ##
    # p_train_WP, = plt.plot(iterations_axis, train_loss_list_WP,color='m')
    # p_test_WP, = plt.plot(iterations_axis, test_loss_list_WP,color='r')
    # p_erm_reg_WP, = plt.plot(iterations_axis, erm_lamdas_WP,color='g')
    pass

def plot_iter_vs_all_errors(iterations_axis, train_loss_list,test_loss_list,erm_lamdas, reg_lambda):
    # PLOT ERM+train+test
    fig1 = plt.figure()
    p_train_WP, = plt.plot(iterations_axis, train_loss_list,color='m')
    p_test_WP, = plt.plot(iterations_axis, test_loss_list,color='r')
    p_erm_reg_WP, = plt.plot(iterations_axis, erm_lamdas,color='g')
    plt.xlabel('iterations' )
    plt.ylabel('Error/loss')
    ## legends
    p_train_WP_legend = f'Train error, Weight Parametrization (WP), reg_lambda = {reg_lambda}'
    p_test_WP_legend = f'Test error, Weight Parametrization (WP) reg_lambda = {reg_lambda}'
    p_erm_reg_WP_legend = f'Error+Regularization, Weight Parametrization (WP) reg_lambda = {reg_lambda}'
    ##
    plt.legend([p_erm_reg_WP,p_train_WP,p_test_WP],[p_erm_reg_WP_legend,p_train_WP_legend,p_test_WP_legend])
    plt.title(f'Loss+Regularization,Train,Test vs Iterations, reg_lambda = {reg_lambda}')

def plot_iter_vs_train_test_errors(iterations_axis, train_loss_list,test_loss_list, title_comments,legend_comments,error_type):
    reg_lambda=0
    # PLOT ERM+train+test
    fig1 = plt.figure()
    p_train_WP, = plt.plot(iterations_axis, train_loss_list,color='m')
    p_test_WP, = plt.plot(iterations_axis, test_loss_list,color='r')
    plt.xlabel('iterations' )
    plt.ylabel('Error/loss')
    ## legends
    p_train_WP_legend = f'Train {error_type}, {legend_comments}'
    p_test_WP_legend = f'Test {error_type}, {legend_comments}'
    ##
    plt.legend([p_train_WP,p_test_WP],[p_train_WP_legend,p_test_WP_legend])
    plt.title(f'Train,Test vs Iterations, {title_comments}')

def plot_iter_vs_grads_norm2_4_current_layer(iterations_axis,grads, layer):
    p_grad_legend = f'grad norm(2) for layer = {layer}'
    ##plots
    fig1 = plt.figure()
    p_erm_reg_WP, = plt.plot(iterations_axis, grads,color='g')
    plt.legend([p_erm_reg_WP],[p_grad_legend])
    plt.xlabel('iterations' )
    plt.ylabel('Gradient norm')
    plt.title(f' Iterations vs Gradients')

''' plot for over param scripts '''

def plot_with_params(c_target,X_train,Y_train,lb,ub):
    '''
    Note: training data os govem to visualize the training data set with the model
    '''
    fig = plt.figure()
    deg = c_target.shape[0]-1
    ## plotting data (note this is NOT training data)
    N=5000
    x_plot_points = np.linspace(lb,ub,N).reshape(N,1) # [N,1]
    ## evaluate the model given on plot points
    Kern_plot_points = trig_kernel_matrix(x_plot_points,deg)
    y_plot_points = np.dot(Kern_plot_points,c_target)
    #
    p_mdl, = plt.plot(x_plot_points,y_plot_points)
    p_training_data, = plt.plot(X_train,Y_train,'ro')
    plt.legend([p_mdl,p_training_data], ['function f(x) of degree {}'.format(deg),'data points'])
    ##
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function of degree {}'.format(deg))

def plot_fig4(monomials, train_errors, test_errors, N_train, N_test, target_nb_monomials,alg):
    fig1 = plt.figure()
    p_train, = plt.plot(monomials, train_errors,'-ob')
    p_test, = plt.plot(monomials, test_errors,'-xr')
    p_N_train = plt.axvline(x=N_train,color='g',linestyle='--')
    p_nb_monomials = plt.axvline(x=target_nb_monomials,color='c',linestyle='--')
    plt.legend([p_train,p_test,p_N_train,p_nb_monomials], ['Train error','Test error','# Training data','# of monomials of target function'])
    #plt.ylim(0,100)
    plt.xlabel('Number of monomials' )
    plt.ylabel('Error/loss')
    plt.title(f'Alg {alg}, No-overfitting on sythetic, # of training points = {N_train}, # of test points = {N_test} ')
