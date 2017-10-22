import time
import numpy as np
from numpy.linalg import matrix_rank
import sys

import ast

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from maps import NamedDict as Maps
import pdb

from models_pytorch import *
from inits import *
from sympy_poly import *
from poly_checks_on_deep_net_coeffs import *
from data_file import *

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.integrate as integrate
import scipy
import scipy.io

import argparse

SLURM_JOBID = 6

# def get_argument_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-expt_type','--experiment_type',type=str, help='save the result of the experiment')
#     parser.add_argument('-lb','--lower_bound',type=int, help='lower bound')
#     parser.add_argument('-num','--number_values',type=int, help='number of values in between lb and ub')
#     parser.add_argument('-num_rep','--number_repetitions',type=int, help='number of repetitions per run')
#     parser.add_argument('-save','--save_bulk_experiment',type=bool, help='save the result of the experiment')
#     parser.add_argument('-sj', '--SLURM_JOBID', help='SLURM_JOBID for run')
#     parser.add_argument('-rt_wp', '--reg_type_wp', type=str, default='tikhonov', help='Regularization Type for WP. e.g: VM, tikhonov, V[^2W, etc')
#     cmd_args = parser.parse_args()
#     return cmd_args
#
# cmd_args = get_argument_parser()
# SLURM_JOBID = cmd_args.SLURM_JOBID

##

def get_target_Y_SP_poly(X_train,X_test,Degree_data_set,c_mdl,noise_train=0,noise_test=0):
    ## get data points
    poly_feat = PolynomialFeatures(degree=Degree_data_set)
    ## create poly features
    Kern_train = poly_feat.fit_transform(X_train)
    Kern_test = poly_feat.fit_transform(X_test)
    ## evaluate target function
    Y_train = np.dot(Kern_train,c_mdl)
    Y_test = np.dot(Kern_test,c_mdl)
    ## add noise to target
    Y_train, Y_test = Y_train+noise_train, Y_test+noise_test
    return Y_train, Y_test

def get_c_fit_function(target_f,D0,degree_mdl,N,lb,ub):
    ## evaluate target_f on x_points
    X = np.linspace(lb,ub,N).reshape(N,D0) # [N,D0]
    Y = target_f(X) #
    ## copy that f with the target degree polynomial
    poly_feat = PolynomialFeatures(degree=degree_mdl)
    Kern = poly_feat.fit_transform(X)
    #c_mdl = np.dot(np.linalg.pinv( Kern ), Y)
    c_mdl = np.polyfit(X.reshape((N,)),Y.reshape((N,)),degree_mdl)[::-1]
    return c_mdl

def get_c_fit_data(X,Y,degree_mdl):
    ## copy that f with the target degree polynomial
    poly_feat = PolynomialFeatures(degree=degree_mdl)
    Kern = poly_feat.fit_transform(X)
    c_mdl = np.dot(np.linalg.pinv( Kern ), Y)
    return c_mdl

def plot_target_function(c_mdl,X_train,Y_train,lb,ub,f_2_imitate):
    '''
    Note: training data os govem to visualize the training data set with the model
    '''
    deg = c_mdl.shape[0]-1
    ## plotting data (note this is NOT training data)
    N=5000
    x_plot_points = np.linspace(lb,ub,N).reshape(N,1) # [N,1]
    ## evaluate the model given on plot points
    poly_feat = PolynomialFeatures(degree=deg)
    Kern_plot_points = poly_feat.fit_transform(x_plot_points)
    y_plot_points = np.dot(Kern_plot_points,c_mdl)
    #
    x_for_f = np.linspace(lb,ub,30000)
    #pdb.set_trace()
    y_for_f = f_2_imitate( x_for_f )
    #
    p_mdl, = plt.plot(x_plot_points,y_plot_points)
    p_f_2_imitate, = plt.plot(x_for_f,y_for_f)
    p_training_data, = plt.plot(X_train,Y_train,'ro')
    plt.legend([p_mdl,p_f_2_imitate,p_training_data], ['Target function f(x) of degree {}'.format(deg),'f trying to imitate','data points'])
    ##
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Target Function of degree {}'.format(deg))

def plot_poly_with_params(c_mdl,X_train,Y_train,lb,ub):
    '''
    Note: training data os govem to visualize the training data set with the model
    '''
    fig = plt.figure()
    deg = c_mdl.shape[0]-1
    ## plotting data (note this is NOT training data)
    N=5000
    x_plot_points = np.linspace(lb,ub,N).reshape(N,1) # [N,1]
    ## evaluate the model given on plot points
    poly_feat = PolynomialFeatures(degree=deg)
    Kern_plot_points = poly_feat.fit_transform(x_plot_points)
    y_plot_points = np.dot(Kern_plot_points,c_mdl)
    #
    p_mdl, = plt.plot(x_plot_points,y_plot_points)
    p_training_data, = plt.plot(X_train,Y_train,'ro')
    plt.legend([p_mdl,p_training_data], ['function f(x) of degree {}'.format(deg),'data points'])
    ##
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function of degree {}'.format(deg))

def plot_fig4(monomials, train_errors, test_errors, N_train, N_test):
    fig1 = plt.figure()
    p_train, = plt.plot(monomials, train_errors,'-ob')
    p_test, = plt.plot(monomials, test_errors,'-xr')
    p_v = plt.axvline(x=N_train,color='g',linestyle='--')
    plt.legend([p_train,p_test,p_v], ['Train error','Test error','# Training data'])
    #plt.ylim(0,100)
    plt.xlabel('Number of monomials' )
    plt.ylabel('Error/loss')
    plt.title('No-overfitting on sythetic, # of training points = {}, # of test points = {} '.format(N_train,N_test))

def get_LA_error(X,Y,c,poly_feat):
    N = X.shape[0]
    return (1/N)*(np.linalg.norm(Y-np.dot( poly_feat.fit_transform(X),c) )**2)

def get_nb_monomials(nb_variables,degree):
    return int(scipy.misc.comb(nb_variables+degree,degree))

def my_pinv(X):
    XXt_inv = np.linalg.inv(np.dot(X,X.T))
    X_my_pinv = np.dot(X.T ,XXt_inv )
    return X_my_pinv

def get_errors_pinv_mdls(X_train,Y_train,X_test,Y_test,degrees):
    train_errors, test_errors = [], []
    ranks = []
    s_inv_total, s_inv_max = [], []
    ##
    N_train = X_train.shape[0]
    ##
    for degree_mdl in degrees:
        poly_feat = PolynomialFeatures(degree=degree_mdl)
        # get mdl
        Kern_train = poly_feat.fit_transform(X_train)
        Kern_train_pinv = np.linalg.pinv( Kern_train )
        #Kern_train_pinv = my_pinv(Kern_train)
        c_pinv = np.dot(Kern_train_pinv, Y_train) # c = <K^+,Y>
        c_pinv = np.polyfit(X_train.reshape((N_train,)),Y_train.reshape((N_train,)),degree_mdl)[::-1]
        #pdb.set_trace()
        #c_pinv = np.polyfit(X_train.reshape((N_train,)),Y_train.reshape((N_train,)),deg=degree_mdl)
        ##
        U,S,V = np.linalg.svd(Kern_train_pinv)
        s_inv_total.append( np.sum(S) )
        s_inv_max.append(max(S))
        #pdb.set_trace()
        ##
        rank_kern_train = matrix_rank(Kern_train)
        ranks.append(rank_kern_train)
        # evluate it on train and test
        train_error = get_LA_error(X_train,Y_train,c_pinv,poly_feat)
        test_error = get_LA_error(X_test,Y_test,c_pinv,poly_feat)
        #
        train_errors.append( train_error )
        test_errors.append( test_error )
    return train_errors,test_errors,ranks,s_inv_total,s_inv_max

def get_c(nb_monomials_data):
    #pdb.set_trace()
    #c_mdl = 1*np.arange(0,nb_monomials_data).reshape(nb_monomials_data,1)+np.random.normal(loc=2.0,scale=1.0,size=(nb_monomials_data,1))
    c_mdl = np.array([ 2**d for d in np.arange(0,nb_monomials_data) ]).reshape(nb_monomials_data,1)
    #c_mdl = np.array([ 1.5**d for d in np.arange(0,nb_monomials_data) ]).reshape(nb_monomials_data,1)+np.random.normal(loc=3.0,scale=1.0,size=(nb_monomials_data,1))
    return c_mdl

def my_main(**kwargs):
    ##
    lb,ub = -2,2
    start_time = time.time()
    plotting = kwargs['plotting'] if 'plotting' in kwargs else False
    freq = -1
    ## get target Y
    if 'file_name' in kwargs:
        mat_dict = scipy.io.loadmat(file_name=kwargs['file_name'])
        X_train, X_test = mat_dict['X_train'], mat_dict['X_test']
        Y_train, Y_test = mat_dict['Y_train'], mat_dict['Y_test']
        c_mdl = mat_dict['c_mdl']
        D0 = X_train.shape[1]
        N_train, N_test = X_train.shape[0], X_test.shape[0]
    else:
        ## get X input points
        D0 = 1
        N_train, N_test = 150, 1000
        print('D0 = {}, N_train = {}, N_test = {}'.format(D0,N_train,N_test))
        #X_train, X_test = 2*np.random.rand(N_train,D0)-1, 2*np.random.rand(N_test,D0)-1
        #X_train, X_test = 2*np.random.rand(N_train,D0)-1, 2*np.random.rand(N_test,D0)-1
        X_train, X_test = np.linspace(lb,ub,N_train).reshape(N_train,D0), np.linspace(lb,ub,N_test).reshape(N_test,D0)
        #X_train = np.concatenate( (X_train, X_test) ,axis=0)
        ## get target function
        Degree_data_set = 80
        nb_monomials_data = get_nb_monomials(nb_variables=D0,degree=Degree_data_set)
        #c_mdl = np.arange(1,nb_monomials_data+1).reshape((nb_monomials_data,1))+np.random.normal(loc=3.0,scale=1.0,size=(nb_monomials_data,1))
        #c_mdl = get_c(nb_monomials_data) # [D,1]
        #c_mdl = get_c_fit_function(generate_h_add_1d,D0,Degree_data_set,N=3*N_test,lb=-1,ub=1)
        #c_mdl = get_c_fit_function(generate_h_gabor_1d,D0,Degree_data_set,N=3*N_test,lb=-1,ub=1)
        freq_sin = 15
        freq_cos = 2
        freq = max(freq_sin, freq_cos)
        f_2_imitate = lambda x: np.cos(freq_cos*2*np.pi*x)
        c_mdl = get_c_fit_function(f_2_imitate, D0,Degree_data_set, N=20000, lb=lb,ub=ub) # [Deg,1] sin with period k
        #c_mdl = get_c_fit_function(lambda x: np.exp( -(x**2) )*np.cos(4*np.pi*(x)),  D0,Degree_data_set, N=3*N_test, lb=lb,ub=ub)
        #c_mdl = get_c_fit_function(lambda x: np.exp( -(x**2) )*( np.cos(freq_sin*np.pi*(x)) + np.sin(freq_cos*np.pi*(x)) ),  D0,Degree_data_set, N=30*N_test, lb=lb,ub=ub)
        ##
        def f_data(x):
            poly_feat = PolynomialFeatures(degree=Degree_data_set)
            Kern = poly_feat.fit_transform(x)
            return  np.dot(Kern,c_mdl)
        print('c_mdl = ',c_mdl)
        print('c_mdl.shape = ',c_mdl.shape)
        print('nb_monomials_data = {} \n'.format(nb_monomials_data) )
        ## get noise for target Y
        mu_noise, std_noise = 0, 0.0
        #noise_train, noise_test = np.random.normal(loc=mu_noise,scale=std_noise,size=(N_train,D0)), np.random.normal(loc=mu_noise,scale=std_noise,size=(N_test,D0))
        noise_train, noise_test = 0, 0
        ## get target Y
        Y_train, Y_test = get_target_Y_SP_poly(X_train,X_test, Degree_data_set,c_mdl, noise_train=noise_train,noise_test=noise_test)
    ## get errors from models
    step_deg=2
    smallest_deg,largest_deg = 1,200
    degrees = list(range(smallest_deg,largest_deg,step_deg))
    train_errors,test_errors,ranks,s_inv_total,s_inv_max = get_errors_pinv_mdls(X_train,Y_train,X_test,Y_test,degrees)
    ##
    print('train_errors = ', train_errors)
    print('test_errors = ', test_errors)
    ## plot them
    monomials = [ get_nb_monomials(nb_variables=D0,degree=d) for d in degrees ]
    ## REPORT TIMES
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    print("\a--- {} seconds --- \n --- {} minutes --- \n --- {} hours ---".format(seconds, minutes, hours) )
    print('f_s > 2F_max = N_train > 4 freq = {} > 4*{} =  {} > {} ?, Is it true: {}'.format(N_train,freq, N_train,4*freq, N_train>4*freq))
    print('number of zeros sine = {}'.format( int( 2*2/(1/freq_sin) )   ))
    print('number of zeros cos = {}'.format( int( 2*2/(1/freq_cos) )    ))
    print('total number of zeros = {}'.format(   int( 2*2/(1/freq_cos) ) + int( 2*2/(1/freq_sin) )   ))
    ##
    poly_feat = PolynomialFeatures(degree=Degree_data_set)
    Kern_mdl_truth = poly_feat.fit_transform(X_train)
    print('Degree_data_set = {}'.format(Degree_data_set))
    print('rank(Kern_mdl_truth) = {}'.format( matrix_rank(Kern_mdl_truth) ))
    print('N_train = {} '.format(N_train))
    print('Kern_mdl_truth.shape = {}'.format(Kern_mdl_truth.shape))
    c_mdl_deg_truth = get_c_fit_data(X_train,Y_train,Degree_data_set) # model with same degree as truth but trained on training set
    #i = np.linalg.inv(Kern_mdl_truth)
    #c_mdl_deg_truth = np.dot(i,Y_train)
    #c_mdl_deg_truth = np.linalg.solve(Kern_mdl_truth,Y_train)
    y_truth = np.dot(Kern_mdl_truth,c_mdl)
    y_mdl_deg_truth = np.dot(Kern_mdl_truth,c_mdl_deg_truth)
    print('|| <X_train,c_mdl> - <X_train,c_mdl_deg_truth> ||^2 = {}'.format( np.linalg.norm(y_truth-Y_train) ) )
    print( '|| <X_train,c_mdl> - <X_train,c_mdl_deg_truth> ||^2 = {}'.format( np.linalg.norm(y_truth-y_mdl_deg_truth) ) )
    print('||c_truth - c_mdl_deg_truth||^2 = {}'.format( np.linalg.norm(c_mdl - c_mdl_deg_truth) ))
    if plotting:
        if D0 == 1:
            ## plot target func
            plot_target_function(c_mdl,X_train,Y_train,lb=lb,ub=ub,f_2_imitate=f_2_imitate)
            ## plot models to check
            c_mdls_2_plot = {}
            low_mdl,middle_mdl,high_mdl =int(largest_deg/4),int(largest_deg/2),largest_deg
            #low_mdl,middle_mdl,high_mdl = 21,22,23
            ##
            c_mdls_2_plot[Degree_data_set] = get_c_fit_data(X_train,Y_train,Degree_data_set)
            ##
            c_mdls_2_plot[low_mdl] = get_c_fit_data(X_train,Y_train,low_mdl)
            c_mdls_2_plot[middle_mdl] = get_c_fit_data(X_train,Y_train,middle_mdl)
            c_mdls_2_plot[high_mdl] = get_c_fit_data(X_train,Y_train,high_mdl)
            ##
            plot_poly_with_params(c_mdls_2_plot[Degree_data_set],X_train,Y_train,lb=lb,ub=ub)

            plot_poly_with_params(c_mdls_2_plot[low_mdl],X_train,Y_train,lb=lb,ub=ub)
            plot_poly_with_params(c_mdls_2_plot[middle_mdl],X_train,Y_train,lb=lb,ub=ub)
            plot_poly_with_params(c_mdls_2_plot[high_mdl],X_train,Y_train,lb=lb,ub=ub)
        ## plot errors
        plot_fig4(monomials,train_errors,test_errors,N_train,N_test)
        ## plot ranks
        fig1 = plt.figure()
        p_rank, = plt.plot(monomials,ranks,'c')
        p_h, = plt.plot(monomials, N_train*np.ones( (len(monomials),1) ) ,color='g',linestyle='--')
        p_v = plt.axvline(x=N_train,color='g',linestyle='--')
        plt.legend([p_rank,p_h,p_v], ['Plotting ranks','# Training data = {}'.format(N_train),'# Training data = {}'.format(N_train)])
        plt.xlabel('Monomials')
        plt.ylabel('Rank')
        plt.title('Rank of data set')
        ## plot singular values
        # fig1 = plt.figure()
        # p_s_inv_total, = plt.plot(monomials,s_inv_total,'c')
        # p_s_inv_max, = plt.plot(monomials,s_inv_max,'m')
        # p_v = plt.axvline(x=N_train,color='g',linestyle='--')
        # plt.legend([p_s_inv_total,p_s_inv_max,p_v], ['1/singular values total', '1/singular values max',
        #     '# Training data = {}'.format(N_train)])
        # plt.xlabel('Monomials')
        # plt.ylabel('1/singular values')
        # plt.title('1/singular values statistics of data set')
        ##
        plt.show()
    ##
    if 'file_name' not in kwargs:
        experiment_data = dict(monomials=monomials,train_errors=train_errors,test_errors=test_errors,
                N_train=N_train,N_test=N_test,
                Degree_data_set=Degree_data_set,c_mdl=c_mdl,nb_monomials_data=nb_monomials_data,
                mu_noise=mu_noise,std_noise=std_noise,
                X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test,
                title_fig='Training data size: {}'.format(N_train))
    else:
        new_data = dict(degrees=degrees,monomials=monomials,
            train_errors=train_errors,test_errors=test_errors)
        experiment_data = {**mat_dict,**new_data}
    ##
    if kwargs['save_overparam_experiment']:
        path_to_save = '../plotting/results/overfit_param_pinv_{}.mat'.format(SLURM_JOBID)
        scipy.io.savemat( path_to_save, experiment_data)

if __name__ == '__main__':
    ##
    start_time = time.time()
    ##
    my_main(plotting=True,save_overparam_experiment=True)
    ##
    #my_main(plotting=False,save_overparam_experiment=True,mat_load=True,file_name='../plotting/results/overfit_param_pinv_tommy_email2.mat')
    ##
    print('\a')
