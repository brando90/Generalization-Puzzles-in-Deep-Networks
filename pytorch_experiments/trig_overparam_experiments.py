import time
import numpy as np
from numpy.linalg import matrix_rank
import sys

import ast

import tensorflow as tf

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from maps import NamedDict as Maps
import pdb

from models_pytorch import *
from inits import *
from data_file import *
from pytorch_over_approx_high_dim import L2_norm_2
from pytorch_over_approx_high_dim import *

from plotting_utils import plot_with_params
from plotting_utils import plot_fig4

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.integrate as integrate
import scipy
import scipy.io

import argparse

SLURM_JOBID = 7

def get_f_2_imitate_D0_1(Degree_data_set):
    func_params = {}
    freq1, freq2 = 3, 2
    f_target = lambda x: np.sin(2*np.pi*freq1*x+2*np.pi*freq2*x)
    func_params['freq1']=freq1
    func_params['freq2']=freq2
    f_2_imitate = f_target
    return f_2_imitate, func_params

def get_target_Y_SP_trig(X_train,X_test,Degree_data_set,c_target,noise_train=0,noise_test=0):
    ## create features
    Kern_train = trig_kernel_matrix(X_train,Degree_data_set)
    Kern_test = trig_kernel_matrix(X_test,Degree_data_set)
    ## evaluate target function
    Y_train = np.dot(Kern_train,c_target)
    Y_test = np.dot(Kern_test,c_target)
    ## add noise to target
    Y_train, Y_test = Y_train+noise_train, Y_test+noise_test
    return Y_train, Y_test

def plot_target_function(c_target,X_train,Y_train,lb,ub,f_2_imitate):
    '''
    Note: training data os govem to visualize the training data set with the model
    '''
    ## plotting data (note this is NOT training data)
    N=5000
    x_plot_points = np.linspace(lb,ub,N).reshape(N,1) # [N,1]
    ##
    x_for_f = np.linspace(lb,ub,N)
    y_for_f = f_2_imitate( x_for_f )
    ## evaluate the model given on plot points
    if c_target is not None:
        deg = c_target.shape[0]-1
        Kern_plot_points = trig_kernel_matrix(x_plot_points,deg)
        y_plot_points = np.dot(Kern_plot_points,c_target)
    else:
        deg = 'Unknown/infinity'
        y_plot_points = f_2_imitate( x_for_f )
    #
    p_mdl, = plt.plot(x_plot_points,y_plot_points)
    p_f_2_imitate, = plt.plot(x_for_f,y_for_f)
    p_training_data, = plt.plot(X_train,Y_train,'ro')
    plt.legend([p_mdl,p_f_2_imitate,p_training_data], [f'Target function f(x) of degree {deg}','f trying to imitate','data points'])
    ##
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Target Function of degree {}'.format(deg))

def get_LA_error(X,Y,c,deg):
    N = X.shape[0]
    return (1/N)*(np.linalg.norm(Y-np.dot( trig_kernel_matrix(X,deg),c) )**2)

def get_nb_monomials(nb_variables,degree):
    return int(scipy.misc.comb(nb_variables+degree,degree))

def get_errors_pinv_mdls(X_train,Y_train,X_test,Y_test,degrees,lb,ub,f_target,c_target=None):
    train_errors, test_errors = [], []
    ranks = []
    s_inv_total, s_inv_max = [], []
    diff_truth = []
    ##
    N_train,D0 = X_train.shape
    ##
    for Degree_mdl in degrees:
        Kern_train = trig_kernel_matrix(X_train,Degree_mdl)
        Kern_train_pinv = np.linalg.pinv(Kern_train)
        c_pinv = np.dot(Kern_train_pinv, Y_train) # c = <K^+,Y>
        ##
        s_inv_total.append(-1)
        s_inv_max.append(-1)
        ##
        rank_kern_train = matrix_rank(Kern_train)
        ranks.append(rank_kern_train)
        # evluate it on train and test
        train_error = get_LA_error(X_train,Y_train,c_pinv,Degree_mdl)
        test_error = get_LA_error(X_test,Y_test,c_pinv,Degree_mdl)
        #
        print(f'train_error={train_error},test_error={test_error}')
        train_errors.append( train_error )
        test_errors.append( test_error )
        ##
        ##diff_truth.append(diff_target_pinv)
        if c_target is not None:
            if c_target.shape[0] == c_pinv.shape[0]:
                print('>>>> stats about param that matches target func')
                print( '||c_target - c_pinv||^2 = ',np.linalg.norm(c_target - c_pinv,2) )
                #diff_target_pinv = L2_norm_2(f_target,f_pinv,lb=lb,ub=ub)
                #print(f'diff_target_pinv = {diff_target_pinv}')
                print(f'train_error = {train_error}')
                print(f'test_error = {test_error}')
    return train_errors,test_errors,ranks,s_inv_total,s_inv_max,diff_truth

def get_c(nb_monomials_data):
    #pdb.set_trace()
    #c_target = 1*np.arange(0,nb_monomials_data).reshape(nb_monomials_data,1)+np.random.normal(loc=2.0,scale=1.0,size=(nb_monomials_data,1))
    c_target = np.array([ 2**d for d in np.arange(0,nb_monomials_data) ]).reshape(nb_monomials_data,1)
    #c_target = np.array([ 1.5**d for d in np.arange(0,nb_monomials_data) ]).reshape(nb_monomials_data,1)+np.random.normal(loc=3.0,scale=1.0,size=(nb_monomials_data,1))
    return c_target

def sample_X_D1(lb,ub,eps_train,eps_edge,N_left,N_middle,N_right,D0):
    ## middle
    middle = np.linspace(lb+eps_edge,ub-eps_edge,N_middle)
    #middle = ( ub-eps_train - (lb+eps_train))*np.random.rand(N_middle,D0)+(lb+eps_train)
    #middle = middle.reshape((N_middle,))
    ## edges
    left = np.linspace(lb,lb+eps_train, N_left)
    right = np.linspace(ub-eps_train,ub, N_right)
    #left = eps_train*np.random.rand(N_left,D0)+lb
    #left = left.reshape((N_left,))
    #right = eps_train*np.random.rand(N_right,D0)+(ub-eps_train)
    #right = right.reshape((N_right,))
    ##
    X_train = np.concatenate( (left,middle,right), axis=0)
    ##
    N_train = len(X_train)
    return X_train.reshape( (N_train,D0) )

def my_main(**kwargs):
    SGD = kwargs['SGD']
    ##
    lb,ub = -1,1
    eps_train = 0.0
    ##
    eps_edge = 0.00
    eps_test = eps_train
    eps_test = 0.0
    lb_test, ub_test = lb+eps_test, ub-eps_test
    ##
    start_time = time.time()
    plotting = kwargs['plotting'] if 'plotting' in kwargs else False
    freq = -1
    ## get target Y
    if 'file_name' in kwargs:
        mat_dict = scipy.io.loadmat(file_name=kwargs['file_name'])
        X_train, X_test = mat_dict['X_train'], mat_dict['X_test']
        Y_train, Y_test = mat_dict['Y_train'], mat_dict['Y_test']
        c_target = mat_dict['c_target']
        D0 = X_train.shape[1]
        N_train, N_test = X_train.shape[0], X_test.shape[0]
    else:
        ## properties of Data set
        D0 = 1
        N_test = 150
        N_train = 76
        #N_left,N_middle,N_right = 100,20,100
        #N_train = N_left+N_middle+N_right
        print(f'D0 = {D0}, N_train = {N_train}, N_test = {N_test}')
        ## get function to imitate and X input points
        #Degree_data_set = N_train-1
        Degree_data_set = 2*(N_train-1)
        nb_monomials_data = get_nb_monomials(nb_variables=D0,degree=Degree_data_set)
        print(f'> Degree_data_set={Degree_data_set}, nb_monomials_data={nb_monomials_data}')
        if D0 == 1:
            #X_train, X_test = 2*np.random.rand(N_train,D0)-1, 2*np.random.rand(N_test,D0)-1
            #X_train = (ub-lb)*np.random.rand(N_train,D0) + lb
            #X_test = (lb_test+ub_test)*np.random.rand(N_test,D0)-1
            X_train = np.linspace(lb,ub,N_train).reshape(N_train,D0)
            X_test = np.linspace(lb_test,ub_test,N_test).reshape(N_test,D0)
            #
            #X_train = get_chebyshev_nodes(lb,ub,N_train).reshape(N_train,D0)
            #X_train = sample_X_D1(lb,ub,eps_train,eps_edge,N_left=N_left,N_middle=N_middle,N_right=N_right,D0=D0)
            f_2_imitate,func_params = get_f_2_imitate_D0_1(Degree_data_set)
        elif D0 == 2:
            X_cord_train,Y_cord_train = generate_meshgrid(N_train,lb,ub)
            X_train, _ = make_mesh_grid_to_data_set(X_cord_train,Y_cord_train,Z=None)
            X_cord_test,Y_cord_test = generate_meshgrid(N_test,lb,ub)
            X_test, _ = make_mesh_grid_to_data_set(X_cord_test,Y_cord_test,Z=None)
            ##
            f_2_imitate,func_params = get_f_2_imitate_D0_2()
        else:
            # TODO
            raise ValueError(f'Not implemented D0={D0}')
        ## get actual target function
        N_data = Degree_data_set
        X_data,_ = get_X_Y_data(f_2_imitate, D0=D0, N=N_data, lb=lb,ub=ub)
        Y_data = f_2_imitate(X_data)
        #N_data = N_train
        #X_data,Y_data = X_train, f_2_imitate(X_train)
        c_target = get_c_fit_data(X_data,Y_data,Degree_data_set) # [Deg,1] sin with period k
        if c_target is None:
            nb_monomials_data = '?'
        #c_target = None
        ## get noise for target Y
        mu_noise, std_noise = 0,0
        noise_train, noise_test = 0,0
        ## get target Y
        f_target = f_2_imitate
        f_target.name = 'true_target'
        #Y_train, Y_test = get_target_Y_SP_poly(X_train,X_test, Degree_data_set, c_target, noise_train=noise_train,noise_test=noise_test)
        Y_train, Y_test = f_target(X_train)+noise_train, f_target(X_test)+noise_test
    ## get errors from models
    step_deg=1
    smallest_deg,largest_deg = 1,200
    degrees = list(range(smallest_deg,largest_deg,step_deg))
    train_errors_pinv,test_errors_pinv,ranks,s_inv_total,s_inv_max,diff_truth = get_errors_pinv_mdls(X_train,Y_train,X_test,Y_test,degrees, lb,ub,f_target,c_target)
    #train_errors_pinv,test_errors_pinv,_,_,_,_ = get_errors_pinv_mdls(X_train,Y_train,X_test,Y_test,degrees, lb,ub,f_target,c_target)
    ## plot them
    monomials = [ get_nb_monomials(nb_variables=D0,degree=d) for d in degrees ]
    ## REPORT TIMES
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    print("\a--- {} seconds --- \n --- {} minutes --- \n --- {} hours ---".format(seconds, minutes, hours) )
    ##
    print('Degree_data_set = {}'.format(Degree_data_set))
    print('N_train = {} '.format(N_train))
    # if c_target is not None:
    #     print('||c_truth - c_target_deg_truth||^2 = {}'.format( np.linalg.norm(c_target - c_target_deg_truth) ))
    if plotting:
        if D0==1:
            ## plot target func
            #plot_target_function(c_target,X_train,Y_train,lb=lb,ub=ub,f_2_imitate=f_2_imitate)
            ## plot models to check
            c_targets_2_plot = {}
            low_mdl,middle_mdl,high_mdl =int(largest_deg/6),int(largest_deg/2),largest_deg
            #low_mdl,middle_mdl,high_mdl = 21,22,23
            low_mdl,middle_mdl,high_mdl = 12,25,100
            ##
            c_targets_2_plot[Degree_data_set] = get_c_fit_data(X_train,Y_train,Degree_data_set)
            ##
            c_targets_2_plot[low_mdl] = get_c_fit_data(X_train,Y_train,low_mdl)
            c_targets_2_plot[middle_mdl] = get_c_fit_data(X_train,Y_train,middle_mdl)
            c_targets_2_plot[high_mdl] = get_c_fit_data(X_train,Y_train,high_mdl)
            ##
            #plot_with_params(c_targets_2_plot[Degree_data_set],X_train,Y_train,lb=lb,ub=ub)

            #plot_with_params(c_targets_2_plot[low_mdl],X_train,Y_train,lb=lb,ub=ub)
            #plot_with_params(c_targets_2_plot[middle_mdl],X_train,Y_train,lb=lb,ub=ub)
            #plot_with_params(c_targets_2_plot[high_mdl],X_train,Y_train,lb=lb,ub=ub)
            ##
            if len(diff_truth) != 0:
                fig = plt.figure()
                plt_diff_truth, = plt.plot(monomials,diff_truth)
                plt.legend([plt_diff_truth],['difference of model (pinv) vs c_target'])
        elif D0==2:
            #
            _,_,Z_cord_train = make_meshgrid_data_from_training_data(X_data=X_train, Y_data=Y_train) # meshgrid for trainign points visualization
            _,_,Z_cord_test = make_meshgrid_data_from_training_data(X_data=X_test, Y_data=Y_test) # meshgrid for function visualization
            ## fig target function
            fig1 = plt.figure()
            ax1 = Axes3D(fig1)
            data_pts = ax1.scatter(X_cord_train,Y_cord_train,Z_cord_train, marker='D')
            surf = ax1.plot_surface(X_cord_test,Y_cord_test,Z_cord_test,color='y',cmap=cm.coolwarm)
            ax1.set_xlabel('x1'),ax1.set_ylabel('x2'),ax1.set_zlabel('f(x)')
            surf_proxy = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker ='_')
            ax1.legend([surf_proxy,data_pts],[
                f'target function degree={Degree_data_set}, number of monomials={nb_monomials_data}',
                f'data points, number of data points = {N_train}'])
        else:
            #TODO
            raise ValueError("not implemented yet")
        ## plot errors
        if SGD:
            plot_fig4(monomials,train_errors,test_errors,N_train,N_test,nb_monomials_data,alg=f'SGD, eps_test={eps_test} ')
        plot_fig4(monomials,train_errors_pinv,test_errors_pinv,N_train,N_test,nb_monomials_data,alg=f'pinv, eps_test={eps_test} ')
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
        print(f'\a N<2*sqrt(N_train), ={2*(N_train**0.5)}')
        plt.show()
    ##
    if c_target is None:
        c_target = 'None'
    if 'file_name' not in kwargs:
        if not SGD:
            train_errors, test_errors = train_errors_pinv, test_errors_pinv
        experiment_data = dict(monomials=monomials,train_errors=train_errors,test_errors=test_errors,
                N_train=N_train,N_test=N_test,
                Degree_data_set=Degree_data_set,c_target=c_target,nb_monomials_data=nb_monomials_data,
                mu_noise=mu_noise,std_noise=std_noise,
                X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test,
                title_fig=f'Training data size: {N_train}' )
    else:
        new_data = dict(degrees=degrees,monomials=monomials,
            train_errors=train_errors,test_errors=test_errors)
        experiment_data = {**mat_dict,**new_data}
    ##
    if kwargs['save_overparam_experiment']:
        print('SAVING EXPT')
        path_to_save = f'../plotting/results/overfit_param_pinv_{SLURM_JOBID}.mat'
        scipy.io.savemat( path_to_save, experiment_data)
    ##

if __name__ == '__main__':
    ##
    start_time = time.time()
    ##
    my_main(plotting=True,save_overparam_experiment=True,SGD=False,save_data_set=True)
    ##
    #my_main(plotting=False,save_overparam_experiment=True,mat_load=True,file_name='../plotting/results/overfit_param_pinv_tommy_email2.mat')
    ##
    print('\a')
