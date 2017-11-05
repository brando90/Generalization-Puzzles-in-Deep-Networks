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

def get_f_2_imitate_D0_1(Degree_data_set):
    # TODO
    D0 = 1
    freq_cos = 2
    freq_sin = 1
    #c_target = np.arange(1,nb_monomials_data+1).reshape((nb_monomials_data,1))+np.random.normal(loc=3.0,scale=1.0,size=(nb_monomials_data,1))
    #c_target = get_c(nb_monomials_data) # [D,1]
    #c_target = get_c_fit_function(generate_h_add_1d,D0,Degree_data_set,N=3*N_test,lb=-1,ub=1)
    #c_target = get_c_fit_function(generate_h_gabor_1d,D0,Degree_data_set,N=3*N_test,lb=-1,ub=1)
    # freq = max(freq_sin, freq_cos)
    # f_2_imitate = lambda x: np.cos(freq_cos*2*np.pi*x)
    # c_target = get_c_fit_function(f_2_imitate, D0,Degree_data_set, N=20000, lb=lb,ub=ub) # [Deg,1] sin with period k
    #c_target = get_c_fit_function(lambda x: np.exp( -(x**2) )*np.cos(4*np.pi*(x)),  D0,Degree_data_set, N=3*N_test, lb=lb,ub=ub)
    #c_target = get_c_fit_function(lambda x: np.exp( -(x**2) )*( np.cos(freq_sin*np.pi*(x)) + np.sin(freq_cos*np.pi*(x)) ),  D0,Degree_data_set, N=30*N_test, lb=lb,ub=ub)
    ##
    def poly(x):
        N = x.shape[0]
        poly_feat = PolynomialFeatures(degree=Degree_data_set)
        Kern = poly_feat.fit_transform( x.reshape(N,D0) ) # low degrees first [1,x,x**2,...]
        nb_monomials_data = Kern.shape[1]
        #c_target = np.random.normal(loc=10.0,scale=5.0,size=(nb_monomials_data,1))
        #c_target = np.arange(1,nb_monomials_data+1).reshape((nb_monomials_data,1))+np.random.normal(loc=3.0,scale=1.0,size=(nb_monomials_data,1))
        c_target = np.arange(1,nb_monomials_data+1).reshape((nb_monomials_data,1))
        #c_target = np.array( [ c**10 for c in np.arange(1,nb_monomials_data+1)] )[::-1]
        #c_target = np.array( [ c**10 for c in np.arange(1,nb_monomials_data+1)] )
        #print(c_target)
        # y=1
        # #d=np.linspace(lb,ub,Degree_data_set)
        # d=(ub-lb)*np.random.rand(Degree_data_set,1)-lb
        # for i in range(len(d)):
        #     y=y*(x-d[i])
        #y = np.dot(Kern,c_target).reshape((N,))
        y = np.dot(Kern,c_target)
        #print('poly',y.shape)
        return y
    #f_2_imitate = lambda x: poly(x)+np.exp(-x**2)*np.cos(freq_cos*2*np.pi*x)
    def f_2_imitate(x):
        N = x.shape[0]
        #y1 = poly(x)
        #print('y1',y1.shape)
        #y2 = np.exp(-x**2)*np.cos(freq_cos*2*np.pi*x)
        #y2 = np.exp(-x**2)*np.sin(freq_sin*2*np.pi*x)
        #y2 = np.sin(freq_sin*2*np.pi*x)
        #y2 = np.cos(freq_cos*2*np.pi*x)
        #y2 = np.cos(freq_cos*2*np.pi*x+1.3*np.pi)*np.sin(freq_sin*2*np.pi*x)
        #y2 = 100*np.cos(freq_cos*2*np.pi*x)*np.sin(freq_sin*2*np.pi*x)
        #print('y2',y2.shape)
        #y=y1+y2.reshape((N,D0))
        y=y2.reshape((N,D0))
        #y=y1
        return y
    return f_2_imitate

def get_f_2_imitate_D0_2(Degree_data_set):
    #
    f_2_imitate = lambda X,Y: sin(2*pi*X) + 4*(Y - 0.5).^2
    #f_2_imitate = lambda X,Y: np.exp( -(X**2 + Y**2) )*np.cos(2*np.pi*(X+Y))
    return f_2_imitate

def get_target_Y_SP_poly(X_train,X_test,Degree_data_set,c_target,noise_train=0,noise_test=0):
    ## get data points
    poly_feat = PolynomialFeatures(degree=Degree_data_set)
    ## create poly features
    Kern_train = poly_feat.fit_transform(X_train)
    Kern_test = poly_feat.fit_transform(X_test)
    ## evaluate target function
    Y_train = np.dot(Kern_train,c_target)
    Y_test = np.dot(Kern_test,c_target)
    ## add noise to target
    Y_train, Y_test = Y_train+noise_train, Y_test+noise_test
    return Y_train, Y_test

def get_c_fit_function(target_f,D0,degree_mdl,N,lb,ub):
    ## evaluate target_f on x_points
    if D0 == 1:
        X = np.linspace(lb,ub,N).reshape(N,D0) # [N,D0]
        Y = target_f(X) #
        ## copy that f with the target degree polynomial
        #poly_feat = PolynomialFeatures(degree=degree_mdl)
        #Kern = poly_feat.fit_transform(X)
        #c_target = np.dot(np.linalg.pinv( Kern ), Y)
        print(X.shape)
        print(Y.shape)
        print(N)
        c_target = np.polyfit(X.reshape((N,)),Y.reshape((N,)),degree_mdl)[::-1]
    elif D0 == 2:
        ## TODO
        ## Lift data/Kernelize data
        poly_feat = PolynomialFeatures(degree=Degree_mdl)
        Kern_train = poly_feat.fit_transform(X_train)
        Kern_test = poly_feat.fit_transform(X_test)
        ## LA models
        c_target = np.dot(np.linalg.pinv( Kern_train ),Y_train)
    else:
        # TODO
        raise ValueError(f'Not implemented D0={D0}')
    return c_target

def get_c_fit_data(X,Y,degree_mdl):
    N = X.shape[0]
    ## copy that f with the target degree polynomial
    poly_feat = PolynomialFeatures(degree=degree_mdl)
    Kern = poly_feat.fit_transform(X)
    #c_target = np.dot(np.linalg.pinv( Kern ), Y)
    # print(degree_mdl)
    # print(Y)
    # print(degree_mdl)
    c_target  = np.polyfit( X.reshape((N,)) , Y.reshape((N,)) , degree_mdl )[::-1]
    return c_target

def plot_target_function(c_target,X_train,Y_train,lb,ub,f_2_imitate):
    '''
    Note: training data os govem to visualize the training data set with the model
    '''
    deg = c_target.shape[0]-1
    ## plotting data (note this is NOT training data)
    N=50000
    x_plot_points = np.linspace(lb,ub,N).reshape(N,1) # [N,1]
    ## evaluate the model given on plot points
    poly_feat = PolynomialFeatures(degree=deg)
    Kern_plot_points = poly_feat.fit_transform(x_plot_points)
    y_plot_points = np.dot(Kern_plot_points,c_target)
    #
    x_for_f = np.linspace(lb,ub,50000)
    #pdb.set_trace()
    y_for_f = f_2_imitate( x_for_f )
    # print(x_for_f)
    # print(f_2_imitate( x_for_f ))
    # N=5
    # lb,ub = -6,6
    # freq_cos = 0.05
    # freq_sin = 0.3
    # f = lambda x: np.cos(freq_cos*2*np.pi*x)*np.sin(freq_sin*2*np.pi*x)
    # x = np.linspace(lb,ub,N)
    # print(x)
    # print(f(x))
    # pdb.set_trace()
    #
    p_mdl, = plt.plot(x_plot_points,y_plot_points)
    p_f_2_imitate, = plt.plot(x_for_f,y_for_f)
    p_training_data, = plt.plot(X_train,Y_train,'ro')
    plt.legend([p_mdl,p_f_2_imitate,p_training_data], ['Target function f(x) of degree {}'.format(deg),'f trying to imitate','data points'])
    ##
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Target Function of degree {}'.format(deg))

def plot_poly_with_params(c_target,X_train,Y_train,lb,ub):
    '''
    Note: training data os govem to visualize the training data set with the model
    '''
    fig = plt.figure()
    deg = c_target.shape[0]-1
    ## plotting data (note this is NOT training data)
    N=5000
    x_plot_points = np.linspace(lb,ub,N).reshape(N,1) # [N,1]
    ## evaluate the model given on plot points
    poly_feat = PolynomialFeatures(degree=deg)
    Kern_plot_points = poly_feat.fit_transform(x_plot_points)
    y_plot_points = np.dot(Kern_plot_points,c_target)
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
        #Kern_train_pinv = np.linalg.pinv( Kern_train )
        #Kern_train_pinv = my_pinv(Kern_train)
        #c_pinv = np.dot(Kern_train_pinv, Y_train) # c = <K^+,Y>
        c_pinv = np.polyfit(X_train.reshape((N_train,)),Y_train.reshape((N_train,)),degree_mdl)[::-1]
        #pdb.set_trace()
        #c_pinv = np.polyfit(X_train.reshape((N_train,)),Y_train.reshape((N_train,)),deg=degree_mdl)[::-1]
        ##
        #try:
        # Kern_train_pinv = np.linalg.pinv( Kern_train )
        # U,S,V = np.linalg.svd(Kern_train_pinv)
        # s_inv_total.append( np.sum(S) )
        # s_inv_max.append(max(S))
        # except:
        #     s_inv_total.append( -1 )
        #     s_inv_max.append(-1)
        #pdb.set_trace()
        ##
        #ranks.append(-1)
        s_inv_total.append( -1 )
        s_inv_max.append(-1)
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
    #c_target = 1*np.arange(0,nb_monomials_data).reshape(nb_monomials_data,1)+np.random.normal(loc=2.0,scale=1.0,size=(nb_monomials_data,1))
    c_target = np.array([ 2**d for d in np.arange(0,nb_monomials_data) ]).reshape(nb_monomials_data,1)
    #c_target = np.array([ 1.5**d for d in np.arange(0,nb_monomials_data) ]).reshape(nb_monomials_data,1)+np.random.normal(loc=3.0,scale=1.0,size=(nb_monomials_data,1))
    return c_target

def my_main(**kwargs):
    ##
    lb,ub = 1,2
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
        D0 = 2
        N_train, N_test = 40, 60
        print('D0 = {}, N_train = {}, N_test = {}'.format(D0,N_train,N_test))
        ## get function to imitate and X input points
        Degree_data_set = 500
        nb_monomials_data = get_nb_monomials(nb_variables=D0,degree=Degree_data_set)
        if D0 == 1:
            #X_train, X_test = 2*np.random.rand(N_train,D0)-1, 2*np.random.rand(N_test,D0)-1
            X_train, X_test = np.linspace(lb,ub,N_train).reshape(N_train,D0), np.linspace(lb,ub,N_test).reshape(N_test,D0)
            f_2_imitate = get_f_2_imitate_D0_1(Degree_data_set)
        elif D0 == 2:
            X,Y = generate_meshgrid(N,lb,ub)
            X_train, _ = make_mesh_grid_to_data_set(X,Y,Z=None)
            X_data,Y_data = f_2_imitate = get_f_2_imitate_D0_2(Degree_data_set)
        else:
            # TODO
            raise ValueError(f'Not implemented D0={D0}')
        ## get actual (polynomial) target function
        c_target = get_c_fit_function(f_2_imitate, D0,Degree_data_set, N=25000, lb=lb,ub=ub) # [Deg,1] sin with period k
        def f_target(x):
            poly_feat = PolynomialFeatures(degree=Degree_data_set)
            Kern = poly_feat.fit_transform(x)
            return  np.dot(Kern,c_target)
        ## get noise for target Y
        mu_noise, std_noise = 0, 0.0
        noise_train, noise_test = 0, 0
        ## get target Y
        Y_train, Y_test = get_target_Y_SP_poly(X_train,X_test, Degree_data_set, c_target, noise_train=noise_train,noise_test=noise_test)
    ## print
    print('c_target = ',c_target)
    print('c_target.shape = ',c_target.shape)
    print('nb_monomials_data = {} \n'.format(nb_monomials_data) )
    ## get errors from models
    step_deg=1
    smallest_deg,largest_deg = 1,70
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
    # print('f_s > 2F_max = N_train > 4 freq = {} > 4*{} =  {} > {} ?, Is it true: {}'.format(N_train,freq, N_train,4*freq, N_train>4*freq))
    # print('number of zeros sine = {}'.format( int( 2*2/(1/freq_sin) )   ))
    # print('number of zeros cos = {}'.format( int( 2*2/(1/freq_cos) )    ))
    # print('total number of zeros = {}'.format(   int( 2*2/(1/freq_cos) ) + int( 2*2/(1/freq_sin) )   ))
    ##
    poly_feat = PolynomialFeatures(degree=Degree_data_set)
    Kern_mdl_truth = poly_feat.fit_transform(X_train)
    print('Degree_data_set = {}'.format(Degree_data_set))
    print('rank(Kern_mdl_truth) = {}'.format( matrix_rank(Kern_mdl_truth) ))
    print('N_train = {} '.format(N_train))
    print('Kern_mdl_truth.shape = {}'.format(Kern_mdl_truth.shape))
    c_target_deg_truth = get_c_fit_data(X_train,Y_train,Degree_data_set) # model with same degree as truth but trained on training set
    #i = np.linalg.inv(Kern_mdl_truth)
    #c_target_deg_truth = np.dot(i,Y_train)
    #c_target_deg_truth = np.linalg.solve(Kern_mdl_truth,Y_train)
    y_truth = np.dot(Kern_mdl_truth,c_target)
    y_mdl_deg_truth = np.dot(Kern_mdl_truth,c_target_deg_truth)
    print('|| <X_train,c_target> - <X_train,c_target_deg_truth> ||^2 = {}'.format( np.linalg.norm(y_truth-Y_train) ) )
    print( '|| <X_train,c_target> - <X_train,c_target_deg_truth> ||^2 = {}'.format( np.linalg.norm(y_truth-y_mdl_deg_truth) ) )
    print('||c_truth - c_target_deg_truth||^2 = {}'.format( np.linalg.norm(c_target - c_target_deg_truth) ))
    if plotting:
        if D0 == 1:
            ## plot target func
            plot_target_function(c_target,X_train,Y_train,lb=lb,ub=ub,f_2_imitate=f_2_imitate)
            ## plot models to check
            c_targets_2_plot = {}
            low_mdl,middle_mdl,high_mdl =int(largest_deg/6),int(largest_deg/2),largest_deg
            #low_mdl,middle_mdl,high_mdl = 21,22,23
            ##
            c_targets_2_plot[Degree_data_set] = get_c_fit_data(X_train,Y_train,Degree_data_set)
            ##
            c_targets_2_plot[low_mdl] = get_c_fit_data(X_train,Y_train,low_mdl)
            c_targets_2_plot[middle_mdl] = get_c_fit_data(X_train,Y_train,middle_mdl)
            c_targets_2_plot[high_mdl] = get_c_fit_data(X_train,Y_train,high_mdl)
            ##
            plot_poly_with_params(c_targets_2_plot[Degree_data_set],X_train,Y_train,lb=lb,ub=ub)

            plot_poly_with_params(c_targets_2_plot[low_mdl],X_train,Y_train,lb=lb,ub=ub)
            plot_poly_with_params(c_targets_2_plot[middle_mdl],X_train,Y_train,lb=lb,ub=ub)
            plot_poly_with_params(c_targets_2_plot[high_mdl],X_train,Y_train,lb=lb,ub=ub)
        elif D0==2:
            #
            Xp,Yp,Zp = make_meshgrid_data_from_training_data(X_test=X_test, Y_test=Y_test) # meshgrid for visualization
            Xp_train,Yp_train,Zp_train = make_meshgrid_data_from_training_data(X_test=X_train, Y_test=Y_train) # meshgrid for trainign points
            #
            Y_pinv = np.dot(poly_feat.fit_transform(X_test),c_pinv)
            _,_,Zp_pinv = make_meshgrid_data_from_training_data(X_test=X_test, Y_test=Y_pinv)
            ## FIG PINV
            fig1 = plt.figure()
            ax1 = Axes3D(fig1)
            data_pts = ax1.scatter(Xp_train,Yp_train,Zp_train, marker='D')
            surf = ax1.plot_surface(Xp,Yp,Zp_pinv,color='y',cmap=cm.coolwarm)
            ax1.set_xlabel('x1'),ax1.set_ylabel('x2'),ax1.set_zlabel('f(x)')
            surf_proxy = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker ='_')
            ax1.legend([surf_proxy,data_pts],[
                'minimum norm solution Degree model={}, number of monomials={}'.format(Degree_mdl,nb_monomials),
                'data points, number of data points = {}'.format(N_train)])
            ## FIG SGD standard param
            fig = plt.figure()
            ax3 = Axes3D(fig)
            data_pts = ax3.scatter(Xp_train,Yp_train,Zp_train, marker='D')
            surf = ax3.plot_surface(Xp,Yp,Zp_sgd_stand, cmap=cm.coolwarm)
            ax3.set_xlabel('x1'),ax3.set_ylabel('x2'),ax3.set_zlabel('f(x)')
            ax3.legend([surf_proxy,data_pts],[
                'SGD solution standard parametrization Degree model={}, number of monomials={}, param count={}, batch-size={}, iterations={}, step size={}'.format(degree_sgd,nb_monomials,nb_monomials,M_standard_sgd,nb_iter_standard_sgd,eta_standard_sgd),
                'data points, number of data points = {}'.format(N_train)])
            ## PLOT train surface
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # points_scatter = ax.scatter(Xp_train,Yp_train,Zp_train, marker='D')
            # surf = ax.plot_surface(Xp_train,Yp_train,Zp_train, cmap=cm.coolwarm)
            # plt.title('Train function')
            # ## PLOT test surface
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # points_scatter = ax.scatter(Xp,Yp,Zp, marker='D')
            # surf = ax.plot_surface(Xp,Yp,Zp, cmap=cm.coolwarm)
            # plt.title('Test function')
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
                Degree_data_set=Degree_data_set,c_target=c_target,nb_monomials_data=nb_monomials_data,
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
