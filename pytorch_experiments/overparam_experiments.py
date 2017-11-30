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
from sympy_poly import *
from poly_checks_on_deep_net_coeffs import *
from data_file import *
from pytorch_over_approx_high_dim import L2_norm_2
from pytorch_over_approx_high_dim import *

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.integrate as integrate
import scipy
import scipy.io

import argparse

SLURM_JOBID = 7

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

def get_batch(X,Y,M):
    N = len(Y)
    valid_indices = np.array( range(N) )
    batch_indices = np.random.choice(valid_indices,size=M,replace=False)
    batch_xs = X[batch_indices,:]
    batch_ys = Y[batch_indices]
    return batch_xs, batch_ys

def train_tf(nb_monomials,Kern_train,Y_train, Kern_test,Y_test, eta,nb_iter,M):
    N_train,_ = Y_train.shape
    ##
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32, [None, nb_monomials])
        Y = tf.placeholder(tf.float32, [None,1])
        w = tf.Variable( tf.zeros([nb_monomials,1]) )
        #w = tf.Variable( tf.truncated_normal([Degree_mdl,1],mean=0.0,stddev=1.0) )
        ##
        f = tf.matmul(X,w) # [N,1] = [N,D] x [D,1]
        #loss = tf.reduce_sum(tf.square(Y - f))
        loss = tf.reduce_sum( tf.reduce_mean(tf.square(Y-f), 0))
        l2loss_tf = (1/N_train)*2*tf.nn.l2_loss(Y-f)
        ##
        learning_rate = eta
        #global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(learning_rate=eta, global_step=global_step,decay_steps=nb_iter/2, decay_rate=1, staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
        with tf.Session(graph=graph) as sess:
            Y_train = Y_train.reshape(N_train,1)
            tf.global_variables_initializer().run()
            # Train
            for i in range(nb_iter):
                #if i % (nb_iter/10) == 0:
                #if i % (nb_iter/10) == 0 or i == 0:
                if True:
                    current_loss = sess.run(fetches=loss, feed_dict={X: Kern_train, Y: Y_train})
                    print(f'tf: i = {i}, current_loss = {current_loss}')
                ## train
                batch_xs, batch_ys = get_batch(Kern_train,Y_train,M)
                sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})
            ## prepare tf plot point
            #x_horizontal = np.linspace(lb,ub,1000).reshape(1000,1)
            #Kern_plot = poly_feat.fit_transform(x_horizontal)
            #Y_tf = sess.run(f,feed_dict={X:Kern_plot, Y:Y_train})
            train_error = sess.run(fetches=loss, feed_dict={X: Kern_train, Y: Y_train})
            test_error = sess.run(fetches=loss, feed_dict={X: Kern_test, Y: Y_test})
            return train_error, test_error

def g():
    Y_pinv = np.dot(poly_feat.fit_transform(X_test),c_pinv)
    _,_,Zp_pinv = make_meshgrid_data_from_training_data(X_test=X_test, Y_test=Y_pinv)
    return

def f_target(x):
    poly_feat = PolynomialFeatures(degree=Degree_data_set)
    Kern = poly_feat.fit_transform(x)
    return  np.dot(Kern,c_target)

def get_f_2_imitate_D0_1(Degree_data_set):
    # TODO
    func_params = {}
    ##
    D0 = 1
    freq_cos = 1
    #freq_sin = 4
    freq_sin = 2.3
    func_params['freq_sin'] = freq_sin
    #c_target = np.arange(1,nb_monomials_data+1).reshape((nb_monomials_data,1))+np.random.normal(loc=3.0,scale=1.0,size=(nb_monomials_data,1))
    #c_target = get_c(nb_monomials_data) # [D,1]
    #c_target = get_c_fit_data(generate_h_add_1d,D0,Degree_data_set,N=3*N_test,lb=-1,ub=1)
    #c_target = get_c_fit_data(generate_h_gabor_1d,D0,Degree_data_set,N=3*N_test,lb=-1,ub=1)
    # freq = max(freq_sin, freq_cos)
    # f_2_imitate = lambda x: np.cos(freq_cos*2*np.pi*x)
    # c_target = get_c_fit_data(f_2_imitate, D0,Degree_data_set, N=20000, lb=lb,ub=ub) # [Deg,1] sin with period k
    #c_target = get_c_fit_data(lambda x: np.exp( -(x**2) )*np.cos(4*np.pi*(x)),  D0,Degree_data_set, N=3*N_test, lb=lb,ub=ub)
    #c_target = get_c_fit_data(lambda x: np.exp( -(x**2) )*( np.cos(freq_sin*np.pi*(x)) + np.sin(freq_cos*np.pi*(x)) ),  D0,Degree_data_set, N=30*N_test, lb=lb,ub=ub)
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
        if type(x) != float:
            N = x.shape[0]
        else:
            N=1
        #y1 = poly(x)
        #print('y1',y1.shape)
        #y2 = np.exp(-x**2)*np.cos(freq_cos*2*np.pi*x)
        #y2 = np.exp(-x**2)*np.sin(freq_sin*2*np.pi*x)
        y2 = np.sin(freq_sin*2*np.pi*x)
        #y2 = np.cos(freq_cos*2*np.pi*x)
        #y2 = np.cos(freq_cos*2*np.pi*x+1.3*np.pi)*np.sin(freq_sin*2*np.pi*x)
        #y2 = 100*np.cos(freq_cos*2*np.pi*x)*np.sin(freq_sin*2*np.pi*x)
        #print('y2',y2.shape)
        #y=y1+y2.reshape((N,D0))
        y=y2.reshape((N,D0))
        #y=y1
        return y
    return f_2_imitate, func_params

def get_f_2_imitate_D0_2():
    #
    #f_2_imitate = lambda X,Y: np.sin(2*np.pi*X) + 4*(Y - 2)**2
    #f_2_imitate = lambda X,Y: np.exp( -(X**2 + Y**2) )*np.cos(2*np.pi*(X+Y))
    f_2_imitate = lambda X,Y: np.cos(2*np.pi*(X+Y))
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

def plot_target_function(c_target,X_train,Y_train,lb,ub,f_2_imitate):
    '''
    Note: training data os govem to visualize the training data set with the model
    '''
    ## plotting data (note this is NOT training data)
    N=50000
    x_plot_points = np.linspace(lb,ub,N).reshape(N,1) # [N,1]
    ##
    x_for_f = np.linspace(lb,ub,50000)
    y_for_f = f_2_imitate( x_for_f )
    ## evaluate the model given on plot points
    if c_target is not None:
        deg = c_target.shape[0]-1
        poly_feat = PolynomialFeatures(degree=deg)
        Kern_plot_points = poly_feat.fit_transform(x_plot_points)
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

def get_LA_error(X,Y,c,poly_feat):
    N = X.shape[0]
    return (1/N)*(np.linalg.norm(Y-np.dot( poly_feat.fit_transform(X),c) )**2)

def get_nb_monomials(nb_variables,degree):
    return int(scipy.misc.comb(nb_variables+degree,degree))

def my_pinv(X):
    XXt_inv = np.linalg.inv(np.dot(X,X.T))
    X_my_pinv = np.dot(X.T ,XXt_inv )
    return X_my_pinv

def get_errors_pinv_mdls(X_train,Y_train,X_test,Y_test,degrees,lb,ub,f_target,c_target=None):
    train_errors, test_errors = [], []
    ranks = []
    s_inv_total, s_inv_max = [], []
    diff_truth = []
    ##
    N_train,D0 = X_train.shape
    ##
    for degree_mdl in degrees:
        poly_feat = PolynomialFeatures(degree=degree_mdl)
        # get mdl
        Kern_train = poly_feat.fit_transform(X_train)
        if D0==1:
            #c_pinv = np.polyfit(X_train .reshape((N_train,)),Y_train.reshape((N_train,)),degree_mdl)[::-1]
            #c_pinv = np.polyfit(X_train.reshape((N_train,)),Y_train.reshape((N_train,)),degree_mdl)
            Kern_train_pinv = np.linalg.pinv( Kern_train )
            c_pinv = np.dot(Kern_train_pinv, Y_train)
        else:
            Kern_train_pinv = np.linalg.pinv( Kern_train )
            c_pinv = np.dot(Kern_train_pinv, Y_train) # c = <K^+,Y>
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
        print(f'train_error={train_error},test_error={test_error}')
        train_errors.append( train_error )
        test_errors.append( test_error )
        # difference with truth/target function
        #diff_truth.append( np.linalg.norm(c_target - c_pinv,2) )
        #f_target= get_func_pointer_poly(c_target,c_target.shape[0]-1,D0)
        f_pinv =  get_func_pointer_poly(c_pinv,degree_mdl,D0)
        diff_target_pinv = L2_norm_2(f_target,f_pinv,lb=lb+0.2,ub=ub-0.2)
        #diff_target_pinv = L2_norm_2(f_target,f_pinv,lb=lb,ub=ub)
        diff_truth.append( diff_target_pinv )
        if c_target is not None:
            if c_target.shape[0] == c_pinv.shape[0]:
                print('>>>> stats about param that matches target func')
                print( '||c_target - c_pinv||^2 = ',np.linalg.norm(c_target - c_pinv,2) )
                diff_target_pinv = L2_norm_2(f_target,f_pinv,lb=lb,ub=ub)
                print(f'diff_target_pinv = {diff_target_pinv}')
                print(f'train_error = {train_error}')
                print(f'test_error = {test_error}')
    return train_errors,test_errors,ranks,s_inv_total,s_inv_max,diff_truth

def get_errors_pinv_mdls_SGD(X_train,Y_train,X_test,Y_test,degrees,lb,ub,f_target,c_target=None,bias=False):
    train_errors, test_errors = [], []
    ranks = []
    s_inv_total, s_inv_max = [], []
    diff_truth = []
    ##
    N_train,D0 = X_train.shape
    N_test,D_out = Y_test.shape
    ##
    M = 3
    eta = 0.01
    nb_iter = 10
    A = 0
    logging_freq = 100
    #
    print(f'X_train.shape ,Y_train.shape ,X_test.shape ,Y_test.shape = {X_train.shape},{Y_train.shape} ,{X_test.shape},{Y_test.shape}')
    dtype = torch.FloatTensor
    ##
    for degree_mdl in degrees:
        print(f'\ndegree_mdl={degree_mdl}')
        poly_feat = PolynomialFeatures(degree=degree_mdl)
        # get mdl
        Kern_train = poly_feat.fit_transform(X_train)
        Kern_test = poly_feat.fit_transform(X_test)
        print(f'Kern_train.shape={Kern_train.shape},Kern_test={Kern_test.shape}')
        Kern_train_pinv = np.linalg.pinv( Kern_train )
        c_pinv = np.dot(Kern_train_pinv, Y_train)
        train_error_pinv = get_LA_error(X_train,Y_train,c_pinv,poly_feat)
        test_error_pinv = get_LA_error(X_test,Y_test,c_pinv,poly_feat)
        print(f'train_error_pinv={train_error_pinv}, test_error_pinv={test_error_pinv}')
        nb_monomials = int(scipy.misc.comb(D0+degree_mdl,degree_mdl))
        print(f'nb_monomials={nb_monomials}')
        mdl_sgd = torch.nn.Sequential(torch.nn.Linear(nb_monomials,D_out,bias=bias))
        #mdl_sgd[0].weight.data.copy_( torch.FloatTensor(c_pinv) )
        mdl_sgd[0].weight.data.fill_(0)
        print(f'mdl_sgd={mdl_sgd[0].weight.data.size()}')
        # evluate it on train and test
        #print(data)
        ##
        data = get_data_struct(X_train,Y_train,X_test,Y_test, Kern_train,Kern_test, dtype)
        data.X_train, data.X_test = data.Kern_train, data.Kern_test
        output = train_SGD( arg,mdl_sgd,data, M,eta,nb_iter,A ,logging_freq ,dtype=torch.FloatTensor,c_pinv=c_pinv,reg_lambda=0)
        #train_loss_list_WP,test_loss_list_WP,grad_list_weight_sgd,func_diff_weight_sgd,erm_lamdas_WP,nb_module_params = output
        ##
        train_error = (1/N_train)*( mdl_sgd(data.X_train) - data.Y_train ).pow(2).sum().data.numpy()
        test_error = (1/N_test)*( mdl_sgd(data.X_test) - data.Y_test ).pow(2).sum().data.numpy()
        #
        train_errors.append( train_error )
        test_errors.append( test_error )
        print(f'train_error={train_error}, test_error={test_error}')
        ##
        f_pinv =  get_func_pointer_poly(c_pinv,degree_mdl,D0)
        #diff_target_pinv = L2_norm_2(f_target,f_pinv,lb=lb+0.2,ub=ub-0.2)
        #diff_truth.append(-1)
        ##
        rank_kern_train = matrix_rank(Kern_train)
        ranks.append(rank_kern_train)
    return train_errors,test_errors,ranks,s_inv_total,s_inv_max,diff_truth

def get_errors_pinv_mdls_SGD_TF(X_train,Y_train,X_test,Y_test,degrees,lb,ub,f_target,c_target=None,bias=False):
    train_errors, test_errors = [], []
    ranks = []
    s_inv_total, s_inv_max = [], []
    diff_truth = []
    ##
    N_train,D0 = X_train.shape
    N_test,D_out = Y_test.shape
    ##
    M = 5
    eta = 0.1
    nb_iter = 100*1000
    A = 0
    logging_freq = 100
    #
    print(f'X_train.shape ,Y_train.shape ,X_test.shape ,Y_test.shape = {X_train.shape},{Y_train.shape} ,{X_test.shape},{Y_test.shape}')
    dtype = torch.FloatTensor
    ##
    for degree_mdl in degrees:
        print(f'\ndegree_mdl={degree_mdl}')
        poly_feat = PolynomialFeatures(degree=degree_mdl)
        # get mdl
        Kern_train = poly_feat.fit_transform(X_train)
        Kern_test = poly_feat.fit_transform(X_test)
        print(f'Kern_train.shape={Kern_train.shape},Kern_test={Kern_test.shape}')
        Kern_train_pinv = np.linalg.pinv( Kern_train )
        c_pinv = np.dot(Kern_train_pinv, Y_train)
        train_error_pinv = get_LA_error(X_train,Y_train,c_pinv,poly_feat)
        test_error_pinv = get_LA_error(X_test,Y_test,c_pinv,poly_feat)
        print(f'train_error_pinv={train_error_pinv}, test_error_pinv={test_error_pinv}')
        nb_monomials = int(scipy.misc.comb(D0+degree_mdl,degree_mdl))
        ##
        train_error,test_error = train_tf(nb_monomials, Kern_train,Y_train, Kern_test,Y_test, eta,nb_iter,M)
        ##
        train_errors.append( train_error )
        test_errors.append( test_error )
        print(f'train_error={train_error}, test_error={test_error}')
        ##
        f_pinv =  get_func_pointer_poly(c_pinv,degree_mdl,D0)
        ##
        rank_kern_train = matrix_rank(Kern_train)
        ranks.append(rank_kern_train)
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
    eps_edge = 0.05
    eps_test = eps_train
    eps_test = 0.5
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
        N_test = 100
        N_train = 12
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
        ## get actual (polynomial) target function
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
        #f_target = get_func_pointer_poly(c_target,Degree_data_set,D0)
        #f_target.name = 'poly'
        f_target = f_2_imitate
        f_target.name = 'true_target'
        #Y_train, Y_test = get_target_Y_SP_poly(X_train,X_test, Degree_data_set, c_target, noise_train=noise_train,noise_test=noise_test)
        Y_train, Y_test = f_target(X_train)+noise_train, f_target(X_test)+noise_test
    ## print
    if kwargs['save_data_set']:
        freq_sin=func_params['freq_sin']
        #file_name=''
        if f_target.name == 'poly':
            file_name=f'poly_degree{Degree_data_set}_fit_2_sin_{freq_sin}_N_train_{N_train}_N_test_{N_test}_lb_train,ub_train_{lb,ub}_lb_test,ub_test_{lb_test,ub_test}'
        elif f_target.name == 'true_target':
            file_name=f'f_target_fit_2_sin_{freq_sin}_N_train_{N_train}_N_test_{N_test}_lb_train,ub_train_{lb,ub}_lb_test,ub_test_{lb_test,ub_test}'
            #file_name=f'f_target_fit_2_sin_{freq_sin}_N_train_{N_train}_N_test_{N_test}_lb_train,ub_train_{lb,ub}_lb_test,ub_test_{lb_test,ub_test}_cheby_nodes'
        else:
            raise ValueError(f'Unknown name for target func {f_target.name}')
        path_to_save=f'./data/{file_name}'
        print(f'path_to_save = {path_to_save}')
        experiment_data = dict(
            X_train=X_train,Y_train=Y_train, X_test=X_test,Y_test=Y_test,
            lb=lb,ub=ub,lb_test=lb_test,ub_test=ub_test
        )
        np.savez( path_to_save, **experiment_data)
    #print('c_target = ',c_target)
    #print('c_target.shape = ',c_target.shape)
    #print('nb_monomials_data = {} \n'.format(nb_monomials_data) )
    ## get errors from models
    step_deg=1
    smallest_deg,largest_deg = 1,100
    degrees = list(range(smallest_deg,largest_deg,step_deg))
    train_errors_pinv,test_errors_pinv,ranks,s_inv_total,s_inv_max,diff_truth = get_errors_pinv_mdls(X_train,Y_train,X_test,Y_test,degrees, lb,ub,f_target,c_target)
    #train_errors_pinv,test_errors_pinv,_,_,_,_ = get_errors_pinv_mdls(X_train,Y_train,X_test,Y_test,degrees, lb,ub,f_target,c_target)
    if SGD:
        train_errors,test_errors,ranks,s_inv_total,s_inv_max,diff_truth = get_errors_pinv_mdls_SGD(X_train,Y_train,X_test,Y_test,degrees, lb,ub,f_target, c_target=c_target,bias=False)
    #train_errors,test_errors,ranks,s_inv_total,s_inv_max,diff_truth = get_errors_pinv_mdls_SGD_TF(X_train,Y_train,X_test,Y_test,degrees, lb,ub,f_target, c_target=c_target,bias=False)
    ##
    #print('train_errors = ', train_errors)
    #print('test_errors = ', test_errors)
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
    y_truth = f_target(X_train)
    y_mdl_deg_truth = np.dot(Kern_mdl_truth,c_target_deg_truth)
    print('|| <X_train,c_target> - <X_train,c_target_deg_truth> ||^2 = {}'.format( np.linalg.norm(y_truth-Y_train) ) )
    print( '|| <X_train,c_target> - <X_train,c_target_deg_truth> ||^2 = {}'.format( np.linalg.norm(y_truth-y_mdl_deg_truth) ) )
    if c_target is not None:
        print('||c_truth - c_target_deg_truth||^2 = {}'.format( np.linalg.norm(c_target - c_target_deg_truth) ))
    if plotting:
        if D0==1:
            ## plot target func
            #pdb.set_trace()
            plot_target_function(c_target,X_train,Y_train,lb=lb,ub=ub,f_2_imitate=f_2_imitate)
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
            plot_poly_with_params(c_targets_2_plot[Degree_data_set],X_train,Y_train,lb=lb,ub=ub)

            plot_poly_with_params(c_targets_2_plot[low_mdl],X_train,Y_train,lb=lb,ub=ub)
            plot_poly_with_params(c_targets_2_plot[middle_mdl],X_train,Y_train,lb=lb,ub=ub)
            plot_poly_with_params(c_targets_2_plot[high_mdl],X_train,Y_train,lb=lb,ub=ub)
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
