#!/usr/bin/env python
#SBATCH --mem=7000
#SBATCH --time=0-11:00
#SBATCH --array=60-100
#SBATCH --mail-type=END
#SBATCH --mail-user=brando90@mit.edu
'''
    #SBATCH --gres=gpu:1
'''

import time
from datetime import date
import calendar

import os
import sys

current_directory = os.getcwd() #The method getcwd() returns current working directory of a process.
sys.path.append(current_directory)

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import PolynomialFeatures

import data_utils
import data_regression as data_reg
import data_classification as data_class

import model_logistic_regression as mdl_lreg
import training_algorithms as tr_alg
import hyper_kernel_methods as hkm

import dispatcher_code
import plot_utils

from maps import NamedDict

import pdb
from pdb import set_trace as st

import unittest

import argparse

## python expt_file.py -satid 1 -sj 1
parser = argparse.ArgumentParser(description='Run experiment')
parser.add_argument('-satid', '--satid',type=int,
                    help='satid',default=0)
parser.add_argument('-sj', '--sj',type=int,
                    help='sj',default=0)
args = parser.parse_args()
if args.sj==0 or args.satid==0:
    satid = int(os.environ['SLURM_ARRAY_TASK_ID'])
    sj = int(os.environ['SLURM_JOBID'])
else:
    satid = int(args.satid)
    sj = int(args.sj)

def main(**kwargs):
    ''' setup'''
    np.set_printoptions(suppress=True) #Whether or not suppress printing of small floating point values using scientific notation (default False).
    ##dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
    ''' pytorch dtype setup '''
    dtype_x = torch.FloatTensor
    dtype_y = torch.LongTensor
    dtype_y = torch.FloatTensor
    ''' date parameters setup'''
    today_obj = date.today() # contains datetime.date(year, month, day); accessible via .day etc
    day = today_obj.day
    month = calendar.month_name[today_obj.month]
    ''' Model to train setup param '''
    MDL_2_TRAIN='logistic_regression_vec_mdl'
    #MDL_2_TRAIN='logistic_regression_poly_mdl'
    MDL_2_TRAIN = 'HBF'
    ''' data file names '''
    truth_filename=''
    data_filename=''
    data_filename = 'classification_manual'
    data_filename = 'regression_manual'
    ''' Folder for experiment '''
    experiment_name = 'unit_logistic_regression'
    ##########
    ''' Regularization '''
    ##
    #reg_type = 'VW'
    #reg_type = 'V2W_D3'
    reg_type = ''
    reg = 0
    ''' Experiment LAMBDA experiment params '''
    # expt_type = 'LAMBDAS'
    # N_lambdas = 50
    # lb,ub = 0.01,10000
    # one_over_lambdas = np.linspace(lb,ub,N_lambdas)
    # lambdas = list( 1/one_over_lambdas )
    # lambdas = N_lambdas*[0.0]
    # nb_iterations = [int(1.4*10**6)]
    # repetitions = len(lambdas)*[15]
    ''' Experiment ITERATIONS experiment params '''
    # expt_type = 'ITERATIONS'
    # N_iterations = 30
    # lb,ub = 1,60*10**4
    # lambdas = [0]
    # nb_iterations = [ int(i) for i in np.linspace(lb,ub,N_iterations)]
    # repetitions = len(nb_iterations)*[10]
    ''' Experiment DEGREE/MONOMIALS '''
    # expt_type='DEGREES'
    # step_deg=1
    # lb_deg,ub_deg = 1,100
    # degrees = list(range(lb_deg,ub_deg+1,step_deg))
    # lambdas = [0]
    # nb_iterations = [int(10000)]
    # repetitions = len(degrees)*[1]
    ''' Experiment Number of vector elements'''
    expt_type='NB_VEC_ELEMENTS'
    step=1
    lb_vec,ub_vec = 1,100
    nb_elements_vecs = list(range(lb_vec,ub_vec+1,step))
    lambdas = [0]
    nb_iterations = [int(1000)]
    repetitions = len(nb_elements_vecs)*[1]
    ''' Get setup for process to run '''
    ps_params = NamedDict() # process params
    if expt_type == 'LAMBDAS':
        ps_params.degrees=[]
        ps_params.reg_lambda = get_hp_to_run(hyper_params=lambdas,repetitions=repetitions,satid=satid)
        ps_params.nb_iter = nb_iterations[0]
        ps_params.prefix_experiment = f'it_{nb_iter}/lambda_{reg_lambda}_reg_{reg_type}'
    elif expt_type == 'ITERATIONS':
        ps_params.degrees=[]
        ps_params.reg_lambda = lambdas[0]
        ps_params.nb_iter = get_hp_to_run(hyper_params=nb_iterations,repetitions=repetitions,satid=satid)
        ps_params.prefix_experiment = f'lambda_{reg_lambda}/it_{nb_iter}_reg_{reg_type}'
    elif expt_type == 'DEGREES':
        ps_params.reg_lambda = lambdas[0]
        ps_params.degree_mdl = get_hp_to_run(hyper_params=degrees,repetitions=repetitions,satid=satid)
        ps_params.prefix_experiment = f'fig4_expt_lambda_{reg_lambda}_it_{nb_iter}/deg_{Degree_mdl}'
    elif expt_type == 'NB_VEC_ELEMENTS':
        ps_params.reg_lambda = lambdas[0]
        ps_params.nb_elements_vec = dispatcher_code.get_hp_to_run(hyper_params=nb_elements_vecs,repetitions=repetitions,satid=satid)
        ps_params.nb_iter = nb_iterations[0]
        ps_params.prefix_experiment = f'it_{ps_params.nb_iter}/lambda_{ps_params.reg_lambda}_reg_{reg_type}'
    else:
        raise ValueError(f'Experiment type expt_type={expt_type} does not exist, try a different expt_type.')
    print(f'ps_params={ps_params}')
    ######## data set
    ''' Get data set'''
    if data_filename == 'classification_manual':
        N_train,N_val,N_test = 81,100,121
        lb,ub = -1,1
        f_target = lambda x: np.int64( (np.dot( np.array([1,1]), x) > 0).astype(int) )
        Xtr,Ytr, Xv,Yv, Xt,Yt = data_class.get_2D_classification_data(N_train,N_val,N_test,lb,ub,f_target)
    elif data_filename == 'regression_manual':
        N_train,N_val,N_test = 16,100,121
        lb,ub = -1,1
        f_target = lambda x: np.sin(2*np.pi*4*x)
        Xtr,Ytr, Xv,Yv, Xt,Yt = data_reg.get_2D_regression_data(N_train,N_val,N_test,lb,ub,f_target)
    else:
        data = np.load( './data/{}'.format(data_filename) )
        if 'lb' and 'ub' in data:
            data_lb, data_ub = data['lb'],data['ub']
        else:
            raise ValueError('Error, go to code and fix lb and ub')
    N_train,N_test = Xtr.shape[0], Xt.shape[0]
    print(f'N_train={N_train}, N_test={N_test}')
    ########
    ''' SGD params '''
    optimizer = 'SGD_AND_PERTURB'
    M = int(Xtr.shape[0]/20)
    M = int(2)
    eta = 0.1
    nb_iter = nb_iterations[0]
    A = 0.0
    ##
    logging_freq = 1
    ''' MODEL '''
    if MDL_2_TRAIN == 'logistic_regression_vec_mdl':
        in_features=2
        n_classes=2
        bias=True
        mdl = mdl_lreg.get_logistic_regression_mdl(in_features,n_classes,bias)
        loss = torch.nn.CrossEntropyLoss(size_average=True)
        ##
        loss_collector = lambda mdl,X,Y: tr_alg.calc_loss(mdl,loss,X,Y)
        acc_collector = tr_alg.calc_accuracy
        stats_collector = tr_alg.StatsCollector()
    elif MDL_2_TRAIN =='HBF':
        bias=True
        D_in, D_out = Xtr.shape[0], Ytr.shape[1]
        ## RBF
        std = (Xtr[1] - Xtr[0])/ 4 # less than half the avg distance #TODO use np.mean
        mdl = hkm.OneLayerHBF(D_in,D_out, centers=Xtr,std=std, train_centers=False,train_std=False)
        loss = torch.nn.MSELoss(size_average=True)
        ''' stats collector '''
        loss_collector = lambda mdl,X,Y: tr_alg.calc_loss(mdl,loss,X,Y)
        acc_collector = loss_collector
        ''' dynamic stats collector '''
        c_pinv = hkm.get_rbf_coefficients(X=Xtr,centers=Xtr,Y=Ytr,std=std)
        def diff_GD_vs_PINV(storer, i, mdl, Xtr,Ytr,Xv,Yv,Xt,Yt):
            c_pinv_torch = torch.FloatTensor( c_pinv )
            diff_GD_pinv = (mdl.C.weight.data.t() - c_pinv_torch).norm(2)
            storer.append(diff_GD_pinv)
        dynamic_stats = NamedDict(diff_GD_vs_PINV=([],diff_GD_vs_PINV))
        ##
        stats_collector = tr_alg.StatsCollector(mdl, loss_collector,acc_collector,dynamic_stats=dynamic_stats)
    else:
        raise ValueError(f'MDL_2_TRAIN={MDL_2_TRAIN}')
    ''' TRAIN '''
    #train_args = NamedDict({})
    if optimizer =='SGD_AND_PERTURB':
        perturb_freq = 1000
        perturb_magnitude = 0
        ##
        momentum = 0
        optim = torch.optim.SGD(mdl.parameters(), lr=eta, momentum=momentum)
        tr_alg.SGD_perturb(mdl, Xtr,Ytr,Xv,Yv,Xt,Yt, optim,loss, M,eta,nb_iter,A ,logging_freq,
            dtype_x,dtype_y, perturb_freq,perturb_magnitude,
            reg=reg,reg_lambda=ps_params.reg_lambda,
            stats_collector=stats_collector)
    else:
        raise ValueError(f'MDL_2_TRAIN={MDL_2_TRAIN} not implemented')
    ''' Plots and Print statements'''
    print('\n----')
    print(f'some SGD params: batch_size={M}, eta={eta}, nb_iterations={nb_iter}')
    if MDL_2_TRAIN=='HBF':
        ''' print statements R/HBF'''
        print(f'distance_btw_data_points={Xtr[1] - Xtr[0]}')
        print(f'std={std}')
        print(f'less than half the average distance?={(std < (Xtr[1] - Xtr[0])/2)}')
        ''' plots for R/HBF'''
        f_mdl = lambda x: mdl( Variable(torch.FloatTensor(x),requires_grad=False) ).data.numpy()
        f_pinv = lambda x: hkm.f_rbf(x,c=c_pinv,centers=Xtr,std=std)
        f_target = f_target
        iterations = np.array(range(0,nb_iter))
        N_denseness = 1000
        ## plots
        plot_utils.plot_loss_errors(iterations,stats_collector,test_error_pinv=data_utils.l2_np_loss(f_pinv(Xt),Yt))
        plot_utils.visualize_1D_reconstruction(lb,ub,N_denseness, f_mdl,f_target=f_target,f_pinv=f_pinv,X=Xtr,Y=Ytr,legend_data_set='Training data points')
        # plot_utils.plot_sgd_vs_pinv_distance_during_training(iterations,stats_collector)
        # plot_utils.print_gd_vs_pinv_params(mdl,c_pinv)
        plt.show()

if __name__ == '__main__':
    start_time = time.time()
    #main(save_bulk_experiment=True,plotting=True)
    main()
