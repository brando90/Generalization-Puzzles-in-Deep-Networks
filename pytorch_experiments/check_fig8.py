#!/usr/bin/env python
#SBATCH --mem=90000
#SBATCH --time=0-20:00
#SBATCH --array=1-30
#SBATCH --mail-type=END
#SBATCH --mail-user=brando90@mit.edu
''''
#SBATCH --gres=gpu:1
'''

import time
from datetime import date
import calendar

#from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.pipeline import make_pipeline

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

import utils

import data_utils
import data_regression as data_reg
import data_classification as data_class

import model_logistic_regression as mdl_lreg
import training_algorithms as tr_alg
import hyper_kernel_methods as hkm
import save_to_matlab_format as save2matlab

import dispatcher_code
import plot_utils

from maps import NamedDict

from metrics import calc_accuracy
from metrics import calc_error
from metrics import calc_loss

import pdb
from pdb import set_trace as st

import unittest

import argparse

## python expt_file.py -satid 1 -sj 1
## sbatch expt_file.py
parser = argparse.ArgumentParser(description='Run experiment')
parser.add_argument('-satid', '--satid',type=int,
                    help='satid',default=0)
parser.add_argument('-sj', '--sj',type=int,
                    help='sj',default=0)
parser.add_argument('-debug','--debug',dest='debug',action='store_true')
args = parser.parse_args()
if args.sj==0 or args.satid==0:
    satid = int(os.environ['SLURM_ARRAY_TASK_ID'])
    sj = int(os.environ['SLURM_JOBID'])
else:
    satid = int(args.satid)
    sj = int(args.sj)
debug = '_debug' if args.debug else ''

def main(plotting=False,save=False):
    ''' setup'''
    start_time = time.time()
    np.set_printoptions(suppress=True) #Whether or not suppress printing of small floating point values using scientific notation (default False).
    ##dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
    ''' pytorch dtype setup '''
    # dtype_y = torch.LongTensor
    dtype_x = torch.FloatTensor
    dtype_y = torch.FloatTensor
    # dtype_x = torch.cuda.FloatTensor
    # dtype_y = torch.cuda.FloatTensor
    ''' date parameters setup'''
    today_obj = date.today() # contains datetime.date(year, month, day); accessible via .day etc
    day = today_obj.day
    month = calendar.month_name[today_obj.month]
    ''' Model to train setup param '''
    #MDL_2_TRAIN='logistic_regression_vec_mdl'
    #MDL_2_TRAIN='logistic_regression_poly_mdl'
    MDL_2_TRAIN = 'regression_poly_mdl'
    #MDL_2_TRAIN = 'HBF'
    ''' data file names '''
    truth_filename=''
    data_filename=''
    #data_filename = 'classification_manual'
    data_filename = 'regression_manual'
    ''' Folder for experiment '''
    experiment_name = 'RedoFig5_Cheby'
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
    expt_type='DEGREES'
    step_deg=1
    lb_deg,ub_deg = 39,39
    degrees = list(range(lb_deg,ub_deg+1,step_deg))
    lambdas = [0]
    #nb_iterations = [int(2500000)]
    #nb_iterations = [int(1000000)]
    #nb_iterations = [int(5 * 10**6)]
    #nb_iterations = [int(1.1 * 10 ** 7)]
    repetitions = len(degrees)*[30]
    ''' Experiment Number of vector elements'''
    expt_type='NB_VEC_ELEMENTS'
    step=1
    lb_vec,ub_vec = 30,30
    nb_elements_vecs = list(range(lb_vec,ub_vec+1,step))
    lambdas = [0]
    nb_iterations = [int(250000)]
    #nb_iterations = [int(2500)]
    repetitions = len(nb_elements_vecs)*[1]
    ''' Get setup for process to run '''
    ps_params = NamedDict() # process params
    if expt_type == 'LAMBDAS':
        ps_params.degrees=[]
        ps_params.reg_lambda = dispatcher_code.get_hp_to_run(hyper_params=lambdas,repetitions=repetitions,satid=satid)
        ps_params.nb_iter = nb_iterations[0]
        #ps_params.prefix_experiment = f'it_{nb_iter}/lambda_{reg_lambda}_reg_{reg_type}'
    elif expt_type == 'ITERATIONS':
        ps_params.degrees=[]
        ps_params.reg_lambda = lambdas[0]
        ps_params.nb_iter = dispatcher_code.get_hp_to_run(hyper_params=nb_iterations,repetitions=repetitions,satid=satid)
        #ps_params.prefix_experiment = f'lambda_{reg_lambda}/it_{nb_iter}_reg_{reg_type}'
    elif expt_type == 'DEGREES':
        ps_params.reg_lambda = lambdas[0]
        ps_params.degree_mdl = dispatcher_code.get_hp_to_run(hyper_params=degrees,repetitions=repetitions,satid=satid)
        #ps_params.prefix_experiment = f'fig4_expt_lambda_{reg_lambda}_it_{nb_iter}/deg_{Degree_mdl}'
        hp_param = ps_params.degree_mdl
    elif expt_type == 'NB_VEC_ELEMENTS':
        ps_params.reg_lambda = lambdas[0]
        ps_params.nb_elements_vec = dispatcher_code.get_hp_to_run(hyper_params=nb_elements_vecs,repetitions=repetitions,satid=satid)
        ps_params.nb_iter = nb_iterations[0]
        #ps_params.prefix_experiment = f'it_{ps_params.nb_iter}/lambda_{ps_params.reg_lambda}_reg_{reg_type}'
    else:
        raise ValueError(f'Experiment type expt_type={expt_type} does not exist, try a different expt_type.')
    print(f'ps_params={ps_params}')
    ######## data set
    ''' Get data set'''
    if data_filename == 'classification_manual':
        N_train,N_val,N_test = 81,100,500
        lb,ub = -1,1
        w_target = np.array([1,1])
        f_target = lambda x: np.int64( (np.dot(w_target,x) > 0).astype(int) )
        Xtr,Ytr, Xv,Yv, Xt,Yt = data_class.get_2D_classification_data(N_train,N_val,N_test,lb,ub,f_target)
    elif data_filename == 'regression_manual':
        N_train,N_val,N_test = 9,81,100
        lb,ub = -1,1
        f_target = lambda x: np.sin(2*np.pi*4*x)
        Xtr,Ytr, Xv,Yv, Xt,Yt = data_reg.get_2D_regression_data_chebyshev_nodes(N_train,N_val,N_test,lb,ub,f_target)
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
    #optimizer_mode = 'SGD_AND_PERTURB'
    optimizer_mode = 'SGD_train_then_pert'
    M = int(Xtr.shape[0])
    #M = int(81)
    eta = 0.2
    momentum = 0.0
    nb_iter = nb_iterations[0]
    A = 0.0
    ##
    logging_freq = 1
    ''' MODEL '''
    if MDL_2_TRAIN=='logistic_regression_vec_mdl':
        in_features=31
        n_classes=1
        bias=False
        mdl = mdl_lreg.get_logistic_regression_mdl(in_features,n_classes,bias)
        loss = torch.nn.CrossEntropyLoss(size_average=True)
        ''' stats collector '''
        loss_collector = lambda mdl,X,Y: calc_loss(mdl,loss,X,Y)
        acc_collector = calc_accuracy
        acc_collector = calc_error
        stats_collector = tr_alg.StatsCollector(mdl, loss_collector,acc_collector)
        ''' make features for data '''
        poly = PolynomialFeatures(in_features-1)
        Xtr,Xv,Xt = poly.fit_transform(Xtr), poly.fit_transform(Xv), poly.fit_transform(Xt)
    elif MDL_2_TRAIN == 'regression_poly_mdl':
        in_features = degrees[0]+1
        mdl = mdl_lreg.get_logistic_regression_mdl(in_features, 1, bias=False)
        loss = torch.nn.MSELoss(size_average=True)
        ''' stats collector '''
        loss_collector = lambda mdl, X, Y: calc_loss(mdl, loss, X, Y)
        acc_collector = loss_collector
        acc_collector = loss_collector
        stats_collector = tr_alg.StatsCollector(mdl, loss_collector, acc_collector)
        ''' make features for data '''
        poly = PolynomialFeatures(in_features - 1)
        Xtr, Xv, Xt = poly.fit_transform(Xtr), poly.fit_transform(Xv), poly.fit_transform(Xt)
    elif MDL_2_TRAIN=='HBF':
        bias=True
        D_in, D_out = Xtr.shape[0], Ytr.shape[1]
        ## RBF
        std = (Xtr[1] - Xtr[0])/ 0.8 # less than half the avg distance #TODO use np.mean
        centers=Xtr
        mdl = hkm.OneLayerHBF(D_in,D_out, centers=centers,std=std, train_centers=False,train_std=False)
        mdl[0].weight.data.fill_(0)
        mdl[0].bias.data.fill_(0)
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
    perturbfreq = 1.1 * 10**5
    perturb_magnitude = 0.45
    if optimizer_mode =='SGD_AND_PERTURB':
        ##
        momentum = 0.0
        optim = torch.optim.SGD(mdl.parameters(), lr=eta, momentum=momentum)
        ##
        reg_lambda = ps_params.reg_lambda
        tr_alg.SGD_perturb(mdl, Xtr,Ytr,Xv,Yv,Xt,Yt, optim,loss, M,eta,nb_iter,A ,logging_freq,
            dtype_x,dtype_y, perturbfreq,perturb_magnitude,
            reg=reg,reg_lambda=reg_lambda,
            stats_collector=stats_collector)
    elif optimizer_mode == 'SGD_train_then_pert':
        iterations_switch_mode = 1 # never perturb
        #iterations_switch_mode = nb_iter # always perturb
        iterations_switch_mode = nb_iter/2 # perturb for half
        print(f'iterations_switch_mode={iterations_switch_mode}')
        ##
        optimizer = torch.optim.SGD(mdl.parameters(), lr=eta, momentum=momentum)
        ##
        reg_lambda = ps_params.reg_lambda
        tr_alg.SGD_pert_then_train(mdl, Xtr,Ytr,Xv,Yv,Xt,Yt, optimizer,loss, M,nb_iter ,logging_freq ,dtype_x,dtype_y,
                                   perturbfreq,perturb_magnitude, iterations_switch_mode, reg,reg_lambda, stats_collector)
    else:
        raise ValueError(f'MDL_2_TRAIN={MDL_2_TRAIN} not implemented')
    seconds,minutes,hours = utils.report_times(start_time)
    ''' Plots and Print statements'''
    print('\n----\a\a')
    print(f'some SGD params: batch_size={M}, eta={eta}, nb_iterations={nb_iter}')
    if save:
        ''' save experiment results to maltab '''
        experiment_results=stats_collector.get_stats_dict()
        experiment_results=NamedDict(seconds=seconds,minutes=minutes,hours=hours,**experiment_results)
        save2matlab.save_experiment_results_2_matlab(experiment_results=experiment_results,
            root_path=f'./test_runs_flatness3',
            experiment_name=experiment_name,
            training_config_name=f'nb_iterations_{nb_iterations[0]}_N_train_{Xtr.shape[0]}_N_test_{Xt.shape[0]}_batch_size_{M}_perturb_freq_{perturbfreq}_perturb_magnitude_{perturb_magnitude}_momentum_{momentum}_iterations_switch_mode_{iterations_switch_mode}',
            main_experiment_params=f'{expt_type}_lambda_{ps_params.reg_lambda}_it_{nb_iter}_reg_{reg_type}',
            expt_type=f'expt_type_{expt_type}_{hp_param}',
            matlab_file_name=f'satid_{satid}_sid_{sj}_{month}_{day}'
            )
    if MDL_2_TRAIN=='HBF':
        ''' print statements R/HBF'''
        print(f'distance_btw_data_points={Xtr[1] - Xtr[0]}')
        print(f'std={std}')
        print(f'less than half the average distance?={(std < (Xtr[1] - Xtr[0])/2)}')
        beta = (1.0/std)**2
        rank = np.linalg.matrix_rank( np.exp( -beta*hkm.euclidean_distances_manual(x=Xtr,W=centers.T) ) )
        print(f'rank of Kernel matrix = Rank(K) = {rank}')
        ''' plots for R/HBF'''
        f_mdl = lambda x: mdl( Variable(torch.FloatTensor(x),requires_grad=False) ).data.numpy()
        f_pinv = lambda x: hkm.f_rbf(x,c=c_pinv,centers=Xtr,std=std)
        f_target = f_target
        iterations = np.array(range(0,nb_iter))
        N_denseness = 1000
        legend_hyper_params=f'N_train={Xtr.shape[0]},N_test={Xt.shape[0]},batch-size={M},learning step={eta},# iterations = {nb_iter} momentum={momentum}, Model=Gaussian, # centers={centers.shape[0]}, std={std[0]}'
        ''' PLOT '''
        ## plots
        plot_utils.plot_loss_errors(iterations,stats_collector,test_error_pinv=data_utils.l2_np_loss(f_pinv(Xt),Yt),legend_hyper_params=legend_hyper_params)
        plot_utils.visualize_1D_reconstruction(lb,ub,N_denseness, f_mdl,f_target=f_target,f_pinv=f_pinv,X=Xtr,Y=Ytr,legend_data_set='Training data points')
        plot_utils.plot_sgd_vs_pinv_distance_during_training(iterations,stats_collector)
        #plot_utils.print_gd_vs_pinv_params(mdl,c_pinv)
        plt.show()
    elif MDL_2_TRAIN=='logistic_regression_vec_mdl':
        ''' arguments for plotting things '''
        f_mdl = lambda x: mdl( Variable(torch.FloatTensor(x),requires_grad=False) ).data.numpy()
        f_target = lambda x: -1*(w_target[0]/w_target[1])*x
        iterations = np.array(range(0,nb_iter))
        N_denseness = 1000
        legend_hyper_params=f'N_train={Xtr.shape[0]},N_test={Xt.shape[0]},batch-size={M},learning step={eta},# iterations = {nb_iter} momentum={momentum}, Model=Logistic Regression'
        ''' PLOT '''
        ## plots
        plot_utils.plot_loss_errors(iterations,stats_collector,legend_hyper_params=legend_hyper_params)
        plot_utils.plot_loss_classification_errors(iterations,stats_collector,legend_hyper_params=legend_hyper_params)
        plot_utils.visualize_classification_data_learned_planes_2D(lb,ub,N_denseness,Xtr,Ytr,f_mdl,f_target)
        plot_utils.plot_weight_norm_vs_iterations(iterations,stats_collector.w_norms[0])
        plt.show()
    if plotting:
        legend_hyper_params = f'N_train={Xtr.shape[0]},N_test={Xt.shape[0]},batch-size={M},learning step={eta},# iterations = {nb_iter} momentum={momentum}, Model=Regression'
        iterations = np.array(range(0, nb_iter))
        plot_utils.plot_loss_errors(iterations, stats_collector, legend_hyper_params=legend_hyper_params)
        plot_utils.plot_weight_norm_vs_iterations(iterations, stats_collector.w_norms[0])
        plt.show()


if __name__ == '__main__':
    main(plotting=False,save=True)
