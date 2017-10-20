#!/usr/bin/env python
#SBATCH --mem=7000
#SBATCH --time=7-00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=brando90@mit.edu
'''
#SBATCH --array=1-200
#SBATCH --gres=gpu:1

'''

import time
import numpy as np
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
from pytorch_over_approx_high_dim import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy
import numpy as np
import scipy.io

import argparse

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-expt_type','--experiment_type',type=str, help='save the result of the experiment')
    parser.add_argument('-lb','--lower_bound',type=int, help='lower bound')
    parser.add_argument('-ub','--upper_bound',type=int, help='upper bound')
    parser.add_argument('-num','--number_values',type=int, help='number of values in between lb and ub')
    parser.add_argument('-num_rep','--number_repetitions',type=int, help='number of repetitions per run')
    parser.add_argument('-save','--save_bulk_experiment',type=bool, help='save the result of the experiment')
    parser.add_argument('-sj', '--SLURM_JOBID', help='SLURM_JOBID for run')
    parser.add_argument('-rt_wp', '--reg_type_wp', type=str, default='tikhonov', help='Regularization Type for WP. e.g: VM, tikhonov, V[^2W, etc')
    cmd_args = parser.parse_args()
    return cmd_args

cmd_args = get_argument_parser()
SLURM_JOBID = cmd_args.SLURM_JOBID

##

def serial_multiple_lambdas(**kwargs):
    lambdas, repetitions = kwargs['lambdas'], kwargs['repetitions']
    reg_type_wp = kwargs['reg_type_wp']
    one_over_lambdas = 1/lambdas
    nb_lambdas = lambdas.shape[0]
    ##
    train_errors = np.zeros( (nb_lambdas,repetitions) ) # row lambda, trial of experiment
    test_errors = np.zeros( (nb_lambdas,repetitions) ) # row lambda, trial of experiment
    erm_regs = np.zeros( (nb_lambdas,repetitions) ) # row lambda, trial of experiment
    ## collect the errors for every lambda
    for lambda_index in range(nb_lambdas):
        reg_lambda = lambdas[lambda_index]
        print('reg_lambda = ', reg_lambda)
        for current_repetition in range(repetitions):
            train_error, test_error, erm_reg = main(experiment_type='serial_multiple_lambdas',reg_lambda_WP=reg_lambda,reg_type_wp=reg_type_wp,plotting=False)
            ##
            train_errors[lambda_index,current_repetition] = train_error
            test_errors[lambda_index,current_repetition] = test_error
            erm_regs[lambda_index,current_repetition] = erm_reg
    ##
    train_means, train_stds = np.zeros( nb_lambdas ), np.zeros( nb_lambdas ) # one moment per lambda
    test_means, test_stds = np.zeros( nb_lambdas ), np.zeros( nb_lambdas ) # one moment per lambda
    ## collect the statistics for every lambda
    for lambda_index in range(nb_lambdas):
        train_means[lambda_index] = np.mean( train_errors[lambda_index,:] )
        train_stds[lambda_index] = np.std( train_errors[lambda_index,:] )
        #
        test_means[lambda_index] = np.mean( test_errors[lambda_index,:] )
        test_stds[lambda_index] = np.std( test_errors[lambda_index,:] )
    ##
    if kwargs['save_bulk_experiment']:
        path_to_save = '../plotting/results/experiment_lambdas_oct7_{}.mat'.format(SLURM_JOBID)
        scipy.io.savemat( path_to_save, dict(lambdas=lambdas,one_over_lambdas=one_over_lambdas, train_means=train_means,train_stds=train_stds, test_means=test_means,test_stds=test_stds) )

def serial_multiple_iterations(**kwargs):
    print('serial_multiple_iterations')
    iterations, repetitions = kwargs['iterations'], kwargs['repetitions']
    nb_iterations = iterations.shape[0]
    ##
    train_errors = np.zeros( (nb_iterations,repetitions) ) # row lambda, trial of experiment
    test_errors = np.zeros( (nb_iterations,repetitions) ) # row lambda, trial of experiment
    erm_regs = np.zeros( (nb_iterations,repetitions) ) # row lambda, trial of experiment
    ## collect the errors for every lambda
    for iter_index in range(nb_iterations):
        current_nb_repetitions = iterations[iter_index]
        print('current_nb_repetitions = ', current_nb_repetitions)
        for current_repetition in range(repetitions):
            train_error, test_error, erm_reg = main(experiment_type='serial_multiple_lambdas',nb_iterations_WP=current_nb_repetitions,plotting=False)
            ##
            train_errors[iter_index,current_repetition] = train_error
            test_errors[iter_index,current_repetition] = test_error
            erm_regs[iter_index,current_repetition] = erm_reg
    ##
    train_means, train_stds = np.zeros( nb_iterations ), np.zeros( nb_iterations ) # one moment per lambda
    test_means, test_stds = np.zeros( nb_iterations ), np.zeros( nb_iterations ) # one moment per lambda
    ## collect the statistics for every lambda
    for iter_index in range(nb_iterations):
        train_means[iter_index] = np.mean( train_errors[iter_index,:] )
        train_stds[iter_index] = np.std( train_errors[iter_index,:] )
        #
        test_means[iter_index] = np.mean( test_errors[iter_index,:] )
        test_stds[iter_index] = np.std( test_errors[iter_index,:] )
    ##
    if kwargs['save_bulk_experiment']:
        print('save_bulk_experiment')
        path_to_save = '../plotting/results/experiment_iter_oct7_{}.mat'.format(SLURM_JOBID)
        scipy.io.savemat( path_to_save, dict(iterations=iterations, train_means=train_means,train_stds=train_stds, test_means=test_means,test_stds=test_stds) )

##

def main_lambda(lb, ub, num, num_rep, save, reg_type_wp):
    ## real experiment
    #lambdas = np.linspace(20,200,num=5)
    #repetitions=5
    ## unit tests
    one_over_lambdas = np.linspace(lb,ub,num=num)
    lambdas = 1/one_over_lambdas
    repetitions=num_rep
    ##
    save_bulk_experiment = save
    serial_multiple_lambdas(lambdas=lambdas,repetitions=repetitions,save_bulk_experiment=save_bulk_experiment,reg_type_wp=reg_type_wp)

def main_iterations(lb, ub, num, num_rep, save):
    ## real experiment
    #lambdas = np.linspace(20,200,num=5)
    #repetitions=5
    ## unit tests
    iterations = np.linspace(lb,ub,num=num)
    iterations = np.array( [ int(iteration) for iteration in iterations ] )
    repetitions=num_rep
    ##
    save_bulk_experiment = save
    serial_multiple_iterations(iterations=iterations,repetitions=repetitions,save_bulk_experiment=save_bulk_experiment)

if __name__ == '__main__':
    print(cmd_args)
    experiment_type, lb,ub,num,num_rep,save = cmd_args.experiment_type, cmd_args.lower_bound, cmd_args.upper_bound, cmd_args.number_values, cmd_args.number_repetitions, cmd_args.save_bulk_experiment
    reg_type_wp = cmd_args.reg_type_wp
    ##
    start_time = time.time()
    ##
    if experiment_type == 'lambda':
        main_lambda(lb,ub,num,num_rep,save,reg_type_wp)
    elif experiment_type == 'iterations':
        main_iterations(lb,ub,num,num_rep,save)
    else:
        raise ValueError('The flag value for expt_type {} is invalid, give a valid experiment.')
    ## REPORT TIMES
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    print('\a')
    print("--- %s seconds ---" % seconds )
    print("--- %s minutes ---" % minutes )
    print("--- %s hours ---" % hours )
