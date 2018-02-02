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

current_directory = ps.getcwd() #The method getcwd() returns current working directory of a process.
sys.path.append(current_directory)

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import PolynomialFeatures

from maps import NamedDict

import pdb

import unittest

SLURM_ARRAY_TASK_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
SLURM_JOBID = int(os.environ['SLURM_JOBID'])

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
    ''' organize parameters from dispatcher params '''
    MDL_2_TRAIN='logistic_regression_vec_mdl'
    #MDL_2_TRAIN='logistic_regression_poly_mdl'
    ''' data file names '''
    truth_filename=''
    data_filename=''
    data_filename = 'classification_manual'
    ''' Folder for experiment '''
    experiment_name = 'unit_logistic_regression'
    ''' Experiment Type '''
    expt_type='SP_fig4'
    ##########
    ''' Regularization '''
    ##
    #reg_type = 'VW'
    #reg_type = 'V2W_D3'
    reg_type = ''
    ''' Experiment LAMBDA experiment params '''
    ## LAMBDAS
    # expt_type = 'LAMBDAS'
    # N_lambdas = 50
    # lb,ub = 0.01,10000
    # one_over_lambdas = np.linspace(lb,ub,N_lambdas)
    # lambdas = list( 1/one_over_lambdas )
    # lambdas = N_lambdas*[0.0]
    # nb_iterations = [int(1.4*10**6)]
    # repetitions = len(lambdas)*[15]
    ''' Experiment ITERATIONS experiment params '''
    ## ITERATIONS
    # expt_type = 'ITERATIONS'
    # N_iterations = 30
    # lb,ub = 1,60*10**4
    # lambdas = [0]
    # nb_iterations = [ int(i) for i in np.linspace(lb,ub,N_iterations)]
    # repetitions = len(nb_iterations)*[10]
    ''' DEGREE/MONOMIALS '''
    ## SP DEGREE/MONOMIALS
    step_deg=1
    lb_deg,ub_deg = 1,100
    degrees = list(range(lb_deg,ub_deg+1,step_deg))
    lambdas = [0]
    nb_iterations = [int(10000)]
    repetitions = len(degrees)*[1]
    ######## data set
    ''' Get data set'''
    if data_filename == 'classification_manual':
        X_train,Y_train, X_test,Y_test = get_2D_classification_data()
    else:
        data = np.load( './data/{}'.format(data_filename) )
        if 'lb' and 'ub' in data:
            data_lb, data_ub = data['lb'],data['ub']
        else:
            raise ValueError('Error, go to code and fix lb and ub')
    N_train,N_test = X_train.shape[0], X_test.shape[0]
    print(f'N_train={N_train}, N_test={N_test}')
    ########
    ''' SGD params '''
    M = 11
    eta = 0.2
    A = 0.0
    momentum = 0
    ''' MODEL '''
    if MDL_2_TRAIN = 'logistic_regression_vec_mdl':
            loss = torch.nn.CrossEntropyLoss(size_average=True) #TODO fix
            optimizer = torch.optim.SGD(mdl_sgd.parameters(), lr=eta, momentum=0.98) #TODO fix
    else:
        raise ValueError(f'MDL_2_TRAIN={MDL_2_TRAIN}')
    ''' '''
    return

if __name__ == '__main__':
    start_time = time.time()
    #main(save_bulk_experiment=True,plotting=True)
    main()
