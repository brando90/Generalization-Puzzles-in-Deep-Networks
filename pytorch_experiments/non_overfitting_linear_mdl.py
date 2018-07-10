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
import save_to_matlab_format as save2matlab

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
parser.add_argument('-debug','--debug',dest='debug',action='store_true')
args = parser.parse_args()
if args.sj==0 or args.satid==0:
    satid = int(os.environ['SLURM_ARRAY_TASK_ID'])
    sj = int(os.environ['SLURM_JOBID'])
else:
    satid = int(args.satid)
    sj = int(args.sj)
debug = '_debug' if args.debug else ''

def get_models():
    '''
    '''

def main():
    expt_type = 'polynomial'
    data_filename = 'classification_manual'
    ######## data set
    ''' Get data set'''
    if data_filename == 'classification_manual':
        N_train,N_val,N_test = 81,100,500
        lb,ub = -1,1
        w_target = np.array([1,1])
        f_target = lambda x: np.int64( (np.dot(w_target,x) > 0).astype(int) )
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
    ''' Loop through the linear model'''
    if expt_type == 'hermite_poly':
        results = get_hermite_poly()

if __name__ == '__main__':
    start_time = time.time()
    #main(save_bulk_experiment=True,plotting=True)
    main()
