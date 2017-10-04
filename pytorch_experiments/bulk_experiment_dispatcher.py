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


def serial_multiple_lambdas(**kwargs):
    lambdas, repetitions = kwargs['lambdas'], kwargs['repetitions']
    nb_lambdas = lambdas.shape[0]
    ##
    train_errors = np.zeros( (nb_lambdas,repetitions) ) # row lambda, trial of experiment
    test_errors = np.zeros( (nb_lambdas,repetitions) ) # row lambda, trial of experiment
    erm_regs = np.zeros( (nb_lambdas,repetitions) ) # row lambda, trial of experiment
    ## collect the errors for every lambda
    for lambda_index in range(nb_lambdas):
        reg_lambda = lambdas[lambda_index]
        for current_repetition in range(repetitions):
            train_error, test_error, erm_reg = main(experiment_type='quick_run',reg_lambda_WP=reg_lambda,plotting=False)
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
        test_means[lambda_index] = np.mean( test_errors[reg_lambda,:] )
        test_stds[lambda_index] = np.std( test_errors[reg_lambda,:] )
    ##
    scipy.io.savemat( 'test.mat', dict(train_means=train_means,train_stds=train_stds, test_means=test_means,test_stds=test_stds) )

if __name__ == '__main__':
    ## real experiment
    #lambdas = np.linspace(20,200,num=5)
    #repetitions=5
    ## unit tests
    lambdas = np.linspace(20,200,num=3)
    repetitions=3
    ##
    serial_multiple_lambdas(lambdas=lambdas,repetitions=repetitions)
