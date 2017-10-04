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
    train_errors = { lambda_key: [] for lambda_key in kwargs['lambdas']} # maps lambda to train errors
    test_errors = { lambda_key: [] for lambda_key in kwargs['lambdas']} # maps lambda to train errors
    erm_regs = { lambda_key: [] for lambda_key in kwargs['lambdas']} # maps lambda to train errors
    ## collect the errors for every lambda
    for reg_lambda in kwargs['lambdas']:
        #
        for current_repetition for repetitions:
            train_error, test_error, erm_reg = main(experiment_type='quick_run',reg_lambda_WP=reg_lambda_WP)
            ##
            train_errors[reg_lambda].append( train_error )
            test_errors[reg_lambda].append( test_error )
            erm_regs[reg_lambda].append( erm_regs )
    ##
    train_means = { lambda_key: 0 for lambda_key in kwargs['lambdas']}
    train_stds = { lambda_key: 0 for lambda_key in kwargs['lambdas']}
    test_means = { lambda_key: 0 for lambda_key in kwargs['lambdas']}
    test_stds = { lambda_key: 0 for lambda_key in kwargs['lambdas']}
    ## collect the statistics for every lambda
    for reg_lambda in kwargs['lambdas']:
        train_means[reg_lambda] = np.mean( train_errors[reg_lambda] )
        train_stds[reg_lambda] = np.std( train_errors[reg_lambda] )
        #
        test_means[reg_lambda] = np.mean( test_errors[reg_lambda] )
        test_stds[reg_lambda] = np.std( test_errors[reg_lambda] )
    ##
    reg_lamdas = np.array( kwargs.keys() )
    scipy.io.savemat('test.mat', dict(x=x, y=y))



if __name__ == '__main__':
    ## real experiment
    #lambdas = linspace(20,200,num=5)
    #repetitions=5
    ## unit tests
    lambdas = linspace(20,200,num=3)
    repetitions=3
    ##
    serial_multiple_lambdas(lambdas=lambdas,repetitions=repetitions)
