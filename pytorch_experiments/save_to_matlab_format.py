import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io

import utils

from maps import NamedDict

import pdb
from pdb import set_trace as st

def save_experiment_results_2_matlab(experiment_results, root_path,experiment_name,training_config_name,main_experiment_params,expt_type,matlab_file_name):
    '''
        Format for saving directories:
            {root_path}/{experiment_name}/{training_config_name}/{main_experiment_params}/{expt_1...i...N}
        1) root_path= e.g. {test_runs}
        2) experiment_name=name of the experiment e.g. {unit_logistic_regression}
        3) training_config_name=configuration for training e.g. e.g {const_noise_pert_reg_N_train_9_M_9_frac_norm_0}
        4) main_experiment_params=parameters for the set of experiments e.g. {LAMBDAS_lambda_0_it_10000}
        5) expt_type1...i...N=the configuration for the experiment e.g {LAMBDAS_lambda_val}
    '''
    ##
    ''' 1) e.g. .test_runs '''
    root_path = f'{root_path}'
    ''' 2) e.g. unit_logistic_regression'''
    experiment_name = f'{experiment_name}'
    ''' 3) e.g. const_noise_pert_reg_N_train_9_M_9_frac_norm_0 '''
    training_config_name = f'{training_config_name}' # '_reg_{reg_type}_expt_type_{expt_type}_N_train_{N_train}_M_{M}'
    ''' 4) e.g. lambdas_lambda_0_it_10000 '''
    main_experiment_params = f'{main_experiment_params}'
    ''' 5) e.g. LAMBDAS '''
    expt_type = f'{expt_type}'
    ''' asseble path to save AND check if you need to make it'''
    path_to_save = f'{root_path}/{experiment_name}/{training_config_name}/{expt_type}/'
    utils.make_and_check_dir(path_to_save)
    ''' save data '''
    io.savemat(f'{path_to_save}/{matlab_file_name}',experiment_results)

def save2matlab(path_to_save,stats_collector,other_stats):
    stats = stats_collector.get_stats_dict()
    experiment_results = NamedDict(stats,**other_stats)
    ##
    scipy.io.savemat(path_to_save,experiment_results)

def save2matlab_flatness_expt(path_to_filename, stats_collector, other_stats={}):
    '''
    Saves the current results from flatnes¯s experiment.

    results_root = location of main folder where results are. e.g. './test_runs_flatness'
    expt_path = path
    '''
    stats = stats_collector.get_stats_dict()
    experiment_results = NamedDict(stats,**other_stats)
    ##
    scipy.io.savemat(path_to_filename,experiment_results)
