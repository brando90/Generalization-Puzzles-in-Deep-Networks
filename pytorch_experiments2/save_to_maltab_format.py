import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import scipy

from maps import NamedDict

import pdb
from pdb import set_trace as st

def get_path_2_save(root_path,experiment_name,training_config_name):
    '''
        Format for saving directories:
            root_path/{experiment_name}/{training_config_name}/{experiment_params}/expt_{1...i...N}
        root_path=test_runs
        experiment_name=name of the experiment e.g. unit_logistic_regression
        training_config=configuration for training
    '''
    ''' set root path, e.g. .test_runs '''
    root_path = f'{root_path}'
    ''' set experiment name e.g. unit_logistic_regression'''
    experiment_name = f'{experiment_name}'
    ''' set training_config_name'''
    training_config_name = f'{}'
    '_reg_{reg_type}_expt_type_{expt_type}_N_train_{N_train}_M_{M}'
    path_to_save=f'{path_to_save}/{prefix_experiment}'
    ##
    make_and_check_dir(path_to_save)
    path_to_save = f'{path_to_save}/satid_{SLURM_ARRAY_TASK_ID}_sid_{SLURM_JOBID}_{month}_{day}'


def save2matlab(path_to_save,stats_collector,other_stats):
    stats = stats_collector.get_stats_dict()
    experiment_results = NamedDict(stats,**other_stats)
    ##
    scipy.io.savemat(path_to_save,experiment_results)

def save2matlab_old():
    if kwargs['save_bulk_experiment']:
        print('saving expt')
        path_to_save = f'./test_runs/{experiment_name}_reg_{reg_type}_expt_type_{expt_type}_N_train_{N_train}_M_{M}'
        experiment_results= dict(
            SLURM_JOBID=SLURM_JOBID,SLURM_ARRAY_TASK_ID=SLURM_ARRAY_TASK_ID,
            reg_type=reg_type,
            reg_lambda=reg_lambda,nb_iter=nb_iter,Degree_mdl=Degree_mdl,
            lambdas=lambdas,nb_iterations=nb_iterations,repetitions=repetitions,degrees=degrees,
            seconds=seconds,minutes=minutes,hours=hours,
            truth_filename=truth_filename,data_filename=data_filename,
            expt_type=expt_type,
            MDL_2_TRAIN=MDL_2_TRAIN,
            M=M,eta=eta,A=A
            )
        if MDL_2_TRAIN == 'PERT' or MDL_2_TRAIN == 'TRIG_PERT':
            experiment_results['w_norms'] = w_norms
            experiment_results['train_loss_list_WP'] = train_loss_list_WP
            experiment_results['test_loss_list_WP'] = test_loss_list_WP
            experiment_results['grad_list_weight_sgd'] = grad_list_weight_sgd
            experiment_results['frac_norm'] = frac_norm
            experiment_results['logging_freq'] = logging_freq
            experiment_results['perturbation_freq'] = perturbation_freq
            path_to_save = f'{path_to_save}_frac_norm_{frac_norm}_logging_freq_{logging_freq}_perturbation_freq_{perturbation_freq}'
        ##
        path_to_save=f'{path_to_save}/{prefix_experiment}'
        make_and_check_dir(path_to_save)
        path_to_save = f'{path_to_save}/satid_{SLURM_ARRAY_TASK_ID}_sid_{SLURM_JOBID}_{month}_{day}'
        scipy.io.savemat( path_to_save, experiment_results)
