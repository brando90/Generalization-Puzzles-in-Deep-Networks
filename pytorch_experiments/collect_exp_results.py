import os

import scipy.io
import numpy as np

import time
from datetime import date
import calendar

import pdb

def make_second_entry_numpy_array(experiment_results):
    return [ np.array(results_list) for hp,results_list in experiment_results ]

def make_tuple_2_into_float(error_dict):
    return [ (hp,float(error[0])) for hp,error in error_dict.items() ]

##
today_obj = date.today() # contains datetime.date(year, month, day); accessible via .day etc
day = today_obj.day
month = calendar.month_name[today_obj.month]
##
SLURM_JOBID = 1
## the name for lambdas or iterations, depending of type of experiment
expt_type_dirname = 'unit_test_reg_VW_expt_type_LAMBDAS'
expt_type_dirname = 'unit_test_reg_VW_expt_type_ITERATIONS'
expt_type_dirname = 'nonlinear_VW_expt1_reg_VW_expt_type_LAMBDAS' ## the name for experiments, for lambdas its the #iters for that set of lambdas, for iters its the specific lambda tried for that experiment
expt_type_dirname = 'nonlinear_V2W_D3_expt1_reg_V2W_D3_expt_type_LAMBDAS'
expt_type_dirname = 'expt_test_SP_reg__expt_type_SP_fig4'
#expt_type_dirname = 'linear_VW_expt1_reg_VW_expt_type_LAMBDAS'
#expt_type_dirname = 'linear_VW_expt1_reg_VW_expt_type_ITERATIONS'
#set_experiments_dirname = 'lambda_80000'
#set_experiments_dirname = 'lambda_0'
#set_experiments_dirname = 'it_12000'
#set_experiments_dirname = 'it_80000'
#set_experiments_dirname = 'it_1400000'
#set_experiments_dirname = 'it_4199999'
#set_experiments_dirname = 'it_60000'
set_experiments_dirname = 'fig4_expt_lambda_0_it_1600000'
##
path = f'./test_runs/{expt_type_dirname}/{set_experiments_dirname}'
print(f'path = {path}')
##
path_exists = os.path.exists(path)
print(f'Does path = {path}, exist? Answer: {path_exists}')
## here we decide which is the hyper param that we are testing makes a change, indepedent variable
if 'LAMBDAS' in expt_type_dirname:
    varying_hp = 'reg_lambda_WP'
    prefix_hp = f'experiment_lambdas_{month}_{day}_'
elif 'ITERATIONS' in expt_type_dirname:
    varying_hp = 'nb_iter'
    prefix_hp = f'experiment_iterations_{month}_{day}_'
elif 'SP_fig4' in expt_type_dirname:
    varying_hp = 'Degree_mdl'
    prefix_hp = f'experiment_degrees_{month}_{day}_'
else:
    raise ValueError(f'Look at path {expt_type_dirname}, contains neither LAMBDAS nor ITERATIONS.')
## go through the experiments and collect statistics
train_errors = {}
test_errors = {}
duration_secs = {}
for dirpath, dirnames, filenames in os.walk(path):
    if dirpath != path: # check is not neccessary, but essentially is a reminder that now its going through the actual contents of a dir with experiments
        for i,single_run_fname in enumerate(filenames):
            path_to_file = f'{dirpath}/{single_run_fname}'
            expt_result = scipy.io.loadmat(path_to_file)
            ##
            pdb.set_trace()
            hp = float(expt_result[varying_hp])
            train_error, test_error = expt_result['train_error_WP'], expt_result['test_error_WP']
            seconds = expt_result['seconds']
            print(f'varying_hp = {varying_hp}')
            print(f'hp = {hp}')
            print(f'train_error, test_error = {train_error},{test_error}')
            ##
            if hp not in train_errors:
                train_errors[hp] = np.zeros(len(filenames))
                test_errors[hp] = np.zeros(len(filenames))
                duration_secs[hp] = np.zeros(len(filenames))
            else:
                train_errors[hp][i] = train_error
                test_errors[hp][i] = test_error
                duration_secs[hp][i] = seconds
        #pdb.set_trace()
#pdb.set_trace()
##
train_errors = list( train_errors.items() )
test_errors = list( test_errors.items() )
duration_secs = list( duration_secs.items() )
## sort based on hp param
train_errors.sort(key=lambda x: x[0],reverse=True)
test_errors.sort(key=lambda x: x[0],reverse=True)
duration_secs.sort(key=lambda x: x[0],reverse=True)

## get [[erro_1],...,[error_K]]
hps = np.array([ hp for hp,_ in train_errors ])
train_errors = [ errors for hp,errors in train_errors ]
test_errors = [ errors for hp,errors in test_errors ]
duration_secs = [ seconds for hp,seconds in duration_secs ]
##
N_hp = len(train_errors)
train_means, train_stds = np.zeros( N_hp ), np.zeros( N_hp ) # one moment per lambda
test_means, test_stds = np.zeros( N_hp ), np.zeros( N_hp ) # one moment per lambda
## collect the statistics for every lambda
for i in range(len(train_errors)):
    train_means[i] = np.mean( train_errors[i] )
    train_stds[i] = np.std( train_errors[i] )
    #
    test_means[i] = np.mean( test_errors[i] )
    test_stds[i] = np.std( test_errors[i] )
##
#SLURM_JOBID = expt_result['SLURM_JOBID']
#SLURM_JOBID = 9582799
print(f'\nexpt_type_dirname={expt_type_dirname}')
print(f'varying_hp={varying_hp} \nprefix_hp={prefix_hp}')
print('saving\a')
path_to_save = f'../plotting/results/{prefix_hp}_{SLURM_JOBID}.mat'
print(f'path_to_save = {path_to_save}')
print(f'file_name = {prefix_hp}_{SLURM_JOBID}')
if varying_hp == 'reg_lambda_WP':
    one_over_lambdas = 1/hps
    scipy.io.savemat( path_to_save, dict(one_over_lambdas=one_over_lambdas, train_means=train_means,train_stds=train_stds, test_means=test_means,test_stds=test_stds) )
else:
    iterations = hps
    scipy.io.savemat( path_to_save, dict(iterations=iterations, train_means=train_means,train_stds=train_stds, test_means=test_means,test_stds=test_stds) )
