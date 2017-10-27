import os

import scipy.io

expt_type_dirname = 'unit_test_reg_VW_expt_type_LAMBDAS'
expt_type_dirname = 'nonlinear_VW_expt1_reg_VW_expt_type_LAMBDAS'
## the name for experiments, for lambdas its the #iters for that set of lambdas, for iters its the specific lambda tried for that experiment
#set__experiments_dirname = 'lambda_0'
set_experiments_dirname = 'it_1400000'
##
path = f'./test_runs/{expt_type_dirname}/{set_experiments_dirname}'
for dirpath, dirnames, filenames in os.walk(path):
    if dirpath != path: # check is not neccessary, but essentially is a reminder that now its going through the actual contents of a dir with experiments
        for single_run_fname in filenames:
            path_to_file = f'{dirpath}/{single_run_fname}'
            print(f'path_to_file = {path_to_file}')
            expt_result = scipy.io.loadmat(path_to_file)
            print('\'SLURM_JOBID\' in expt_result: {}'.format( 'SLURM_JOBID' in expt_result ))
            if 'SLURM_JOBID' in expt_result:
                print(expt_result['SLURM_JOBID'])
            expt_result['SLURM_JOBID'] = 9584643
            path_2_f = f'{path}/{single_run_fname}'
            #print(f'expt_result = {expt_result}')
            ##
            path_to_save = path_to_file
            scipy.io.savemat( path_to_save, expt_result)
