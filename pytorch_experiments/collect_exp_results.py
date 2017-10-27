import os

import scipy

def make_second_entry_numpy_array(experiment_results):
    return [ np.array(results_list) for hp,results_list in experiment_results ]

## the name for lambdas or iterations, depending of type of experiment
expt_type_dirname = 'unit_test_reg_VW_expt_type_LAMBDAS'
## the name for experiments, for lambdas its the #iters for that set of lambdas, for iters its the specific lambda tried for that experiment
#set__experiments_dirname = 'lambda_0'
set_experiments_dirname = 'it_1400'
##
path = f'./test_runs/{expt_type_dirname}/{set_experiments_dirname}'
print(f'path = {path}')
##
path_exists = os.path.exists(path)
print(f'Does path = {path}, exist? Answer: {path_exists}')
## here we decide which is the hyper param that we are testing makes a change, indepedent variable
if 'LAMBDAS' in expt_type_dirname:
    varying_hp = 'reg_lambda_WP'
    prefix_hp = 'experiment_lambdas_oct7_'
elif 'ITERATIONS' in expt_type_dirname:
    varying_hp = 'nb_iter'
    prefix_hp = 'experiment_iter_oct7_'
else:
    raise ValueError(f'Look at path {expt_type_dirname}, contains neither LAMBDAS nor ITERATIONS.')
## go through the experiments and collect statistics
train_errors = {}
test_errors = {}
duration_secs = {}
for dirpath, dirnames, filenames in os.walk(path):
    if dirpath != path: # check is not neccessary, but essentially is a reminder that now its going through the actual contents of a dir with experiments
        for single_run_fname in filenames:
            path_to_file = f'{dirpath}/{single_run_fname}'
            expt_result = scipy.io.loadmat(path_to_file)
            ##
            train_error, test_error = expt_result['train_error_WP'], expt_result['test_error_WP']
            seconds = expt_result['seconds']
            if varying_hp not in train_errors:
                train_errors[varying_hp] = [train_error]
                test_errors[varying_hp] = [test_error]
                duration_secs[varying_hp] = [seconds]
            else:
                train_errors[varying_hp].append(train_error)
                test_errors[varying_hp].append(test_error)
                duration_secs[varying_hp].append(seconds)
## conversion table
train_errors = list( train_errors.items() ).sort(key=lambda x: x[0])
test_errors = list( test_errors.items() ).sort(key=lambda x: x[0])
duration_secs = list( duration_secs.items() ).sort(key=lambda x: x[0])

train_errors = make_second_entry_numpy_array(train_errors)
test_errors = make_second_entry_numpy_array(test_errors)
duration_secs = make_second_entry_numpy_array(duration_secs)
##
train_means, train_stds = np.zeros( nb_iterations ), np.zeros( nb_iterations ) # one moment per lambda
test_means, test_stds = np.zeros( nb_iterations ), np.zeros( nb_iterations ) # one moment per lambda
## collect the statistics for every lambda
for i in range(len(train_errors)):
    train_means[i] = np.mean( train_errors[i,:] )
    train_stds[i] = np.std( train_errors[i,:] )
    #
    test_means[i] = np.mean( test_errors[i,:] )
    test_stds[i] = np.std( test_errors[i,:] )
##
SLURM_JOBID = expt_result['SLURM_JOBID']
print('saving')
path_to_save = f'../plotting/results/{prefix_hp}_{SLURM_JOBID}.mat'
scipy.io.savemat( path_to_save, dict(iterations=iterations, train_means=train_means,train_stds=train_stds, test_means=test_means,test_stds=test_stds) )
