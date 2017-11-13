#!/usr/bin/env python
#SBATCH --mem=7000
#SBATCH --time=0-06:00
#SBATCH --array=1-300
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

sys.path.append(os.getcwd())

from pytorch_over_approx_high_dim import *
from models_pytorch import *
from inits import *
from sympy_poly import *
from poly_checks_on_deep_net_coeffs import *
from data_file import *
from plotting_utils import *

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy

from maps import NamedDict

import pdb

import unittest

SLURM_ARRAY_TASK_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
SLURM_JOBID = int(os.environ['SLURM_JOBID'])
#SLURM_ARRAY_TASK_ID = 1
#SLURM_JOBID = 0

print(os.getcwd())

print('SLURM_ARRAY_TASK_ID = ',SLURM_ARRAY_TASK_ID)
print('SLURM_JOBID = ',SLURM_JOBID)

def make_and_check_dir(path):
    '''
        tries to make dir/file, if it exists already does nothing else creates it.
    '''
    try:
        os.makedirs(path)
    except OSError:
        pass

def get_hp_to_run(hyper_params,repetitions,satid):
    '''
    Returns the hyper parameter the current satid (SLURM_ARRAY_TASK_ID) corresponds to.

    The way it works is by counting up from sum_i(repetitions[i]), once the current satid
    is larger than the current counter, then it figures out that it belongs to the previous batch
    of repeitions corresponding to that HP. So it picks the hyper_parm and runs it.
    '''
    if satid == 0:
        raise ValueError(f'The SLURM_ARRAY_TASK_ID = {satid} is illegal. Start your job at 1 please.')
    start_next_bundle_batch_jobs=1
    for hp_job_nb in range(len(hyper_params)):
        # print('----')
        # print('hp_job_nb = ', hp_job_nb)
        # print('start_next_bundle_batch_jobs ', start_next_bundle_batch_jobs)
        start_next_bundle_batch_jobs+=repetitions[hp_job_nb]
        # print('start_next_bundle_batch_jobs ', start_next_bundle_batch_jobs)
        if start_next_bundle_batch_jobs > satid:
            # print('---- DONE')
            # print('hp_job_nb = ', hp_job_nb)
            # print('start_next_bundle_batch_jobs ', start_next_bundle_batch_jobs)
            # print('satid ',satid)
            # print('----')
            return hyper_params[hp_job_nb]
    raise ValueError('There is something wrong with the number of jobs you submitted compared.')

def main(**kwargs):
    WP=False
    ##
    start_time = time.time()
    np.set_printoptions(suppress=True) #Whether or not suppress printing of small floating point values using scientific notation (default False).
    ##dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
    dtype = torch.FloatTensor
    ##
    today_obj = date.today() # contains datetime.date(year, month, day); accessible via .day etc
    day = today_obj.day
    month = calendar.month_name[today_obj.month]
    ## Data file names
    #truth_filename='data_gen_type_mdl=WP_D_layers_[30, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_30_N_test_32_lb_-1_ub_1_act_linear_nb_params_32_msg_'
    #data_filename='data_numpy_type_mdl=WP_D_layers_[30, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_30_N_test_32_lb_-1_ub_1_act_linear_nb_params_32_msg_.npz'
    #truth_filename='data_gen_type_mdl=WP_D_layers_[3, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_8_N_test_20_lb_-1_ub_1_act_poly_act_degree2_nb_params_5_msg_'
    #data_filename='data_numpy_type_mdl=WP_D_layers_[3, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_8_N_test_20_lb_-1_ub_1_act_poly_act_degree2_nb_params_5_msg_.npz'
    #truth_filename=''
    #data_filename='degree4_fit_2_sin_N_train_5_N_test_200.npz'
    truth_filename=''
    data_filename='sin_freq_sin_4_N_train_7_N_test_200_lb_train,ub_train_(0, 1)_lb_test,ub_test_(0.2, 0.8).npz'
    ## Folder for experiment
    #experiment_name = 'unit_test'
    #experiment_name = 'linear_unit_test'
    #experiment_name = 'nonlinear_VW_expt1'
    #experiment_name = 'nonlinear_V2W_D3_expt1'
    #experiment_name = 'unit_test_nonlinear_V2W_D3_expt1'
    #experiment_name = 'unit_test_SP'
    experiment_name = 'expt_test_SP'
    #experiment_name = 'linear_VW_expt1'
    ## Regularization
    #reg_type = 'tikhonov'
    #reg_type = 'VW'
    #reg_type = 'V2W_D3'
    reg_type = ''
    ## config params
    ## LAMBDAS
    # expt_type = 'LAMBDAS'
    # N_lambdas = 50
    # lb,ub = 0.01,10000
    # one_over_lambdas = np.linspace(lb,ub,N_lambdas)
    # lambdas = list( 1/one_over_lambdas )
    # lambdas = N_lambdas*[0.0]
    # nb_iterations = [int(1.4*10**6)]
    # nb_iterations = [int(8*10**4)]
    # nb_iterations = [int(100*1000)]
    # repetitions = len(lambdas)*[15]
    ## ITERATIONS
    # expt_type = 'ITERATIONS'
    # N_iterations = 30
    # lb,ub = 1,60*10**4
    # lambdas = [0]
    # nb_iterations = [ int(i) for i in np.linspace(lb,ub,N_iterations)]
    # repetitions = len(nb_iterations)*[10]
    ## SP DEGREE/MONOMIALS
    expt_type = 'SP_fig4'
    step_deg=1
    lb_deg,ub_deg = 1,100
    degrees = list(range(lb_deg,ub_deg+1,step_deg))
    lambdas = [0]
    nb_iter = 1600*1000
    #nb_iter = 1*100
    nb_iterations = [nb_iter]
    repetitions = len(degrees)*[3]
    ##
    #debug, debug_sgd = True, False
    ## Hyper Params SGD weight parametrization
    M = 5
    eta = 0.1 # eta = 1e-6
    A = 0.0
    logging_freq = 100
    ## pick the right hyper param
    if expt_type == 'LAMBDAS':
        degrees=[]
        reg_lambda = get_hp_to_run(hyper_params=lambdas,repetitions=repetitions,satid=SLURM_ARRAY_TASK_ID)
        nb_iter = nb_iterations[0]
        prefix_experiment = f'it_{nb_iter}/lambda_{reg_lambda}_reg_{reg_type}'
    elif expt_type == 'ITERATIONS':
        degrees=[]
        reg_lambda = lambdas[0]
        nb_iter = get_hp_to_run(hyper_params=nb_iterations,repetitions=repetitions,satid=SLURM_ARRAY_TASK_ID)
        prefix_experiment = f'lambda_{reg_lambda}/it_{nb_iter}_reg_{reg_type}'
    elif expt_type == 'SP_fig4':
        reg_lambda = lambdas[0]
        Degree_mdl = get_hp_to_run(hyper_params=degrees,repetitions=repetitions,satid=SLURM_ARRAY_TASK_ID)
        print(f'\n--Degree_mdl={Degree_mdl} \n--degrees={degrees} \n--SLURM_ARRAY_TASK_ID={SLURM_ARRAY_TASK_ID} \n')
        prefix_experiment = f'fig4_expt_lambda_{reg_lambda}_it_{nb_iter}/deg_{Degree_mdl}'
    else:
        raise ValueError(f'Experiment type expt_type={expt_type} does not exist, try a different expt_type.')
    print('reg_lambda = ',reg_lambda)
    print('nb_iter = ',nb_iter)
    #### Get Data set
    if truth_filename != '':
        mdl_truth_dict = torch.load('./data/'+truth_filename)
        D_layers_truth=extract_list_filename(truth_filename)
    ## load data
    data = np.load( './data/{}'.format(data_filename) )
    if 'lb' and 'ub' in data:
        data_lb, data_ub = data['lb'],data['ub']
    else:
        data_lb, data_ub = 0,1 #TODO change!
    X_train, Y_train = data['X_train'], data['Y_train']
    #X_train, Y_train = X_train[0:6], Y_train[0:6]
    X_test, Y_test = data['X_test'], data['Y_test']
    D_data = X_test.shape[1]
    ## get nb data points
    D0 = D_data
    N_train,_ = X_train.shape
    N_test,_ = X_test.shape
    ## activation function
    if WP:
        print('--->training WP mdl')
        adegree = 2
        ax = np.concatenate( (np.linspace(-20,20,100), np.linspace(-10,10,1000)) )
        aX = np.concatenate( (ax,np.linspace(-2,2,100000)) )
        act, c_pinv_relu = get_relu_poly_act2(aX,degree=adegree) # ax**2+bx+c, #[1, x^1, ..., x^D]
        print('c_pinv_relu = ', c_pinv_relu)
        #act = relu
        #act = lambda x: x
        #act.__name__ = 'linear'
        # plot_activation_func(act,lb=-20,ub=20,N=1000)
        # plt.show()
        #### 2-layered mdl

        H1 = 12
        D0,D1,D2 = D0,H1,1
        D_layers,act = [D0,D1,D2], act

        # H1,H2 = 20,20
        # D0,D1,D2,D3 = D0,H1,H2,1
        # D_layers,act = [D0,D1,D2,D3], act

        # H1,H2,H3 = 15,15,15
        # D0,D1,D2,D3,D4 = D0,H1,H2,H3,1
        # D_layers,act = [D0,D1,D2,D3,D4], act

        # H1,H2,H3,H4 = 25,25,25,25
        # D0,D1,D2,D3,D4,D5 = D0,H1,H2,H3,H4,1
        # D_layers,act = [D0,D1,D2,D3,D4,D5], act

        nb_layers = len(D_layers)-1 #the number of layers include the last layer (the regression layer)
        biases = [None] + [True] + (nb_layers-1)*[False] #bias only in first layer
        #biases = [None] + (nb_layers)*[True] # biases in every layer
        ## mdl degree and D
        nb_hidden_layers = nb_layers-1 #note the last "layer" is a summation layer for regression and does not increase the degree of the polynomial
        Degree_mdl = adegree**( nb_hidden_layers ) # only hidden layers have activation functions
        ## Lift data/Kernelize data
        poly_feat = PolynomialFeatures(degree=Degree_mdl)
        Kern_train, Kern_test = poly_feat.fit_transform(X_train), poly_feat.fit_transform(X_test)
        ## LA models
        if D0 == 1:
            c_pinv = np.polyfit( X_train.reshape((N_train,)) , Y_train.reshape((N_train,)) , Degree_mdl )[::-1]
        else:
            ## TODO: https://stackoverflow.com/questions/10988082/multivariate-polynomial-regression-with-numpy
            c_pinv = np.dot(np.linalg.pinv( Kern_train ),Y_train)
        ## inits
        init_config = Maps( {'w_init':'w_init_normal','mu':0.0,'std':0.01, 'bias_init':'b_fill','bias_value':0.01,'biases':biases ,'nb_layers':len(D_layers)} )
        w_inits_sgd, b_inits_sgd = get_initialization(init_config)
        init_config_standard_sgd = Maps( {'mu':0.0,'std':0.001, 'bias_value':0.01} )
        mdl_stand_initializer = lambda mdl: lifted_initializer(mdl,init_config_standard_sgd)
        ## SGD models
        if truth_filename:
            mdl_truth = NN(D_layers=D_layers_truth,act=act,w_inits=w_inits_sgd,b_inits=b_inits_sgd,biases=biases)
            mdl_truth.load_state_dict(mdl_truth_dict)
        mdl_sgd = NN(D_layers=D_layers,act=act,w_inits=w_inits_sgd,b_inits=b_inits_sgd,biases=biases)
        ## data to TORCH
        data = get_data_struct(X_train,Y_train,X_test,Y_test,Kern_train,Kern_test,dtype)
        ##
        nb_terms = c_pinv.shape[0]
        legend_mdl = f'SGD solution weight parametrization, number of monomials={nb_terms}, batch-size={M}, iterations={nb_iter}, step size={eta}'
    else:
        print('--->training SP mdl')
        ## Lift data/Kernelize data
        poly_feat = PolynomialFeatures(degree=Degree_mdl)
        Kern_train, Kern_test = poly_feat.fit_transform(X_train), poly_feat.fit_transform(X_test)
        ## LA models
        if D0 == 1:
            c_pinv = np.polyfit( X_train.reshape((N_train,)) , Y_train.reshape((N_train,)) , Degree_mdl )[::-1]
        else:
            ## TODO: https://stackoverflow.com/questions/10988082/multivariate-polynomial-regression-with-numpy
            c_pinv = np.dot(np.linalg.pinv( Kern_train ),Y_train)
        mdl_sgd = get_sequential_lifted_mdl(nb_monomials=c_pinv.shape[0],D_out=1, bias=False)
        #mdl_sgd[0].weight.data.fill_(0)
        ##
        data = get_data_struct(X_train,Y_train,X_test,Y_test,Kern_train,Kern_test,dtype)
        data.X_train, data.X_test = data.Kern_train, data.Kern_test
        ##
        nb_terms = c_pinv.shape[0]
        legend_mdl = f'SGD solution standard parametrization, number of monomials={nb_terms}, batch-size={M}, iterations={nb_iter}, step size={eta}'
    ## check number of monomials
    nb_monomials = int(scipy.misc.comb(D0+Degree_mdl,Degree_mdl))
    if nb_terms != int(scipy.misc.comb(D0+Degree_mdl,Degree_mdl)):
       raise ValueError(f'nb of monomials dont match D0={D0},Degree_mdl={Degree_mdl}, number of monimials fron pinv={nb_terms}, number of monomials analyticall = {nb_monomials}')
    ########################################################################################################################################################
    ## some debugging print statements
    print('nb = ', nb_iter)
    print('reg_lambda = ', reg_lambda)
    print('reg_type = ', reg_type)
    ##
    arg = Maps(reg_type=reg_type)
    keep_training=True
    train_loss_list_WP,test_loss_list_WP,grad_list_weight_sgd,func_diff_weight_sgd,erm_lamdas_WP,nb_module_params = train_SGD( arg,mdl_sgd,data, M,eta,nb_iter,A ,logging_freq ,dtype,c_pinv, reg_lambda)
    # while keep_training:
    #     try:
    #         train_loss_list_WP,test_loss_list_WP,grad_list_weight_sgd,func_diff_weight_sgd,erm_lamdas_WP,nb_module_params = train_SGD(
    #             arg,mdl_sgd,data, M,eta,nb_iter,A ,logging_freq ,dtype,c_pinv, reg_lambda
    #         )
    #         keep_training=False
    #     except Exception as e:
    #         err_msg = str(e)
    #         print(f'\nException caught during training with msg: {err_msg}')
    #         if WP:
    #             w_inits_sgd, b_inits_sgd = get_initialization(init_config)
    #             mdl_sgd = NN(D_layers=D_layers,act=act,w_inits=w_inits_sgd,b_inits=b_inits_sgd,biases=biases)
    #         else:
    #             mdl_sgd = get_sequential_lifted_mdl(nb_monomials=c_pinv.shape[0],D_out=1, bias=False)
    ##
    train_error_WP = (1/N_train)*(mdl_sgd.forward(data.X_train) - data.Y_train).pow(2).sum().data.numpy()
    test_error_WP = (1/N_test)*(mdl_sgd.forward(data.X_test) - Variable(torch.FloatTensor(Y_test)) ).pow(2).sum().data.numpy()
    reg = get_regularizer_term(arg, mdl_sgd,reg_lambda,X=data.X_train,Y=data.Y_train,l=2)
    erm_reg_WP = (1/N_train)*(mdl_sgd.forward(data.X_train) - data.Y_train).pow(2).sum() + reg_lambda*reg
    ## REPORT TIMES
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    print("--- %s seconds ---" % seconds )
    print("--- %s minutes ---" % minutes )
    print("--- %s hours ---" % hours )
    print('\a')
    if kwargs['save_bulk_experiment']:
        path_to_save = f'./test_runs/{experiment_name}_reg_{reg_type}_expt_type_{expt_type}/{prefix_experiment}/'
        make_and_check_dir(path_to_save)
        experiment_results= dict(
            SLURM_JOBID=SLURM_JOBID,SLURM_ARRAY_TASK_ID=SLURM_ARRAY_TASK_ID,
            reg_type=reg_type,
            reg_lambda=reg_lambda,nb_iter=nb_iter,
            lambdas=lambdas,nb_iterations=nb_iterations,repetitions=repetitions,degrees=degrees,
            train_error_WP=train_error_WP,test_error_WP=test_error_WP,erm_reg_WP=erm_reg_WP,
            seconds=seconds,minutes=minutes,hours=hours,
            truth_filename=truth_filename,data_filename=data_filename,
            expt_type=expt_type
            )
        path_to_save = f'{path_to_save}/satid_{SLURM_ARRAY_TASK_ID}_sid_{SLURM_JOBID}_{month}_{day}'
        scipy.io.savemat( path_to_save, experiment_results)
    ##
    print(f'plotting={kwargs}')
    print(f'lb_test=')
    if kwargs['plotting']:
        print('going to print')
        if D0==1:
            print(f'print D0={D0}')
            #f_sgd = lambda x: f_mdl_eval(x,mdl_sgd,dtype)
            plot_1D_stuff(NamedDict(data_lb=data_lb,data_ub=data_ub,dtype=dtype,poly_feat=poly_feat,mdl_sgd=mdl_sgd,data=data,legend_mdl=legend_mdl,c_pinv=c_pinv,X_train=X_train)
                )
            ##
            plt.show()

class TestStringMethods(unittest.TestCase):

    def test_get_lambda_to_run(self):
        hyper_params = [1,2,3]
        #repetitions = [2,3,4]
        repetitions = [15,15,15]
        satid = 1
        for hp_i in range(len(hyper_params)):
            nb_repetions_current_hp = repetitions[hp_i]
            for r_i in range(nb_repetions_current_hp):
                ans_hyper_param = hyper_params[hp_i]
                hyper_param = get_hp_to_run(hyper_params,repetitions,satid)
                print(f"(hyper_param, ans_hyper_param) = {hyper_param} {ans_hyper_param}")
                self.assertEqual(hyper_param,ans_hyper_param)
                satid+=1

if __name__ == '__main__':
    main(save_bulk_experiment=True,plotting=False)
    #unittest.main()
