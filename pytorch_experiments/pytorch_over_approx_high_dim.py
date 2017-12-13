import time
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

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy
from sklearn.preprocessing import PolynomialFeatures

#TODO make dtype, DTYPE accross all script

def vectors_dims_dont_match(Y,Y_):
    '''
    Checks that vector Y and Y_ have the same dimensions. If they don't
    then there might be an error that could be caused due to wrong broadcasting.
    '''
    DY = tuple( Y.size() )
    DY_ = tuple( Y_.size() )
    if len(DY) != len(DY_):
        return True
    for i in range(len(DY)):
        if DY[i] != DY_[i]:
            return True
    return False

def extract_list_filename(str):
    lbsplit = str.split('[')
    rbsplit = lbsplit[1].split(']')
    return ast.literal_eval("[" + rbsplit[0] + "]")

def print_all_params(mdl_nn):
    for i in range( 1,len(mdl_nn.linear_layers) ):
        W = mdl_nn.linear_layers[i].weight
        print(W)
        if type(mdl_nn.linear_layers[i].bias) != type(None):
            b =  mdl_nn.linear_layers[i].bias
            print(b)
    # for W in mdl_nn.parameters():
    #     print(W)

def print_WV_stats(mdl_truth,mdl_sgd):
    print(' ---------- W and V norms')
    print(' --> W.norm(1)')
    print('mdl_truth.W.norm(1) = ', mdl_truth.linear_layers[1].weight.norm(1) + mdl_truth.linear_layers[1].bias.norm(1) )
    print('mdl_sgd.W.norm(1) = ', mdl_sgd.linear_layers[1].weight.norm(1) + mdl_sgd.linear_layers[1].bias.norm(1) )

    print(' --> W.norm(2)')
    print('mdl_truth.W.norm(2) = ', mdl_truth.linear_layers[1].weight.norm(2) + mdl_truth.linear_layers[1].bias.norm(2) )
    print('mdl_sgd.W.norm(2) = ', mdl_sgd.linear_layers[1].weight.norm(2) + mdl_sgd.linear_layers[1].bias.norm(2) )

    print(' --> V.norm(1)')
    print('mdl_truth.V.norm(1) = ', mdl_truth.linear_layers[2].weight.norm(1) )
    print('mdl_sgd.V.norm(1) = ', mdl_sgd.linear_layers[2].weight.norm(1) )

    print(' --> V.norm(2)')
    print('mdl_truth.V.norm(2) = ', mdl_truth.linear_layers[2].weight.norm(2) )
    print('mdl_sgd.V.norm(2) = ', mdl_sgd.linear_layers[2].weight.norm(2) )
    ##
    print(' ---------- all params norms')
    print( ' --> all_parms.norm(1)')
    print('mdl_truth.all_Params.norm(1) = {}'.format( norm_params_WP(mdl_truth,l=1) ) )
    print('mdl_sgd.all_Params.norm(1) = {}'.format( norm_params_WP(mdl_sgd,l=1) ) )

    print( ' --> all_parms.norm(2)')
    print('mdl_truth.all_Params.norm(2) = {}'.format( norm_params_WP(mdl_truth,l=2) ) )
    print('mdl_sgd.all_Params.norm(2) = {}'.format( norm_params_WP(mdl_sgd,l=2) ) )

def get_norm_layer(mdl,l=2):
    #print('mdl_truth.W.norm(1) = ', mdl_truth.linear_layers[1].weight.norm(1) + mdl_truth.linear_layers[1].bias.norm(1) )
    pass

def norm_params_WP(mdl,l=2):
    norm = 0
    for W in mdl.parameters():
        norm += W.norm(l)
    return norm

def get_data_struct(X_train,Y_train,X_test,Y_test,Kern_train,Kern_test, dtype):
    X_train_pytorch = Variable(torch.FloatTensor(X_train).type(dtype), requires_grad=False)
    Y_train_pytorch = Variable(torch.FloatTensor(Y_train).type(dtype), requires_grad=False)
    X_test_pytorch = Variable(torch.FloatTensor(X_test).type(dtype), requires_grad=False)
    Y_test_pytorch = Variable(torch.FloatTensor(Y_test).type(dtype), requires_grad=False)
    Kern_train_pytorch = Variable(torch.FloatTensor(Kern_train).type(dtype), requires_grad=False)
    Kern_test_pytorch = Variable(torch.FloatTensor(Kern_test).type(dtype), requires_grad=False)
    data_pytorch_struct = Maps( {'X_train':X_train_pytorch,'Y_train':Y_train_pytorch,
        'X_test':X_test_pytorch,'Y_test':Y_test_pytorch,
        'Kern_train':Kern_train_pytorch, 'Kern_test':Kern_test_pytorch}
         )
    return data_pytorch_struct

def get_symbols(D):
    '''
    D is the number of symbols.
    x_0, x_1, ..., x_D-1

    input D+1 if you want:
    x_0, x_1, ..., x_D-1, x_D
    '''
    symbols = []
    for i in range(D):
        symbols.append( 'x_'+str(i))
    return symbols

def f_mdl_LA(x,c,D_mdl=None):
    '''
    evaluates linear algebra (LA) model
    '''
    if type(x)==float:
        Deg,_ = c.shape
        Kern = poly_kernel_matrix( [x],Deg-1 )
        Y = np.dot(Kern,c)
    else:
        poly_feat = PolynomialFeatures(D_mdl)
        Kern = poly_feat.fit_transform(x)
        Y = np.dot(Kern,c)
    return Y

def f_mdl_eval(x,mdl_eval,dtype):
    '''
    evalautes pytorch model
    '''
    _,D = list(mdl_eval.parameters())[0].data.numpy().shape
    #pdb.set_trace()
    if len(list(mdl_eval.parameters())) == 2 or len(list(mdl_eval.parameters())) == 1:
        # TODO: I think this is when we are training a linear model with SGD
        x = poly_kernel_matrix( [x],D-1 )
        x = Variable(torch.FloatTensor([x]).type(dtype))
    else:
        if D==1:
            x = Variable(torch.FloatTensor([x]).type(dtype)).view(1,1)
        elif D==2:
            x = Variable(torch.FloatTensor([x]).type(dtype))
        else:
            raise ValueError('D {} is not supported yet.'.format(D))
    y_pred = mdl_eval.forward(x)
    return y_pred.data.numpy()

def L2_norm_2(f,g,lb=0,ub=1,D=1):
    '''
    compute L2 functional norm
    '''
    if D==1:
        f_g_2 = lambda x: (f(x) - g(x))**2
        result = integrate.quad(func=f_g_2, a=lb,b=ub)
        integral_val = result[0]
    elif D==2:
        gfun,hfun = lambda x: -1, lambda x: 1
        def f_g_2(x,y):
            #pdb.set_trace()
            x_vec = np.array([[x,y]])
            return (f(x_vec) - g(x_vec))**2
        result = integrate.dblquad(func=f_g_2, a=lb,b=ub, gfun=gfun,hfun=hfun)
        integral_val = result[0]
    else:
        raise ValueError(' D {} is not handled yet'.format(D))
    return integral_val

def index_batch(X,batch_indices,dtype):
    '''
    returns the batch indexed/sliced batch
    '''
    if len(X.shape) == 1: # i.e. dimension (M,) just a vector
        batch_xs = torch.FloatTensor(X[batch_indices]).type(dtype)
    else:
        batch_xs = torch.FloatTensor(X[batch_indices,:]).type(dtype)
    return batch_xs

def get_batch2(X,Y,M,dtype):
    '''
    get batch for pytorch model
    '''
    # TODO fix and make it nicer, there is pytorch forum question
    X,Y = X.data.numpy(), Y.data.numpy()
    N = len(Y)
    valid_indices = np.array( range(N) )
    batch_indices = np.random.choice(valid_indices,size=M,replace=False)
    batch_xs = index_batch(X,batch_indices,dtype)
    batch_ys = index_batch(Y,batch_indices,dtype)
    return Variable(batch_xs, requires_grad=False), Variable(batch_ys, requires_grad=False)

def is_NaN(value):
    '''
    Checks is value is problematic by checking if the value:
    is not finite, is infinite or is already NaN
    '''
    return not np.isfinite(value) or np.isinf(value) or np.isnan(value)

def count_params(mdl):
    '''
    count the number of parameters of a pytorch model
    '''
    tot = 0
    #params = []
    for m in mdl.parameters():
        #print('m: ',m)
        #params.append(m)
        tot += m.nelement() # returns Number of elements = nelement
    #print('\nparams: ',params)
    #pdb.set_trace()
    return tot # sum([m.nelement() for m in mdl.parameters()])

def plot_activation_func(act,lb=-20,ub=20,N=1000):
    '''
    plots activation function
    '''
    ## PLOT ACTIVATION
    fig3 = plt.figure()
    x_horizontal = np.linspace(lb,ub,N)
    plt_poly_act, = plt.plot(x_horizontal,act(x_horizontal))
    y_relu = np.maximum(0,x_horizontal)
    plt_relu, = plt.plot(x_horizontal,y_relu)
    plt.legend(
        [plt_poly_act,plt_relu],
        ['polynomial activation {}'.format(act.__name__),'ReLU activation'])
    # plt.legend(
    #     [plt_relu],
    #     ['ReLU activation'])
    plt.title('Activation function: {}'.format(act.__name__))

def save_data_set_mdl_sgd(path, run_type, lb=-1,ub=1,N_train=36,N_test=2025,msg='',visualize=False):
    '''
    generates a data set
    '''
    dtype = torch.FloatTensor
    #
    data_generator, D_layers, act = main(experiment_type='data_generation',run_type=run_type)
    #
    D = D_layers[0]
    if D==1:
        X_train = np.linspace(lb,ub,N_train).reshape(N_train,D)
        X_test = np.linspace(lb,ub,N_train).reshape(N_train,D)
    elif D ==  2:
        Xm_train,Ym_train = generate_meshgrid(N_train,lb,ub)
        X_train,_ = make_mesh_grid_to_data_set(Xm_train,Ym_train)
        #
        Xm_test,Ym_test = generate_meshgrid(N_test,lb,ub)
        X_test,_ = make_mesh_grid_to_data_set(Xm_test,Ym_test)
    else:
        pass
    ## data sets
    Y_train = get_Y_from_new_net(data_generator=data_generator, X=X_train,dtype=dtype)
    Y_test = get_Y_from_new_net(data_generator=data_generator, X=X_test,dtype=dtype)
    ##
    np_filename = 'data_numpy_D_layers_{}_nb_layers{}_N_train_{}_N_test_{}_lb_{}_ub_{}_act_{}_run_type_{}_msg_{}'.format(
        D_layers,len(D_layers),N_train,N_test,lb,ub,run_type,act.__name__,msg
    )
    filename = 'data_numpy_D_layers_{}_nb_layers{}_N_train_{}_N_test_{}_lb_{}_ub_{}_act_{}_run_type_{}_msg_{}'.format(
        D_layers,len(D_layers),N_train,N_test,lb,ub,run_type,act.__name__,msg
    )
    ## save data and data generator
    np.savez(path.format(np_filename), X_train=X_train,Y_train=Y_train, X_test=X_test,Y_test=Y_test)
    torch.save( data_generator.state_dict(), path.format(filename) )
    if visualize:
        if D==1:
            pass
        elif D==2:
            Xp,Yp,Zp = make_meshgrid_data_from_training_data(X_data=X_test, Y_data=Y_test)
            ##
            fig = plt.figure()
            #ax = fig.gca(projection='3d')
            ax = Axes3D(fig)
            surf = ax.plot_surface(Xp,Yp,Zp, cmap=cm.coolwarm)
            plt.title('Test function')
            ##
            plt.show()

def print_debug():
    grad_norm = W.grad.data.norm(2)
    delta = eta*W.grad.data
    for index, W in enumerate(mdl_sgd.parameters()):
        if debug_sgd:
            print('------------- grad_norm={} delta={} ',grad_norm,delta.norm(2))

def stats_logger(arg, mdl, data, eta,loss_list,test_loss_list,grad_list,func_diff,erm_lamdas, i,c_pinv, reg_lambda):
    N_train,_ = tuple(data.X_train.size())
    N_test,_ = tuple(data.X_test.size())
    ## log: TRAIN ERROR
    y_pred_train = mdl.forward(data.X_train)
    current_train_loss = (1/N_train)*(y_pred_train - data.Y_train).pow(2).sum().data.numpy()
    loss_list.append( float(current_train_loss) )
    ##
    y_pred_test = mdl.forward(data.X_test)
    #pdb.set_trace()
    current_test_loss = (1/N_test)*(y_pred_test - data.Y_test).pow(2).sum().data.numpy()
    test_loss_list.append( float(current_test_loss) )
    ## log: GEN DIFF/FUNC DIFF
    y_test_sgd = mdl.forward(data.X_test)
    y_test_pinv = Variable( torch.FloatTensor( np.dot( data.Kern_test.data.numpy(), c_pinv) ) )
    gen_diff = (1/N_test)*(y_test_sgd - y_test_pinv).pow(2).sum().data.numpy()
    func_diff.append( float(gen_diff) )
    ## ERM + regularization
    #erm_reg = get_ERM_lambda(arg,mdl,reg_lambda,X=data.X_train,Y=data.Y_train,l=2).data.numpy()
    reg = get_regularizer_term(arg, mdl,reg_lambda,X=data.X_train,Y=data.Y_train,l=2)
    erm_reg = (1/N_train)*(y_pred_train - data.Y_train).pow(2).sum() + reg_lambda*reg
    erm_lamdas.append( float(erm_reg.data.numpy()) )
    ##
    #func_diff_stand_weight( (1/N_test)*(y_test_sgd - y_test_pinv).pow(2).sum().data.numpy() )
    ## collect param stats
    for index, W in enumerate(mdl.parameters()):
        delta = eta*W.grad.data
        grad_list[index].append( W.grad.data.norm(2) )
        if is_NaN(W.grad.data.norm(2)) or is_NaN(current_train_loss):
            print('\n----------------- ERROR HAPPENED \a')
            print('reg_lambda', reg_lambda)
            print('error happened at: i = {} current_train_loss: {}, grad_norm: {},\n ----------------- \a'.format(i,current_train_loss,W.grad.data.norm(2)))
            #sys.exit()
            #pdb.set_trace()
            raise ValueError('Nan Detected')

def standard_tikhonov_reg(mdl,l):
    lp_reg = None
    for W in mdl.parameters():
        if lp_reg is None:
            lp_reg = W.norm(l)
        else:
            lp_reg = lp_reg + W.norm(l)
    return lp_reg

def VW_reg(mdl,l):
    ##
    b_w = mdl[1].bias
    W_p = mdl[1].weight
    V = mdl[2].weight
    ##
    VW = torch.matmul(V,W_p) + torch.matmul(V,b_w)
    reg = VW.norm(l)
    return reg

def V2W_reg(mdl,l):
    ##
    b_w = mdl[1].bias
    W_p = mdl[1].weight
    V = mdl[2].weight
    ## TODO
    pass

def V2W_D3_reg(mdl,l):
    #print('--> VW new\n')
    #pdb.set_trace()
    ##
    b_w = mdl[1].bias
    W_p = mdl[1].weight
    V = mdl[2].weight
    #pdb.set_trace()
    ##
    W_p_2 = W_p*W_p
    R = torch.sum( torch.matmul(V,W_p_2) )
    indices = [(0,1),(0,2),(1,2)]
    for i,j in indices:
        R += torch.matmul(V,W_p[:,i]*W_p[:,j] )
    indices = [(0,3),(1,3),(2,3)]
    for i,j in indices:
        R += torch.matmul(V,W_p[:,i])
    return R

def get_regularizer_term(arg, mdl,reg_lambda,X,Y,l=2):
    M, _ = tuple( X.size() )
    ## compute regularization
    #reg = standard_tickhonov_reg(mdl,l)
    if type(mdl) ==  NN: # WP
        #print(arg.reg_type)
        if arg.reg_type == 'VW':
            reg = VW_reg(mdl,l)
        elif arg.reg_type == 'V[^2W':
            reg = V2W(mdl,l)
        elif arg.reg_type == 'V2W_D3':
            reg = V2W_D3_reg(mdl,l)
        elif arg.reg_type == 'tikhonov':
            reg = standard_tikhonov_reg(mdl,l)
        elif arg.reg_type == '':
            reg = 0
        else:
            print('Error, arg.reg_type = {} is not valid'.format(arg.reg_type))
            sys.exit()
    else: # SP
        reg = standard_tikhonov_reg(mdl,l)
    return reg

def train_SGD(arg, mdl,data, M,eta,nb_iter,A ,logging_freq ,dtype,c_pinv,reg_lambda):
    '''
    '''
    nb_module_params = len( list(mdl.parameters()) )
    loss_list, grad_list =  [], [ [] for i in range(nb_module_params) ]
    func_diff = []
    erm_lamdas = []
    test_loss_list = []
    #func_current_mdl_to_other_mdl = []
    ##
    #pdb.set_trace()
    N_train, _ = tuple( data.X_train.size() )
    print(f'reg_lambda={reg_lambda}')
    for i in range(nb_iter):
        # Forward pass: compute predicted Y using operations on Variables
        batch_xs, batch_ys = get_batch2(data.X_train,data.Y_train,M,dtype) # [M, D], [M, 1]
        ## FORWARD PASS
        #pdb.set_trace()
        y_pred = mdl.forward(batch_xs)
        #print(f'y_pred={y_pred.data.size()} batch_ys={batch_ys.data.size()}')
        ## Check vectors have same dimension
        if vectors_dims_dont_match(batch_ys,y_pred):
            raise ValueError('You vectors don\'t have matching dimensions. It will lead to errors.')
        ## LOSS + Regularization
        if reg_lambda != 0:
            reg = get_regularizer_term(arg, mdl,reg_lambda,X=batch_xs,Y=batch_ys,l=2)
            batch_loss = (1/M)*(y_pred - batch_ys).pow(2).sum() + reg
        else:
            batch_loss = (1/M)*(y_pred - batch_ys).pow(2).sum()
            #print(f'batch_loss = {batch_loss}')
        #print(f'batch_xs.shape={batch_xs}, batch_ys={batch_ys}')
        #pdb.set_trace()
        ## BACKARD PASS
        batch_loss.backward() # Use autograd to compute the backward pass. Now w will have gradients
        ## SGD update
        for W in mdl.parameters():
            delta = eta*W.grad.data
            #print(f'W.grad.data = {W.grad.data}')
            #print(f'\ndelta = {delta}')

            #print(f'W.data = {W.data}')

            W.data.copy_(W.data - delta) # W - eta*g + A*gdl_eps

            #print(f'W.data = {W.data}')
            #pdb.set_trace()
        ## stats logger
        if i % logging_freq == 0:
            stats_logger(arg, mdl, data, eta,loss_list,test_loss_list,grad_list,func_diff,erm_lamdas, i,c_pinv, reg_lambda)
        ## train stats
        if i % (nb_iter/10) == 0 or i == 0:
        #if False:
            current_train_loss = (1/N_train)*(mdl.forward(data.X_train) - data.Y_train).pow(2).sum().data.numpy()
            print('-------------')
            print(f'i = {i}, current_train_loss = {current_train_loss}')
            print(f'W.data = {W.data}')
            print(f'W.grad.data = {W.grad.data}')
        ## Manually zero the gradients after updating weights
        mdl.zero_grad()
    return loss_list,test_loss_list,grad_list,func_diff,erm_lamdas,nb_module_params

##

def train_SGD_with_perturbations(arg, mdl,data, M,eta,nb_iter,A ,logging_freq ,dtype,c_pinv,reg_lambda, perturbation_freq, frac_norm):
    '''
    '''
    nb_module_params = len( list(mdl.parameters()) )
    loss_list, grad_list =  [], [ [] for i in range(nb_module_params) ]
    func_diff = []
    erm_lamdas = []
    test_loss_list = []
    w_norms = [ [] for i in range(nb_module_params) ]
    ##
    N_train, _ = tuple( data.X_train.size() )
    print(f'reg_lambda={reg_lambda}')
    for i in range(0,nb_iter):
        # Forward pass: compute predicted Y using operations on Variables
        batch_xs, batch_ys = get_batch2(data.X_train,data.Y_train,M,dtype) # [M, D], [M, 1]
        ## FORWARD PASS
        y_pred = mdl.forward(batch_xs)
        ## Check vectors have same dimension
        if vectors_dims_dont_match(batch_ys,y_pred):
            raise ValueError('You vectors don\'t have matching dimensions. It will lead to errors.')
        ## LOSS + Regularization
        if reg_lambda != 0:
            reg = get_regularizer_term(arg, mdl,reg_lambda,X=batch_xs,Y=batch_ys,l=2)
            batch_loss = (1/M)*(y_pred - batch_ys).pow(2).sum() + reg
        else:
            batch_loss = (1/M)*(y_pred - batch_ys).pow(2).sum()
        ## BACKARD PASS
        batch_loss.backward() # Use autograd to compute the backward pass. Now w will have gradients
        ## SGD update
        for W in mdl.parameters():
            delta = eta*W.grad.data
            W.data.copy_(W.data - delta) # W - eta*g + A*gdl_eps
        ## train stats
        if i % (nb_iter/10) == 0 or i == 0:
            current_train_loss = (1/N_train)*(mdl.forward(data.X_train) - data.Y_train).pow(2).sum().data.numpy()
            print('-------------')
            print(f'i = {i}, current_train_loss = {current_train_loss}')
            print(f'W.data = {W.data}')
            print(f'W.grad.data = {W.grad.data}')
        ## stats logger
        if i % logging_freq == 0 or i == 0:
            #indices.append(i)
            stats_logger(arg, mdl, data, eta,loss_list,test_loss_list,grad_list,func_diff,erm_lamdas, i,c_pinv, reg_lambda)
            for index, W in enumerate(mdl.parameters()):
                w_norms[index].append( W.data.norm(2) )
        ## DO OP
        if i % perturbation_freq == 0:
            for W in mdl.parameters():
                #pdb.set_trace()
                Din,Dout = W.data.size()
                ##std = frac_norm*W.norm(2).data*torch.ones(Din,Dout)
                std = frac_norm
                #std = frac_norm*torch.ones(Din,Dout)
                noise = torch.normal(means=0.0*torch.ones(Din,Dout),std=std)
                W.data.copy_(W.data + noise)
        ## Manually zero the gradients after updating weights
        mdl.zero_grad()
    return loss_list,test_loss_list,grad_list,func_diff,erm_lamdas,nb_module_params,w_norms

###############################################################################
###############################################################################
###############################################################################

def main(**kwargs):
    '''
    main code, where experiments for over parametrization are made
    '''
    if 'experiment_type' not in kwargs:
        raise ValueError( 'experiment_type must be present for main to run. Its value was {}'.format(experiment_type) )
    if 'plotting' not in kwargs:
        kwargs['plotting'] = False
    ##dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
    dtype = torch.FloatTensor
    #
    debug = True
    debug_sgd = False
    #debug_sgd = True
    ## Hyper Params SGD weight parametrization
    M = 3
    eta = 0.002 # eta = 1e-6
    if 'nb_iterations_WP' in kwargs:
        nb_iter = kwargs['nb_iterations_WP']
    else:
        #nb_iter = int(80*1000)
        nb_iter = int(1.4*10**1)
    #nb_iter = int(15*1000)
    A = 0.0
    if 'reg_lambda_WP' in kwargs:
        reg_lambda_WP = kwargs['reg_lambda_WP']
    else:
        reg_lambda_WP = 0.0
    print('WP training config')
    print(f'reg_lambda_WP={reg_lambda_WP}')
    print(f'M={M},eta={eta}')
    ## Hyper Params SGD standard parametrization
    M_standard_sgd = 3
    eta_standard_sgd = 0.002 # eta = 1e-6
    #nb_iter_standard_sgd = int(1000)
    nb_iter_standard_sgd = 10
    A_standard_sgd = 0.0
    #reg_lambda_SP = 0.0
    if 'reg_lambda_SP' in kwargs:
        reg_lambda_WP = kwargs['reg_lambda_WP']
    else:
        reg_lambda_SP = reg_lambda_WP
    ##
    logging_freq = 100
    logging_freq_standard_sgd = 5
    #### Get Data set
    if kwargs['experiment_type'] == 'data_generation': # empty dictionaries evluate to false
    #experiment_type='data_generation'
        # only executes this if kwargs dict is NOT empty
        run_type = kwargs['run_type']
        #collect_functional_diffs = kwargs['collect_functional_diffs']
    else:
        #run_type = 'sine'
        #run_type = 'similar_nn'
        run_type = 'from_file'
        #run_type = 'h_add'
    data_filename, truth_filename = None, None
    init_config_data = Maps({})
    f_true = None
    print('run_type = ', run_type)
    if run_type == 'sine':
        collect_functional_diffs = False
        collect_generalization_diffs = True
        N_train=5
        #N_train=1024 # 32**2
        N_test=2025 # 45**2
        X_train = np.linspace(lb,ub,N_train).reshape(N_train,1) # the real data points
        Y_train = np.sin(2*np.pi*X_train)
        X_test = np.linspace(lb,ub,N_test).reshape(N_test,1) # the real data points
        Y_test = np.sin(2*np.pi*X_test)
        #f_true = lambda x: np.sin(2*np.pi*x)]
        if D0 != X_test.shape[1]:
            raise ValueError('Input Dimension of data set and model do not match: expected {} got {}'.format(X_test.shape[1],D0))
    elif run_type == 'similar_nn':
        pass
    elif run_type == 'from_file':
        #collect_functional_diffs = True
        collect_functional_diffs = False
        collect_generalization_diffs = True
        ##
        #data_filename = 'data_numpy_D_layers_[1, 2, 2, 2, 1]_nb_layers5_biasTrue_mu0.0_std2.0_N_train_10_N_test_1000_lb_-1_ub_1_act_quad_ax2_bx_c_msg_.npz'
        #data_filename = 'data_numpy_D_layers_[2, 3, 3, 1]_nb_layers4_biasTrue_mu0.0_std2.0_N_train_10_N_test_1000_lb_-1_ub_1_act_quadratic_msg_.npz'
        #data_filename = 'data_numpy_D_layers_[2, 5, 5, 1]_nb_layers4_N_train_16_N_test_2025_lb_-1_ub_1_act_h_add_run_type_poly_act_degree2_msg_.npz'
        #data_filename = 'data_numpy_D_layers_[2, 5, 5, 1]_nb_layers4_N_train_16_N_test_5041_lb_-1_ub_1_act_h_add_run_type_poly_act_degree2_msg_.npz'
        #data_filename = 'data_numpy_D_layers_[2, 15, 15, 1]_nb_layers4_N_train_225_N_test_5041_lb_-1_ub_1_act_h_add_run_type_poly_act_degree2_msg_.npz'

        #data_filename = 'data_numpy_D_layers_[2, 15, 15, 1]_nb_layers4_bias[None, True, True, True]_mu0.0_std1.0_N_train_16_N_test_5041_lb_-1_ub_1_act_quad_ax2_bx_c_msg_.npz'
        #data_filename = 'data_numpy_D_layers_[2, 10, 10, 10, 1]_nb_layers5_bias[None, True, True, True, True]_mu0.0_std1.0_N_train_16_N_test_5041_lb_-1_ub_1_act_quad_ax2_bx_c_msg_.npz'
        #data_filename = 'data_numpy_D_layers_[2, 10, 1]_nb_layers3_bias[None, True, True]_mu0.0_std5.0_N_train_9_N_test_5041_lb_-1_ub_1_act_poly_act_degree3_msg_.npz'
        #data_filename = 'data_numpy_type_mdl=SP_D_layers_[2, 10, 1]_nb_layers3_bias[None, True, True]_mu0.0_std5.0_N_train_9_N_test_5041_lb_-1_ub_1_act_poly_act_degree3_msg_.npz'
        ##
        #data_filename = 'data_numpy_type_mdl=WP_D_layers_[2, 10, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_9_N_test_5041_lb_-1_ub_1_act_poly_act_degree3_nb_params_40_msg_.npz'
        #data_filename = 'data_numpy_type_mdl=SP_D_layers_[2, 10, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_9_N_test_5041_lb_-1_ub_1_act_poly_act_degree3_nb_params_10_msg_.npz'
        #data_filename = 'data_numpy_type_mdl=WP_D_layers_[2, 10, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_9_N_test_5041_lb_-1_ub_1_act_poly_act_degree3_nb_params_40_msg_1st_2nd_units_are_zero.npz'
        ##
        #truth_filename='data_gen_type_mdl=WP_D_layers_[2, 2, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_4_N_test_5041_lb_-1_ub_1_act_quad_ax2_bx_c_nb_params_8_msg_'
        #data_filename='data_numpy_type_mdl=WP_D_layers_[2, 2, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_4_N_test_5041_lb_-1_ub_1_act_quad_ax2_bx_c_nb_params_8_msg_.npz'
        #
        #truth_filename='data_gen_type_mdl=WP_D_layers_[2, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_4_N_test_5041_lb_-1_ub_1_act_quad_ax2_bx_c_nb_params_4_msg_'
        #data_filename='data_numpy_type_mdl=WP_D_layers_[2, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_4_N_test_5041_lb_-1_ub_1_act_quad_ax2_bx_c_nb_params_4_msg_.npz'
        #
        ## n=9,D=12, linear!
        # truth_filename='data_gen_type_mdl=WP_D_layers_[30, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_30_N_test_32_lb_-1_ub_1_act_linear_nb_params_32_msg_'
        # data_filename='data_numpy_type_mdl=WP_D_layers_[30, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_30_N_test_32_lb_-1_ub_1_act_linear_nb_params_32_msg_.npz'
        #truth_filename ='data_gen_type_mdl=WP_D_layers_[2, 10, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_9_N_test_5041_lb_-1_ub_1_act_poly_act_degree3_nb_params_40_msg_1st_2nd_units_are_zero'
        ##
        #truth_filename='data_gen_type_mdl=WP_D_layers_[15, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_13_N_test_25_lb_-1_ub_1_act_linear_nb_params_17_msg_'
        #data_filename='data_numpy_type_mdl=WP_D_layers_[15, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_13_N_test_25_lb_-1_ub_1_act_linear_nb_params_17_msg_.npz'
        ##
        truth_filename='data_gen_type_mdl=WP_D_layers_[3, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_8_N_test_20_lb_-1_ub_1_act_poly_act_degree2_nb_params_5_msg_'
        data_filename='data_numpy_type_mdl=WP_D_layers_[3, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_8_N_test_20_lb_-1_ub_1_act_poly_act_degree2_nb_params_5_msg_.npz'
        ##
        truth_filename=''
        data_filename='degree4_fit_2_sin_N_train_5_N_test_200.npz'
        ##
        #truth_filename='data_gen_type_mdl=WP_D_layers_[2, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_4_N_test_25_lb_-1_ub_1_act_poly_act_degree2_nb_params_4_msg_'
        #data_filename='data_numpy_type_mdl=WP_D_layers_[2, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_4_N_test_25_lb_-1_ub_1_act_poly_act_degree2_nb_params_4_msg_.npz'
        if truth_filename is not '':
            mdl_truth_dict = torch.load('./data/'+truth_filename)
            #mdl_truth_dict = torch.load(cwd+'/data'+truth_filename)
            print('mdl_truth_dict: ',mdl_truth_dict)
            print('data_filename = {} \n truth_filename = {}'.format(data_filename,truth_filename))
            ##
            D_layers_truth=extract_list_filename(truth_filename)
        ##
        data = np.load( './data/{}'.format(data_filename) )
        X_train, Y_train = data['X_train'], data['Y_train']
        #X_train, Y_train = X_train[0:6], Y_train[0:6]
        X_test, Y_test = data['X_test'], data['Y_test']
        D_data = X_test.shape[1]
    elif run_type == 'h_add':
        #collect_functional_diffs = True
        collect_functional_diffs = False
        #
        collect_generalization_diffs = True
        #
        N_train=1024
        #N_train=1024 # 32**2
        N_test=2025 # 45**2
        #
        X,Y,Z = generate_meshgrid_h_add(N=N_train,start_val=-1,end_val=1)
        X_train,Y_train = make_mesh_grid_to_data_set(X,Y,Z)
        print('N_train = {}'.format(X_train.shape[0]))
        X,Y,Z = generate_meshgrid_h_add(N=N_test,start_val=-1,end_val=1)
        X_test,Y_test = make_mesh_grid_to_data_set(X,Y,Z)
        print('N_train = {}, N_test = {}'.format(X_train.shape[0],X_test.shape[0]))
    elif run_type == 'h_gabor':
        #collect_functional_diffs = True
        collect_functional_diffs = False
        #
        collect_generalization_diffs = True
        #
        N_train=1024
        #N_train=1024 # 32**2
        N_test=2025 # 45**2
        #
        X,Y,Z = generate_meshgrid_h_gabor(N=N_train,start_val=-1,end_val=1)
        X_train,Y_train = make_mesh_grid_to_data_set(X,Y,Z)
        print('N_train = {}'.format(X_train.shape[0]))
        X,Y,Z = generate_meshgrid_h_gabor(N=N_test,start_val=-1,end_val=1)
        X_test,Y_test = make_mesh_grid_to_data_set(X,Y,Z)
        print('N_train = {}, N_test = {}'.format(X_train.shape[0],X_test.shape[0]))
    ## get nb data points
    N_train,_ = X_train.shape
    N_test,_ = X_test.shape
    print('N_train = {}, N_test = {}'.format(N_train,N_test))
    ## activation params
    #adegree = 2
    #alb, aub = -100, 100
    #aN = 100
    #act = get_relu_poly_act(degree=adegree,lb=alb,ub=aub,N=aN) # ax**2+bx+c
    ##
    adegree = 1
    # ax = np.concatenate( (np.linspace(-20,20,100), np.linspace(-10,10,1000)) )
    # aX = np.concatenate( (ax,np.linspace(-2,2,100000)) )
    # act, c_pinv_relu = get_relu_poly_act2(aX,degree=adegree) # ax**2+bx+c, #[1, x^1, ..., x^D]
    # print('c_pinv_relu = ', c_pinv_relu)
    act = relu
    # act = lambda x: x
    # act.__name__ = 'linear'
    ## plot activation
    # palb, paub = -20, 20
    # paN = 1000swqb
    # print('Plotting activation function')
    # plot_activation_func(act,lb=palb,ub=paub,N=paN)
    # plt.show()
    #### 2-layered mdl
    D0 = D_data

    #H1 = 12
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
    #pdb.set_trace()
    start_time = time.time()
    ##
    np.set_printoptions(suppress=True)
    lb, ub = -1, 1
    ## mdl degree and D
    nb_hidden_layers = nb_layers-1 #note the last "layer" is a summation layer for regression and does not increase the degree of the polynomial
    Degree_mdl = adegree**( nb_hidden_layers ) # only hidden layers have activation functions
    #### 1-layered mdl
    # identity_act = lambda x: x
    # D_1,D_2 = D_sgd,1 # note D^(0) is not present cuz the polyomial is explicitly constructed by me
    # D_layers,act = [D_1,D_2], identity_act
    # init_config = Maps( {'name':'w_init_normal','mu':0.0,'std':1.0} )
    # if init_config.name == 'w_init_normal':
    #     w_inits = [None]+[lambda x: w_init_normal(x,mu=init_config.mu,std=init_config.std) for i in range(len(D_layers)) ]
    # elif init_config.name == 'w_init_zero':
    #     w_inits = [None]+[lambda x: w_init_zero(x) for i in range(len(D_layers)) ]
    # ##b_inits = [None]+[lambda x: b_fill(x,value=0.1) for i in range(len(D_layers)) ]
    # ##b_inits = [None]+[lambda x: b_fill(x,value=0.0) for i in range(len(D_layers)) ]
    # b_inits = []
    # bias = False
    ## Lift data/Kernelize data
    poly_feat = PolynomialFeatures(degree=Degree_mdl)
    Kern_train = poly_feat.fit_transform(X_train)
    Kern_test = poly_feat.fit_transform(X_test)
    ## LA models
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
    mdl_standard_sgd = get_sequential_lifted_mdl(nb_monomials=c_pinv.shape[0],D_out=1, bias=False)
    ## data to TORCH
    data = get_data_struct(X_train,Y_train,X_test,Y_test,Kern_train,Kern_test,dtype)
    data_stand = get_data_struct(X_train,Y_train,X_test,Y_test,Kern_train,Kern_test,dtype)
    data_stand.X_train, data_stand.X_test = data_stand.Kern_train, data_stand.Kern_test
    ## DEBUG PRINTs
    print('>>norm(Y): ', ((1/N_train)*torch.norm(data.Y_train)**2).data.numpy()[0] )
    print('>>l2_loss_torch: ', (1/N_train)*( data.Y_train - mdl_sgd.forward(data.X_train)).pow(2).sum().data.numpy()[0] )
    ## check number of monomials
    nb_monomials = int(scipy.misc.comb(D0+Degree_mdl,Degree_mdl))
    print('>>>>>>>>>>>>> nb_monomials={}, c_pinv.shape[0]={} \n '.format(nb_monomials,c_pinv.shape[0]) )
    print('count_params(mdl_WP) {} '.format( count_params(mdl_sgd) ))
    if c_pinv.shape[0] != int(scipy.misc.comb(D0+Degree_mdl,Degree_mdl)):
       raise ValueError('nb of monomials dont match D0={},Degree_mdl={}, number of monimials fron pinv={}, number of monomials analyticall = {}'.format( D0,Degree_mdl,c_pinv.shape[0],int(scipy.misc.comb(D0+Degree_mdl,Degree_mdl)) )    )
    ########################################################################################################################################################
    print('Weight Parametrization SGD training')
    if 'reg_type_wp' in kwargs:
        reg_type_wp = kwargs['reg_type_wp']
    else:
        reg_type_wp = 'tikhonov'
    print('reg_type_wp = ', reg_type_wp)
    ##
    arg = Maps(reg_type=reg_type_wp)
    keep_training=True
    train_loss_list_WP,test_loss_list_WP,grad_list_weight_sgd,func_diff_weight_sgd,erm_lamdas_WP,nb_module_params = train_SGD(
        arg,mdl_sgd,data, M,eta,nb_iter,A ,logging_freq ,dtype,c_pinv, reg_lambda_WP
    )
    # while keep_training:
    #     try:
    #         train_loss_list_WP,test_loss_list_WP,grad_list_weight_sgd,func_diff_weight_sgd,erm_lamdas_WP,nb_module_params = train_SGD(
    #             arg,mdl_sgd,data, M,eta,nb_iter,A ,logging_freq ,dtype,c_pinv, reg_lambda_WP
    #         )
    #         keep_training=False
    #     except Exception as e:
    #         err_msg = str(e)
    #         print(f'\Exception caught during training with msg: {err_msg}')
    #         w_inits_sgd, b_inits_sgd = get_initialization(init_config)
    #         mdl_sgd = NN(D_layers=D_layers,act=act,w_inits=w_inits_sgd,b_inits=b_inits_sgd,biases=biases)
    ##
    print('Standard Parametrization SGD training')
    arg = Maps(reg_type='tikhonov')
    test_loss_list_SP,test_loss_list_SP,grad_list_standard_sgd,func_diff_standard_sgd,erm_lamdas_SP,nb_module_params = train_SGD(
        arg,mdl_standard_sgd,data_stand, M_standard_sgd,eta_standard_sgd,nb_iter_standard_sgd,A_standard_sgd ,logging_freq ,dtype,c_pinv, reg_lambda_SP
    )
    print('training ended!')
    ########################################################################################################################################################
    # print('--------- mdl_truth')
    # for W in mdl_truth.parameters():
    #     print(W)
    # print('--------- mdl_sgd')
    # for W in mdl_sgd.parameters():
    #     print(W)
    ##
    # if mdl_truth is not None:
    #     print_WV_stats(mdl_truth,mdl_sgd)
    ## print all parameters of WP
    print_all_params(mdl_nn=mdl_sgd)
    #pdb.set_trace()
    ########################################################################################################################################################
    ## SGD pNN
    nb_params = count_params(mdl_sgd)
    #pdb.set_trace()
    ## Do SYMPY magic
    if len(D_layers) <= 2:
        c_WP = list(mdl_sgd.parameters())[0].data.numpy()
        c_WP = c_WP.transpose()
    else:
        # e.g. x = Matrix(2,1,[a,a])
        x_list = [ symbols('x'+str(i)) for i in range(1,D0+1) ]
        x = Matrix(D0,1,x_list)
        tmdl = mdl_sgd
        if act.__name__ == 'poly_act_degree{}'.format(adegree):
            sact = lambda x: s_Poly(x,c_pinv_relu)
            sact.__name__ = 'spoly_act_degree{}'.format(adegree)
            if adegree >= 10:
                sact = sQuad
        elif act.__name__ == 'quadratic':
            sact = sQuad
        elif act.__name__ == 'relu':
            sact = sReLU
        elif act.__name__ == 'linear':
            sact = sLinear
        smdl = sNN(sact,biases,mdl=tmdl)
        ## get simplification
        expr = smdl.forward(x)
        s_expr = poly(expr,x_list)
        # coeffs(order=grlex,grevlex) # for order https://stackoverflow.com/questions/46385303/how-does-one-organize-the-coefficients-of-polynomialfeatures-in-lexicographical
        order='grevlex'
        print('order  = {}'.format(order))
        c_WP = np.array( s_expr.coeffs(order=order)[::-1] )
        nb_terms = len(c_WP)
        c_WP = np.array( [ np.float64(num) for num in c_WP] ).reshape(nb_terms,1)
        c_SP = mdl_standard_sgd[0].weight.data.numpy().reshape(nb_monomials,1)
        #pdb.set_trace()
    if debug:
        #pdb.set_trace()
        print('c_WP_standard = ', c_SP)
        print('c_WP_weight = ', c_WP)
        print('c_pinv: ', c_pinv)
        print('data.X_train = ', data.X_train)
        print('data.Y_train = ', data.Y_train)
        print(mdl_sgd)
        if len(D_layers) > 2:
            print('\n---- structured poly: {}'.format(str(s_expr)) )
    ##
    print('number of monomials wSGD={},sSGD={},pinv={}'.format( len(c_WP), nb_monomials, c_pinv.shape[0]) )
    if act.__name__ != 'linear':
        if len(c_WP) != nb_monomials or len(c_WP) != c_pinv.shape[0] or nb_monomials != c_pinv.shape[0]:
            raise ValueError(' Some error in the number of monomials, these 3 numbers should match but they dont: {},{},{}'.format(len(c_WP),nb_monomials,c_pinv.shape[0]) )
    ## data set Stats
    print('\n----> Data set stats:\n data_filename= {}, run_type={}, init_config_data={}\n'.format(data_filename,run_type,init_config_data) )
    ## Stats of model pNN wSGD model
    print('---- Learning params for weight param')
    print('Degree_mdl = {}, number of monomials = {}, N_train = {}, M = {}, eta = {}, nb_iter = {} nb_params={},D_layers={}'.format(Degree_mdl,len(c_WP),N_train,M,eta,nb_iter,nb_params,D_layers))
    print('Activations: act={}, sact={}'.format(act.__name__,sact.__name__) )
    print('init_config: ', init_config)
    print('number of layers = {}'.format(nb_module_params))
    ## Stats of model standard param (lifted space) sSGD
    print('Degree_mdl = {}, number of monomials = {}, N_train = {}, M_standard_sgd = {}, eta_standard_sgd = {}, nb_iter_standard_sgd = {} nb_params={}'.format(Degree_mdl,c_SP.shape[0],N_train,M_standard_sgd,eta_standard_sgd,nb_iter_standard_sgd,c_SP.shape[0]))
    print('init_config_standard_sgd: ', init_config_standard_sgd)
    ## Parameter Norms
    print('\n---- statistics about learned params')
    print('nb_iter = {}, nb_iter_standard_sgd = {}'.format(nb_iter,nb_iter_standard_sgd))
    print('reg_lambda_WP = {}, reg_lambda_SP = {}'.format(reg_lambda_WP,reg_lambda_SP))
    if len(D_layers) >= 2:
        print('--L1')
        print('||c_pinv||_1 = {} '.format(np.linalg.norm(c_pinv,1)) )
        print('||c_WP_weight||_1 = {} '.format(np.linalg.norm(c_WP,1)) )
        print('||c_SP_stand||_1 = {} '.format(np.linalg.norm(c_SP,1)) )
        print('--L2')
        print('||c_pinv||_2 = ', np.linalg.norm(c_pinv,2))
        print('||c_WP_weight||_2 = ', np.linalg.norm(c_WP,2))
        print('||c_SP_stand||_2 = {} '.format(np.linalg.norm(c_SP,2)) )
    ## Parameter Difference
    print('---- parameters comparison stats')
    if act.__name__ != 'linear':
        print('||c_WP - c_pinv||^2_2 = ', np.linalg.norm(c_WP - c_pinv,2))
        print('||c_WP - c_SP||^2_2 = ', np.linalg.norm(c_WP - c_SP,2))
    print('||c_SP - c_pinv||^2_2 = ', np.linalg.norm(c_SP - c_pinv,2))
    ## Generalization L2 (functional) norm difference
    print('-- Generalization difference L2 (arrpox. functional difference)')
    y_test_pinv = Variable( torch.FloatTensor( np.dot( Kern_test, c_pinv) ) )
    y_test_sgd_stand = mdl_standard_sgd.forward(data.Kern_test)
    y_test_sgd_weight = mdl_sgd.forward(data.X_test)
    print('N_test={}'.format(N_test))
    print('J_Gen|| f_WP - f_SP||^2_2 = {}'.format( (1/N_test)*(y_test_sgd_weight - y_test_sgd_stand).pow(2).sum().data.numpy() ) )
    print('J_Gen|| f_WP - f_pinv||^2_2 = {}'.format( (1/N_test)*(y_test_sgd_weight - y_test_pinv).pow(2).sum().data.numpy() ) )
    print('J_Gen|| f_SP - f_pinv||^2_2 = {}'.format( (1/N_test)*(y_test_sgd_stand - y_test_pinv).pow(2).sum().data.numpy() ) )
    ##
    print('-- test error/Generalization l2 error')
    if f_true ==  None:
        test_error_WP = (1/N_test)*(mdl_sgd.forward(data.X_test) - Variable(torch.FloatTensor(Y_test)) ).pow(2).sum().data.numpy()
        test_error_SP = (1/N_test)*(mdl_standard_sgd.forward(data.Kern_test) - Variable(torch.FloatTensor(Y_test)) ).pow(2).sum().data.numpy()
        test_error_pinv = (1/N_test)*(np.linalg.norm(Y_test-np.dot( poly_feat.fit_transform(X_test),c_pinv))**2)
        print('J_gen(f_sgd)_weight = ', test_error_WP )
        print('J_gen(f_sgd)_standard = ', test_error_SP )
        print('J_gen(f_pinv) = ', test_error_pinv )
    else:
        f_sgd = lambda x: f_mdl_eval(x,mdl_sgd,dtype)
        f_pinv = lambda x: f_mdl_LA(x,c_pinv)
        print('||f_sgd - f_true||^2_2 = ', L2_norm_2(f=f_sgd,g=f_true,lb=lb,ub=ub))
        print('||f_pinv - f_true||^2_2 = ', L2_norm_2(f=f_pinv,g=f_true,lb=lb,ub=ub))
    ## TRAIN ERRORS of mdls
    print('-- Train Error')
    train_error_WP = (1/N_train)*(mdl_sgd.forward(data.X_train) - data.Y_train).pow(2).sum().data.numpy()
    train_error_SP = (1/N_train)*(mdl_standard_sgd.forward( data.Kern_train ) - data.Y_train ).pow(2).sum().data.numpy()
    train_error_pinv = (1/N_train)*(np.linalg.norm(data.Y_train.data.numpy()-np.dot( poly_feat.fit_transform(data.X_train.data.numpy()) ,c_pinv))**2)
    print(' J(f_WP) = ', train_error_WP )
    print(' J(f_SP) = ', train_error_SP )
    print(' J(f_pinv) = ', train_error_pinv )
    ## Errors with Regularization
    #erm_reg_WP = get_ERM_lambda(arg=arg, mdl=mdl_sgd,reg_lambda=reg_lambda_WP,X=data.X_train,Y=data.Y_train).data.numpy()
    reg = get_regularizer_term(arg, mdl_sgd,reg_lambda_WP,X=data.X_train,Y=data.Y_train,l=2)
    erm_reg = (1/N_train)*(mdl_sgd.forward(data.X_train) - data.Y_train).pow(2).sum() + reg_lambda*reg
    print(' ERM_lambda(f_WP) = ', erm_reg_WP )
    #print(' J(c_rls) = ',(1/N)*(np.linalg.norm(Y-(1/N)*(np.linalg.norm(Y-np.dot( poly_kernel_matrix( x_true,D_sgd-1 ),c_rls))**2) )**2) )
    ## REPORT TIMES
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    print("--- %s seconds ---" % seconds )
    print("--- %s minutes ---" % minutes )
    print("--- %s hours ---" % hours )
    ##
    if kwargs['experiment_type'] == 'serial_multiple_lambdas':
        return train_error_WP, test_error_WP, erm_reg_WP
    ## plots
    print('\a')
    if kwargs['plotting']:
        if D0 == 1:
            x_horizontal = np.linspace(lb,ub,1000).reshape(1000,1)
            X_plot = poly_feat.fit_transform(x_horizontal)
            X_plot_pytorch = Variable( torch.FloatTensor(X_plot), requires_grad=False)
            #plots objs
            f_sgd = lambda x: f_mdl_eval(x,mdl_sgd,dtype)
            Y_sgd_stand =  [ float(f_val) for f_val in mdl_standard_sgd.forward(X_plot_pytorch).data.numpy() ]
            p_sgd_stand, = plt.plot(x_horizontal, Y_sgd_stand)
            p_sgd, = plt.plot(x_horizontal, [ float(f_sgd(x_i)[0]) for x_i in x_horizontal ])
            p_pinv, = plt.plot(x_horizontal, np.dot(X_plot,c_pinv))
            p_data, = plt.plot(X_train,data.Y_train.data.numpy(),'ro')
            ## legend
            plt.legend(
                    [p_sgd_stand,p_sgd,p_pinv,p_data],
                    ['SGD solution standard parametrization, number of monomials={}, batch-size={}, iterations={}, step size={}'.format(mdl_standard_sgd[0].weight.data.numpy().shape[1],M_standard_sgd,nb_iter_standard_sgd,eta_standard_sgd),
                    'SGD solution weight parametrization, number of monomials={}, batch-size={}, iterations={}, step size={}'.format(c_pinv.shape[0],M,nb_iter,eta),
                    'min norm (pinv) number of monomials={}'.format(c_pinv.shape[0]),
                    'data points']
                )
            ##
            plt.xlabel('x'), plt.ylabel('f(x)')
            plt.title('SGD vs minimum norm solution curves')
        elif D0 == 2:
            #
            nb_non_linear_layers = len(D_layers)-2
            degree_sgd = adegree**(len(D_layers)-2)
            #
            X_data, Y_data = X_test,Y_test
            Xp,Yp,Zp = make_meshgrid_data_from_training_data(X_data=X_data, Y_data=Y_data) # meshgrid for visualization
            Xp_train,Yp_train,Zp_train = make_meshgrid_data_from_training_data(X_data=X_train, Y_data=Y_train) # meshgrid for trainign points
            ## plot data PINV
            Y_pinv = np.dot(poly_feat.fit_transform(X_data),c_pinv)
            _,_,Zp_pinv = make_meshgrid_data_from_training_data(X_data=X_data, Y_data=Y_pinv)
            ## plot data SGD
            Y_sgd = mdl_sgd.forward(Variable(torch.FloatTensor(X_data))).data.numpy()
            _,_,Zp_sgd = make_meshgrid_data_from_training_data(X_data=X_data, Y_data=Y_sgd)
            ## plot data standard SGD
            Y_sgd_stand = mdl_standard_sgd.forward(Variable(torch.FloatTensor(Kern_test)) ).data.numpy()
            _,_,Zp_sgd_stand = make_meshgrid_data_from_training_data(X_data=X_data, Y_data=Y_sgd_stand)
            ## FIG PINV
            fig1 = plt.figure()
            ax1 = Axes3D(fig1)
            data_pts = ax1.scatter(Xp_train,Yp_train,Zp_train, marker='D')
            surf = ax1.plot_surface(Xp,Yp,Zp_pinv,color='y',cmap=cm.coolwarm)
            ax1.set_xlabel('x1'),ax1.set_ylabel('x2'),ax1.set_zlabel('f(x)')
            surf_proxy = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker ='_')
            ax1.legend([surf_proxy,data_pts],[
                'minimum norm solution Degree model={}, number of monomials={}'.format(Degree_mdl,nb_monomials),
                'data points, number of data points = {}'.format(N_train)])
            ## FIG SGD weight parametrization
            fig2 = plt.figure()
            ax2 = Axes3D(fig2)
            data_pts = ax2.scatter(Xp_train,Yp_train,Zp_train, marker='D')
            surf = ax2.plot_surface(Xp,Yp,Zp_sgd,cmap=cm.coolwarm)
            ax2.set_xlabel('x1'),ax2.set_ylabel('x2'),ax2.set_zlabel('f(x)')
            surf_proxy = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker = '_')
            ax2.legend([surf_proxy,data_pts],[
                'SGD solution weight parametrization Degree model={} non linear-layers={}, number of monomials={}, param count={}, list of units per nonlinear layer={}, batch-size={}, iterations={}, step size={}'.format(degree_sgd,nb_non_linear_layers,len(c_WP),nb_params,D_layers[1:len(D_layers)-1], M,nb_iter,eta),
                'data points, number of data points = {}'.format(N_train)])
            ## FIG SGD standard param
            fig = plt.figure()
            ax3 = Axes3D(fig)
            data_pts = ax3.scatter(Xp_train,Yp_train,Zp_train, marker='D')
            surf = ax3.plot_surface(Xp,Yp,Zp_sgd_stand, cmap=cm.coolwarm)
            ax3.set_xlabel('x1'),ax3.set_ylabel('x2'),ax3.set_zlabel('f(x)')
            ax3.legend([surf_proxy,data_pts],[
                'SGD solution standard parametrization Degree model={}, number of monomials={}, param count={}, batch-size={}, iterations={}, step size={}'.format(degree_sgd,nb_monomials,nb_monomials,M_standard_sgd,nb_iter_standard_sgd,eta_standard_sgd),
                'data points, number of data points = {}'.format(N_train)])
            ## PLOT train surface
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # points_scatter = ax.scatter(Xp_train,Yp_train,Zp_train, marker='D')
            # surf = ax.plot_surface(Xp_train,Yp_train,Zp_train, cmap=cm.coolwarm)
            # plt.title('Train function')
            # ## PLOT test surface
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # points_scatter = ax.scatter(Xp,Yp,Zp, marker='D')
            # surf = ax.plot_surface(Xp,Yp,Zp, cmap=cm.coolwarm)
            # plt.title('Test function')
        ## PLOT LOSSES
        # fig1 = plt.figure()
        # p_loss_w, = plt.plot(np.arange(len(train_loss_list_WP)), train_loss_list_WP,color='m')
        # p_loss_s, = plt.plot(np.arange(len(test_loss_list_SP)), test_loss_list_SP,color='r')
        # plt.legend([p_loss_w,p_loss_s],['plot train loss, weight parametrization','plot train loss, standard parametrization'])
        # plt.title('Loss vs Iterations')
        ## PLOT info
        #iterations_axis = np.arange(1,len(train_loss_list_WP),step=logging_freq)
        start = 1
        iterations_axis = np.arange(1,nb_iter,step=logging_freq)[start:]
        #iterations_axis = np.arange(0,len(train_loss_list_WP))
        train_loss_list_WP, test_loss_list_WP, erm_lamdas_WP = np.array(train_loss_list_WP)[start:], np.array(test_loss_list_WP)[start:], np.array(erm_lamdas_WP)[start:]
        p_train_WP_legend = 'Train error, Weight Parametrization (WP), reg_lambda_WP = {}'.format(reg_lambda_WP)
        p_test_WP_legend = 'Test error, Weight Parametrization (WP) reg_lambda_WP = {}'.format(reg_lambda_WP)
        p_erm_reg_WP_legend = 'Error+Regularization, Weight Parametrization (WP) reg_lambda_WP = {}'.format(reg_lambda_WP)
        ##plots
        fig1 = plt.figure()
        p_erm_reg_WP, = plt.plot(iterations_axis, erm_lamdas_WP,color='g')
        plt.legend([p_erm_reg_WP],[p_erm_reg_WP_legend])
        plt.xlabel('iterations' )
        plt.ylabel('Error/loss')
        plt.title('Loss+Regularization vs Iterations, reg_lambda_WP = {}'.format(reg_lambda_WP))

        fig1 = plt.figure()
        p_train_WP, = plt.plot(iterations_axis, train_loss_list_WP,color='m')
        p_test_WP, = plt.plot(iterations_axis, test_loss_list_WP,color='r')
        plt.xlabel('iterations' )
        plt.ylabel('Error/loss')
        plt.legend([p_train_WP,p_test_WP],[p_train_WP_legend,p_test_WP_legend])
        plt.title('Train,Test vs Iterations, reg_lambda_WP = {}'.format(reg_lambda_WP))
        # PLOT ERM+train+test
        fig1 = plt.figure()
        p_train_WP, = plt.plot(iterations_axis, train_loss_list_WP,color='m')
        p_test_WP, = plt.plot(iterations_axis, test_loss_list_WP,color='r')
        p_erm_reg_WP, = plt.plot(iterations_axis, erm_lamdas_WP,color='g')
        plt.xlabel('iterations' )
        plt.ylabel('Error/loss')
        plt.legend([p_erm_reg_WP,p_train_WP,p_test_WP],[p_erm_reg_WP_legend,p_train_WP_legend,p_test_WP_legend])
        plt.title('Loss+Regularization,Train,Test vs Iterations, reg_lambda_WP = {}'.format(reg_lambda_WP))

        ##
        # for i in range(len(grad_list)):
        #     fig2 = plt.figure()
        #     current_grad_list = grad_list[i]
        #     #pdb.set_trace()
        #     p_grads, = plt.plot(np.arange(len(current_grad_list)), current_grad_list,color='g')
        #     plt.legend([p_grads],['plot grads'])
        #     plt.title('Gradient vs Iterations: # {}'.format(i))
        #
        #plot_activation_func(act)
        ##
        #func_diff
        #train_loss_list_WP,grad_list_weight_sgd,func_diff_weight_sgd,nb_module_params
        fig = plt.figure()
        p_func_diff_standard, = plt.plot(np.arange(len(func_diff_standard_sgd)), func_diff_standard_sgd,color='g')
        p_func_diff_weight, = plt.plot(np.arange(len(func_diff_weight_sgd)), func_diff_weight_sgd,color='b')
        plt.legend([p_func_diff_weight,p_func_diff_standard],
            [' L2 generalization distance: weight parametrization with SGD minus minimum norm solution, number test points = {}'.format(N_test),
            ' L2 generalization distance: standard parametrization with SGD minus minimum norm solution, number test points = {}'.format(N_test)]
            )
        plt.title('Generalization L2 difference between minimum norm and SGD functions')
        ##
        plt.show()
    if kwargs['experiment_type'] == 'data_generation': # empty dicts evaluate to false
        return mdl_sgd, D_layers, act

if __name__ == '__main__':
    print('__main__ started')
    #main(experiment_type='quick_run',plotting=True)
    reg_type_wp='V2W_D3'
    reg_type_wp=''
    reg_lambda = 0
    train_error, test_error, erm_reg = main(experiment_type='serial_multiple_lambdas',reg_lambda_WP=reg_lambda,reg_type_wp=reg_type_wp,plotting=False)
    ##
    #run_type = 'h_add'
    #run_type = 'h_gabor'
    #N_train, N_test = 16, 2025 ## 4**2, 45**2
    #N_train, N_test = 16, 5041 ## 4**2, 71**2
    #save_data_set_mdl_sgd(path='./data/{}', run_type=run_type, lb=-1,ub=1,N_train=N_train,N_test=N_test,msg='',visualize=True)
    print('End')
    print('\a')
