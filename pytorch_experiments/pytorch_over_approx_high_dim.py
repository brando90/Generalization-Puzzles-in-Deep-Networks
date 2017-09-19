import time
import numpy as np
import sys

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

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy

#TODO make dtype, DTYPE accross all script

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
    data_generator, D_layers, act = main(run_type=run_type)
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

def stats_logger(mdl, data, loss_list,func_diff,grad_list, i,logging_freq):
    N_train,_ = data.X_train.shape
    N_test,_ = data.X_test.shape
    if i % logging_freq == 0 or i == 0:
        # log current train loss
        current_train_loss = (1/N_train)*(mdl.forward(data.X_train) - data.Y_train).pow(2).sum().data.numpy()
        loss_list.append(current_train_loss)
        ## generalization/func diff
        y_test_sgd = mdl_sgd.forward(data.X_test)
        y_test_pinv = Variable( torch.FloatTensor( np.dot( data.Kern_test.data.numpy(), c_pinv) ) )
        func_diff.append( (1/N_test)*(y_test_sgd - y_test_pinv).pow(2).sum() )
        ## collect param stats
        for index, W in enumerate(mdl_sgd.parameters()):
            delta = eta*W.grad.data
            grad_list[index].append( W.grad.data.norm(2) )
            if is_NaN(W.grad.data.norm(2)) or is_NaN(current_train_loss):
                print('\n----------------- ERROR HAPPENED')
                print('error happened at: i = {} current_train_loss: {}, grad_norm: {},\n ----------------- \a'.format(i,current_train_loss,W.grad.data.norm(2)))
                sys.exit()

def train_SGD(mdl,data, M,eta,nb_iter,A ,logging_freq ,dtype):
    '''
    '''
    nb_module_params = len( list(mdl.parameters()) )
    loss_list, grad_list =  [], [ [] for i in range(nb_module_params) ]
    func_diff = []
    ##
    N_train, _ = tuple( data.X_train.size() )
    for i in range(nb_iter):
        # Forward pass: compute predicted Y using operations on Variables
        batch_xs, batch_ys = get_batch2(data.X_train,data.Y_train,M,dtype) # [M, D], [M, 1]
        ## FORWARD PASS
        y_pred = mdl.forward(batch_xs)
        ## LOSS
        batch_loss = (1/N_train)*(y_pred - batch_ys).pow(2).sum()
        ## BACKARD PASS
        batch_loss.backward() # Use autograd to compute the backward pass. Now w will have gradients
        ## SGD update
        for W in mdl.parameters():
            delta = eta*W.grad.data
            W.data.copy_(W.data - delta) # W - eta*g + A*gdl_eps
        ## stats logger
        stats_logger(mdl, data, loss_list,func_diff,grad_list, i,logging_freq)
        ## train stats
        if i % (nb_iter/10) == 0 or i == 0:
            current_train_loss = (1/N_train)*(mdl.forward(data.X_train) - data.Y_train).pow(2).sum().data.numpy()
            print('i = {}, current_loss = {}'.format(i, current_train_loss ) )
        ## Manually zero the gradients after updating weights
        mdl.zero_grad()
    return loss_list_standard_sgd,grad_list_standard_sgd,func_diff_standard_sgd,nb_module_params

def main(**kwargs):
    '''
    main code, where experiments for over parametrization are made
    '''
    ##dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
    dtype = torch.FloatTensor
    #
    debug = True
    debug_sgd = False
    #debug_sgd = True
    ## Hyper Params SGD weight parametrization
    M = 3
    eta = 0.02 # eta = 1e-6
    nb_iter = int(20*10)
    A = 0.0
    logging_freq = 500
    ## Hyper Params SGD standard parametrization
    M_standard_sgd = 3
    eta_standard_sgd = 0.02 # eta = 1e-6
    nb_iter_standard_sgd = int(20*10)
    A_standard_sgd = 0.0
    logging_freq_standard_sgd = 100
    ##
    ## activation params
    # alb, aub = -100, 100
    # aN = 100
    adegree = 2
    ax = np.concatenate( (np.linspace(-20,20,100), np.linspace(-10,10,1000)) )
    aX = np.concatenate( (ax,np.linspace(-2,2,100000)) )
    ## activation funcs
    #act = quadratic
    act, c_pinv_relu = get_relu_poly_act2(aX,degree=adegree) # ax**2+bx+c, #[1, x^1, ..., x^D]
    #act = get_relu_poly_act(degree=adegree,lb=alb,ub=aub,N=aN) # ax**2+bx+c
    #act = relu
    ## plot activation
    palb, paub = -20, 20
    paN = 1000
    #print('Plotting activation function')
    #plot_activation_func(act,lb=palb,ub=paub,N=paN)
    #plt.show()
    #### 2-layered mdl
    D0 = 2

    # H1 = 10
    # D0,D1,D2 = 1,H1,1
    # D_layers,act = [D0,D1,D2], act

    # H1,H2 = 5,5
    # D0,D1,D2,D3 = D0,H1,H2,1
    # D_layers,act = [D0,D1,D2,D3], act

    H1,H2,H3 = 5,5,5
    D0,D1,D2,D3,D4 = D0,H1,H2,H3,1
    D_layers,act = [D0,D1,D2,D3,D4], act

    # H1,H2,H3,H4 = 5,5,5,5
    # D0,D1,D2,D3,D4,D5 = D0,H1,H2,H3,H4,1
    # D_layers,act = [D0,D1,D2,D3,D4,D5], act

    bias = True
    #pdb.set_trace()
    start_time = time.time()
    ##
    np.set_printoptions(suppress=True)
    lb, ub = -1, 1
    ## mdl degree and D
    Degree_mdl = adegree**( len(D_layers)-2 )
    #Degree_mdl = 25
    D_sgd = Degree_mdl+1
    D_pinv = Degree_mdl+1
    D_rls = D_pinv
    # RLS
    lambda_rls = 0.001
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
    #### Get Data set
    if kwargs: # empty dictionaries evluate to false
        # only executes this if kwargs dict is NOT empty
        run_type = kwargs['run_type']
        #collect_functional_diffs = kwargs['collect_functional_diffs']
    else:
        #run_type = 'sine'
        #run_type = 'similar_nn'
        run_type = 'from_file'
        #run_type = 'h_add'
    data_filename = None
    init_config_data = Maps({})
    f_true = None
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
        data_filename = 'data_numpy_D_layers_[2, 5, 5, 1]_nb_layers4_N_train_16_N_test_5041_lb_-1_ub_1_act_h_add_run_type_poly_act_degree2_msg_.npz'
        ##
        data = np.load( './data/{}'.format(data_filename) )
        X_train, Y_train = data['X_train'], data['Y_train']
        X_test, Y_test = data['X_test'], data['Y_test']
        D_data = D0
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
    ## get nb data points
    N_train,_ = X_train.shape
    N_test,_ = X_test.shape
    print('N_train = {}, N_test = {}'.format(N_train,N_test))
    ## Lift data/Kernelize data
    poly_feat = PolynomialFeatures(D_pinv)
    Kern_train = poly_feat.fit_transform(X_train)
    Kern_test = poly_feat.fit_transform(X_test)
    ## LA models
    c_pinv = np.dot(np.linalg.pinv( Kern_train ),Y_train) # [D_pinv,1]
    ## inits
    init_config = Maps( {'w_init':'w_init_normal','mu':0.0,'std':1.0, 'bias_init':'b_fill','bias_value':0.01,'bias':bias ,'nb_layers':len(D_layers)} )
    w_inits_sgd, b_inits_sgd = get_initialization(init_config)
    init_config_standard_sgd = Maps( {'mu':0.0,'std':0.001, 'bias_value':0.01} )
    mdl_stand_initializer = lambda mdl: lifted_initializer(mdl,init_config_standard_sgd)
    ## SGD models
    mdl_sgd = NN(D_layers=D_layers,act=act,w_inits=w_inits_sgd,b_inits=b_inits_sgd,bias=bias)
    mdl_standard_sgd = get_sequential_lifted_mdl(nb_monomials=c_pinv.shape[0],D_out=1, bias=False)
    ## data to TORCH
    data = get_data_struct(X_train,Y_train,X_test,Y_test,Kern_train,Kern_test,dtype)
    data_stand = get_data_struct(X_train,Y_train,X_test,Y_test,Kern_train,Kern_test,dtype)
    data_stand.X_train, data_stand.X_test = data_stand.Kern_train, data_stand.Kern_test
    ## DEBUG PRINTs
    print('>>norm(Y): ', ((1/N_train)*torch.norm(data.Y_train)**2).data.numpy()[0] )
    print('>>l2_loss_torch: ', (1/N_train)*( data.Y_train - mdl_sgd.forward(data.X_train)).pow(2).sum().data.numpy()[0] )
    ## check number of monomials
    nb_monomials = int(scipy.misc.comb(D0+D_pinv,D_pinv))
    print(' c_pinv.shape[0]={} \n nb_monomials={} '.format( c_pinv.shape[0], nb_monomials ))
    if c_pinv.shape[0] != int(scipy.misc.comb(D0+D_pinv,D_pinv)):
       raise ValueError('nb of monomials dont match D0={},D_pinv={}, number of monimials fron pinv={}, number of monomials analyticall = {}'.format( D0,D_pinv,c_pinv.shape[0],int(scipy.misc.comb(D0+D_pinv,D_pinv)) )    )
    ########################################################################################################################################################
    loss_list_weight_sgd,grad_list_weight_sgd,func_diff_weight_sgd,nb_module_params = train_SGD(
        mdl_sgd,data, M,eta,nb_iter,A ,logging_freq ,dtype
    )
    #print('training ended!\a')
    loss_list_standard_sgd,grad_list_standard_sgd,func_diff_standard_sgd,nb_module_params = train_SGD(
        mdl_standard_sgd,data_stand, M_standard_sgd,eta_standard_sgd,nb_iter_standard_sgd,A_standard_sgd ,logging_freq ,dtype
    )
    print('training ended!\a')
    ########################################################################################################################################################
    ## SGD pNN
    nb_params = count_params(mdl_sgd)
    ## Do SYMPY magic
    if len(D_layers) <= 2:
        c_sgd = list(mdl_sgd.parameters())[0].data.numpy()
        c_sgd = c_sgd.transpose()
    else:
        # e.g. x = Matrix(2,1,[a,a])
        x_list = [ symbols('x'+str(i)) for i in range(D0) ]
        x = Matrix(D0,1,x_list)
        tmdl = mdl_sgd
        if act.__name__ == 'poly_act_degree{}'.format(adegree):
            sact = lambda x: s_Poly(x,c_pinv_relu)
            sact.__name__ = 'spoly_act_degree{}'.format(adegree)
            if adegree >= 10:
                sact = sQuad
        elif act__name__ == 'quadratic':
            sact = sQuad
        elif act.__name__ == 'relu':
            sact = sReLU
        smdl = sNN(sact,mdl=tmdl)
        ## get simplification
        expr = smdl.forward(x)
        s_expr = poly(expr,x_list)
        c_sgd = np.array( s_expr.coeffs()[::-1] )
        c_sgd = [ np.float64(num) for num in c_sgd]
    if debug:
        print('c_sgd_standard = ', mdl_standard_sgd[0].weight)
        print('c_sgd_weight = ', c_sgd)
        print('c_pinv: ', c_pinv)
        print('data.X_train = ', data.X_train)
        print('data.Y_train = ', data.Y_train)
        print(mdl_sgd)
        if len(D_layers) > 2:
            print('\n---- structured poly: {}'.format(str(s_expr)) )
    ## Stats of models
    print('---- Stats of flattened to Poly models')
    print('\n----> Data set stats:\n data_filename= {}, run_type={}, init_config_data={}\n'.format(data_filename,run_type,init_config_data) )
    print('---- Learning params')
    print('Degree_mdl = {}, N_train = {}, M = {}, eta = {}, nb_iter = {} nb_params={},D_layers={}'.format(Degree_mdl,N_train,M,eta,nb_iter,nb_params,D_layers))
    print('Activations: act={}, sact={}'.format(act.__name__,sact.__name__) )
    print('init_config: ', init_config)
    print('number of layers = {}'.format(nb_module_params))
    ## Parameter Norms
    if len(D_layers) >= 2:
        print('\n---- statistics about learned params')
        print('--L1')
        print('||c_pinv||_1 = {} '.format(np.linalg.norm(c_pinv,1)) )
        print('||c_sgd_weight||_1 = {} '.format(np.linalg.norm(c_sgd,1)) )
        print('||c_sgd_stand||_1 = {} '.format(np.linalg.norm(mdl_standard_sgd[0].weight.data.numpy(),1)) )
        print('--L2')
        print('||c_pinv||_2 = ', np.linalg.norm(c_pinv,2))
        print('||c_sgd_weight||_2 = ', np.linalg.norm(c_sgd,2))
        print('||c_sgd_stand||_2 = {} '.format(np.linalg.norm(mdl_standard_sgd[0].weight.data.numpy(),2)) )
    ## Parameter Difference
    print('---- parameters differences')
    if len(D_layers) >= 2:
        #print('||c_sgd - c_pinv||_2 = ', np.linalg.norm(c_sgd - c_pinv,2))
        #print('||c_sgd - c_avg||_2 = ', np.linalg.norm(c_sgd - c_pinv,2))
        #print('||c_avg - c_pinv||_2 = ', np.linalg.norm(c_avg - c_pinv,2))
        pass
    ## Generalization L2 (functional) norm difference
    print('-- Generalization L2 (functional) norm difference')
    y_test_pinv = Variable( torch.FloatTensor( np.dot( Kern_test, c_pinv) ) )
    y_test_sgd = mdl_sgd.forward(data.X_test)
    loss = (1/N_test)*(y_test_sgd - y_test_pinv).pow(2).sum()
    print('J_Gen((f_sgd(x) - f_pinv(x))^2) = 1/{}sum (f_sgd(x) - f_pinv(x))^2 = {}'.format( N_test, loss.data.numpy()[0] ) )
    ## FUNCTIONAL DIFF or GENERALIZATION DIFF
    print('-- Generalization l2 error')
    if f_true ==  None:
        print('J_gen(f_sgd)_standard = ', (1/N_test)*(mdl_standard_sgd.forward(data.Kern_test) - Variable(torch.FloatTensor(Y_test)) ).pow(2).sum().data.numpy() )
        print('J_gen(f_sgd)_weight = ', (1/N_test)*(mdl_sgd.forward(data.X_test) - Variable(torch.FloatTensor(Y_test)) ).pow(2).sum().data.numpy() )
        print('J_gen(f_pinv) = ', (1/N_test)*(np.linalg.norm(Y_test-np.dot( poly_feat.fit_transform(X_test),c_pinv))**2) )
    else:
        f_sgd = lambda x: f_mdl_eval(x,mdl_sgd,dtype)
        f_pinv = lambda x: f_mdl_LA(x,c_pinv)
        print('||f_sgd - f_true||^2_2 = ', L2_norm_2(f=f_sgd,g=f_true,lb=lb,ub=ub))
        print('||f_pinv - f_true||^2_2 = ', L2_norm_2(f=f_pinv,g=f_true,lb=lb,ub=ub))
    ## TRAIN ERRORS of mdls
    print('-- Train Error')
    print(' J(f_sgd)_standard = ', (1/N_train)*(mdl_standard_sgd.forward( data.Kern_train ) - data.Y_train ).pow(2).sum().data.numpy() )
    print(' J(f_sgd)_weight = ', (1/N_train)*(mdl_sgd.forward(data.X_train) - data.Y_train).pow(2).sum().data.numpy() )
    print(' J(f_pinv) = ',(1/N_train)*(np.linalg.norm(data.Y_train.data.numpy()-np.dot( poly_feat.fit_transform(data.X_train.data.numpy()) ,c_pinv))**2) )
    #print(' J(c_rls) = ',(1/N)*(np.linalg.norm(Y-(1/N)*(np.linalg.norm(Y-np.dot( poly_kernel_matrix( x_true,D_sgd-1 ),c_rls))**2) )**2) )
    ## REPORT TIMES
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    print("--- %s seconds ---" % seconds )
    print("--- %s minutes ---" % minutes )
    print("--- %s hours ---" % hours )
    print('\a')
    ## plots
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
    else:
        #
        nb_non_linear_layers = len(D_layers)-2
        degree_sgd = adegree**(len(D_layers)-2)
        sgd_legend_str = 'Degree model={} non linear-layers={}'.format(degree_sgd,nb_non_linear_layers)
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
            'minimum norm solution Degree model={}, number of monomials={}'.format(str(D_pinv-1),nb_monomials),
            'data points, number of data points = {}'.format(N_train)])
        ## FIG SGD weight parametrization
        fig2 = plt.figure()
        ax2 = Axes3D(fig2)
        data_pts = ax2.scatter(Xp_train,Yp_train,Zp_train, marker='D')
        surf = ax2.plot_surface(Xp,Yp,Zp_sgd,cmap=cm.coolwarm)
        ax2.set_xlabel('x1'),ax2.set_ylabel('x2'),ax2.set_zlabel('f(x)')
        surf_proxy = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker = '_')
        ax2.legend([surf_proxy,data_pts],[
            'SGD solution weight parametrization {}, number of monomials={}, param count={}, batch-size={}, iterations={}, step size={}'.format(sgd_legend_str,nb_monomials,nb_params,M,nb_iter,eta),
            'data points, number of data points = {}'.format(N_train)])
        ## FIG SGD standard param
        fig = plt.figure()
        ax3 = Axes3D(fig)
        data_pts = ax3.scatter(Xp_train,Yp_train,Zp_train, marker='D')
        surf = ax3.plot_surface(Xp,Yp,Zp_sgd_stand, cmap=cm.coolwarm)
        ax3.set_xlabel('x1'),ax3.set_ylabel('x2'),ax3.set_zlabel('f(x)')
        ax3.legend([surf_proxy,data_pts],[
            'SGD solution standard parametrization {}, number of monomials={}, param count={}, batch-size={}, iterations={}, step size={}'.format(sgd_legend_str,nb_monomials,nb_monomials,M_standard_sgd,nb_iter_standard_sgd,eta_standard_sgd),
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
    ##
    loss_list = loss_list_standard_sgd
    fig1 = plt.figure()
    p_loss, = plt.plot(np.arange(len(loss_list)), loss_list,color='m')
    plt.legend([p_loss],['plot loss'])
    plt.title('Loss vs Iterations')
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
    #loss_list_weight_sgd,grad_list_weight_sgd,func_diff_weight_sgd,nb_module_params
    fig = plt.figure()
    p_func_diff_standard, = plt.plot(np.arange(len(func_diff_standard_sgd)), func_diff_standard_sgd,color='g')
    p_func_diff_weight, = plt.plot(np.arange(len(func_diff_weight_sgd)), func_diff_weight_sgd,color='g')
    plt.legend([p_func_diff_standard,p_func_diff_weight],
        [' L2 generalization distance: weight parametrization with SGD minus minimum norm solution, number test points = {}'.format(N_test),
        ' L2 generalization distance: standard parametrization with SGD minus minimum norm solution, number test points = {}'.format(N_test)]
        )
    plt.title('Generalization L2 difference between minimum norm and SGD functions')
    ##
    plt.show()
    ## is kwargs empty? If yes then execute if statement
    if kwargs: # if dictionary is empty, note empty dictionaries evaluate to false, so not false gives true
        return mdl_sgd, D_layers, act

if __name__ == '__main__':
    print('main started')
    main()
    #N_train, N_test = 16, 2025 ## 4**2, 45**2
    #N_train, N_test = 16, 5041 ## 4**2, 71**2
    #N_train, N_test = 16, 10000 ## 4**2, 100**2
    #save_data_set_mdl_sgd(path='./data/{}', run_type='h_add', lb=-1,ub=1,N_train=N_train,N_test=N_test,msg='',visualize=True)
    #print('End')
    print('\a')
