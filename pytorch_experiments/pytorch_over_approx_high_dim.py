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

def f_mdl_LA(x,c):
    D,_ = c.shape
    X = poly_kernel_matrix( [x],D-1 )
    # np.dot(poly_kernel_matrix( [x], c.shape[0]-1 ),c)
    return np.dot(X,c)

def f_mdl_eval(x,mdl_eval,dtype):
    _,D = list(mdl_eval.parameters())[0].data.numpy().shape
    #pdb.set_trace()
    if len(list(mdl_eval.parameters())) == 2 or len(list(mdl_eval.parameters())) == 1:
        x = poly_kernel_matrix( [x],D-1 )
        x = Variable(torch.FloatTensor([x]).type(dtype))
    else:
        x = Variable(torch.FloatTensor([x]).type(dtype))
        x = x.view(1,1)
    y_pred = mdl_eval.forward(x)
    return y_pred.data.numpy()

def L2_norm_2(f,g,lb=0,ub=1):
    f_g_2 = lambda x: (f(x) - g(x))**2
    #import scipy
    #import scipy.integrate as integrate
    #import scipy.integrate.quad as quad
    #pdb.set_trace()
    result = integrate.quad(func=f_g_2, a=lb,b=ub)
    #result = quad(func=f_g_2, a=lb,b=ub)
    integral_val = result[0]
    return integral_val

def get_RLS_soln( X,Y,lambda_rls):
    N,D = X.shape
    XX_lI = np.dot(X.transpose(),X) + lambda_rls*N*np.identity(D)
    w = np.dot( np.dot( np.linalg.inv(XX_lI), X.transpose() ), Y)
    return w

def index_batch(X,batch_indices,dtype):
    if len(X.shape) == 1: # i.e. dimension (M,) just a vector
        batch_xs = torch.FloatTensor(X[batch_indices]).type(dtype)
    else:
        batch_xs = torch.FloatTensor(X[batch_indices,:]).type(dtype)
    return batch_xs

def get_batch2(X,Y,M,dtype):
    # TODO fix and make it nicer
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

def main(argv=None):
    dtype = torch.FloatTensor
    #
    debug = True
    debug_sgd = False
    ## sgd
    M = 8
    eta = 0.00003 # eta = 1e-6
    A = 0.0
    nb_iter = int(100*1000)
    logging_freq = 500
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

    # dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
    #pdb.set_trace()
    start_time = time.time()
    ##
    np.set_printoptions(suppress=True)
    lb, ub = -1, 1
    ## true facts of the data set
    N = 30
    ## mdl degree and D
    Degree_mdl = adegree**( len(D_layers)-2 )
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
    ##
    init_config = Maps( {'w_init':'w_init_normal','mu':0.0,'std':0.01, 'bias_init':'b_fill','bias_value':0.01,'bias':bias ,'nb_layers':len(D_layers)} )
    #init_config = Maps( {'w_init':'xavier_normal','gain':1,'bias_init':'b_fill','bias_value':0.01,'bias':bias,'nb_layers':len(D_layers)})
    w_inits_sgd, b_inits_sgd = get_initialization(init_config)
    #### Get Data set
    ## Get input variables X
    #run_type = 'sine'
    #run_type = 'similar_nn'
    #run_type = 'from_file'
    run_type = 'h_add'
    data_filename = None
    init_config_data = Maps({})
    f_true = None
    if run_type == 'sine':
        x_true = np.linspace(lb,ub,N) # the real data points
        Y = np.sin(2*np.pi*x_true)
        f_true = lambda x: np.sin(2*np.pi*x)
    elif run_type == 'similar_nn':
        pass
        ## Get data values from some net itself
        # x_true = np.linspace(lb,ub,N)
        # x_true.shape = x_true.shape[0],D
        # #
        # init_config_data = Maps( {'w_init':'w_init_normal','mu':0.0,'std':2.0, 'bias_init':'b_fill','bias_value':0.1,'bias':bias ,'nb_layers':len(D_layers)} )
        # w_inits_data, b_inits_data = get_initialization(init_config_data)
        # data_generator = NN(D_layers=D_layers,act=act,w_inits=w_inits_data,b_inits=b_inits_data,bias=bias)
        # Y = get_Y_from_new_net(data_generator=data_generator, X=x_true,dtype=dtype)
        # f_true = lambda x: f_mdl_eval(x,data_generator,dtype)
    elif run_type == 'from_file':
        ##
        #data_filename = 'data_numpy_D_layers_[1, 2, 2, 2, 1]_nb_layers5_biasTrue_mu0.0_std2.0_N_train_10_N_test_1000_lb_-1_ub_1_act_quad_ax2_bx_c_msg_.npz'
        data_filename = 'data_numpy_D_layers_[2, 3, 3, 1]_nb_layers4_biasTrue_mu0.0_std2.0_N_train_10_N_test_1000_lb_-1_ub_1_act_quadratic_msg_.npz'
        ##
        data = np.load( './data/{}'.format(data_filename) )
        X_train, Y_train = data['X_train'], data['Y_train']
        X_test, Y_test = data['X_test'], data['Y_test']
        D_data = D0
    elif run_type == 'h_add':
        X,Y,Z = generate_meshgrid_h_add(N=N,start_val=-1,end_val=1)
        X_train,Y_train = make_mesh_grid_to_data_set(X,Y,Z)
        X,Y,Z = generate_meshgrid_h_add(N=1000,start_val=-1,end_val=1)
        X_test,Y_test = make_mesh_grid_to_data_set(X,Y,Z)
        D_data = D0
    ## LA models
    poly_feat = PolynomialFeatures(D_pinv)
    Kern = poly_feat.fit_transform(X_train)
    c_pinv = np.dot(np.linalg.pinv( Kern ),Y_train) # [D_pinv,1]
    nb_monomials = int(scipy.misc.comb(D_data+D_pinv,D_pinv))
    print(' c_pinv.shape[0]={} \n nb_monomials={} '.format( c_pinv.shape[0], nb_monomials ))
    if c_pinv.shape[0] != int(scipy.misc.comb(D_data+D_pinv,D_pinv)):
        raise ValueError('nb of monomials dont match')
    #pdb.set_trace()
    #c_rls = get_RLS_soln(Kern,Y_train,lambda_rls) # [D_pinv,1]
    ## data to TORCH
    print('len(D_layers) ', len(D_layers))
    X = Variable(torch.FloatTensor(X_train).type(dtype), requires_grad=False)
    Y = Variable(torch.FloatTensor(Y_train).type(dtype), requires_grad=False)
    ## SGD model
    mdl_sgd = NN(D_layers=D_layers,act=act,w_inits=w_inits_sgd,b_inits=b_inits_sgd,bias=bias)
    nb_module_params = len( list(mdl_sgd.parameters()) )
    loss_list, grad_list =  [], [ [] for i in range(nb_module_params) ]
    func_diff = []
    print('>>norm(Y): ', ((1/N)*torch.norm(Y)**2).data.numpy()[0] )
    print('>>l2_loss_torch: ', (1/N)*( Y - mdl_sgd.forward(X)).pow(2).sum().data.numpy()[0] )
    ########################################################################################################################################################
    for i in range(nb_iter):
        # Forward pass: compute predicted Y using operations on Variables
        batch_xs, batch_ys = get_batch2(X,Y,M,dtype) # [M, D], [M, 1]
        ## FORWARD PASS
        y_pred = mdl_sgd.forward(batch_xs)
        ## LOSS
        loss = (1/N)*(y_pred - batch_ys).pow(2).sum()
        ## BACKARD PASS
        loss.backward() # Use autograd to compute the backward pass. Now w will have gradients
        ## SGD update
        for W in mdl_sgd.parameters():
            gdl_eps = torch.randn(W.data.size()).type(dtype)
            #clip=0.001
            #torch.nn.utils.clip_grad_norm(mdl_sgd.parameters(),clip)
            #delta = torch.clamp(eta*W.grad.data,min=-clip,max=clip)
            #print(delta)
            #W.data.copy_(W.data - delta + A*gdl_eps)
            delta = eta*W.grad.data
            W.data.copy_(W.data - delta + A*gdl_eps) # W - eta*g + A*gdl_eps
        #pdb.set_trace()
        ## TRAINING STATS
        if i % logging_freq == 0 or i == 0:
            current_loss = loss.data.numpy()[0]
            loss_list.append(current_loss)

            #func_diff.append( )
            if debug_sgd:
                print('\ni =',i)
                print('current_loss = ',current_loss)
            for index, W in enumerate(mdl_sgd.parameters()):
                grad_norm = W.grad.data.norm(2)
                delta = eta*W.grad.data
                grad_list[index].append( W.grad.data.norm(2) )
                if debug_sgd:
                    print('-------------')
                    print('-> grad_norm: ',grad_norm)
                    #print('----> eta*grad_norm: ',eta*grad_norm)
                    print('------> delta: ', delta.norm(2))
                    #print(delta)
                if is_NaN(grad_norm) or is_NaN(current_loss):
                    print('\n----------------- ERROR HAPPENED')
                    print('loss: {}'.format(current_loss) )
                    print('error happened at: i = {}'.format(i))
                    print('current_loss: {}, grad_norm: {},\n -----------------'.format(current_loss,grad_norm) )
                    #print('grad_list: ', grad_list)
                    print('\a')
                    sys.exit()
        ##
        if i % (nb_iter/4) == 0 or i == 0:
            current_loss = loss.data.numpy()[0]
            print('\ni = {}, current_loss = {}'.format(i,current_loss) )
        ## Manually zero the gradients after updating weights
        mdl_sgd.zero_grad()
        ## COLLECT MOVING AVERAGES
        # for i in range(len(Ws)):
        #     W, W_avg = Ws[i], W_avgs[i]
        #     W_avgs[i] = (1/nb_iter)*W + W_avg
    ########################################################################################################################################################
    #pdb.set_trace()
    print('\ni = {}, current_loss = {}'.format(i,current_loss) )
    print('training ended!\a')
    ##
    nb_params = count_params(mdl_sgd)
    X, Y = X.data.numpy(), Y.data.numpy()
    #
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
        #pdb.set_trace()
        s_expr = poly(expr,x_list)
        c_sgd = np.array( s_expr.coeffs()[::-1] )
        c_sgd = [ np.float64(num) for num in c_sgd]
        #pdb.set_trace()
    if debug:
        print('c_sgd = ', c_sgd)
        print('c_pinv: ', c_pinv)
        print('X = ', X)
        print('Y = ', Y)
        print(mdl_sgd)
        if len(D_layers) > 2:
            print('\n---- structured poly: {}'.format(str(s_expr)) )
    #
    print('\n----> Data set stats:\n data_filename= {}, run_type={}, init_config_data={}\n'.format(data_filename,run_type,init_config_data) )
    print('---- Learning params')
    print('Degree_mdl = {}, N = {}, M = {}, eta = {}, nb_iter = {} nb_params={},D_layers={}'.format(Degree_mdl,N,M,eta,nb_iter,nb_params,D_layers))
    print('Activations: act={}, sact={}'.format(act.__name__,sact.__name__) )
    print('init_config: ', init_config)
    print('number of layers = {}'.format(nb_module_params))
    #
    print('---- Stats of flattened to Poly models')
    print('c_pinv.shape', c_pinv.shape)
    #print('c_sgd.shape', c_pinv.shape)
    #
    if len(D_layers) >= 2:
        print('\n---- statistics about learned params')
        print('--L1')
        print('||c_pinv||_1 = {} '.format(np.linalg.norm(c_pinv,1)) )
        #print('||c_avg||_1 = {} '.format(np.linalg.norm(c_avg,1)) )
        print('||c_sgd||_1 = {} '.format(np.linalg.norm(c_sgd,1)) )
        print('--L2')
        print('||c_pinv||_2 = ', np.linalg.norm(c_pinv,2))
        #print('||c_avg||_2 = {} '.format(np.linalg.norm(c_avg,2))
        print('||c_sgd||_2 = ', np.linalg.norm(c_sgd,2))
    print('---- parameters differences')
    if len(D_layers) >= 2:
        #print('||c_sgd - c_pinv||_2 = ', np.linalg.norm(c_sgd - c_pinv,2))
        #print('||c_sgd - c_avg||_2 = ', np.linalg.norm(c_sgd - c_pinv,2))
        #print('||c_avg - c_pinv||_2 = ', np.linalg.norm(c_avg - c_pinv,2))
        pass
    f_sgd = lambda x: f_mdl_eval(x,mdl_sgd,dtype)
    #f_sgd = lambda x: np.dot(poly_kernel_matrix( [x], c_sgd.shape[0]-1 ),c_sgd)
    f_pinv = lambda x: f_mdl_LA(x,c_pinv)
    print('-- functional L2 norm difference')
    #pdb.set_trace()
    print('||f_sgd - f_pinv||^2_2 = ', L2_norm_2(f=f_sgd,g=f_pinv,lb=lb,ub=ub))
    #print('||f_avg - f_pinv||^2_2 = ', L2_norm_2(f=f_avg,g=f_pinv,lb=0,ub=1))
    print('-- Generalization (error vs true curve) functional l2 norm')
    if f_true ==  None:
        print('J_gen(f_sgd) = ', (1/N)*(mdl_sgd.forward(Variable(torch.FloatTensor(X_test))) - Variable(torch.FloatTensor(Y_test)) ).pow(2).sum().data.numpy() )
        print('J_gen(f_pinv) = ', (1/N)*(np.linalg.norm(Y_test-np.dot( poly_feat.fit_transform(X_test),c_pinv))**2) )
    else:
        print('||f_sgd - f_true||^2_2 = ', L2_norm_2(f=f_sgd,g=f_true,lb=lb,ub=ub))
        print('||f_pinv - f_true||^2_2 = ', L2_norm_2(f=f_pinv,g=f_true,lb=lb,ub=ub))
    #
    print('-- Train Error')
    print(' J(f_sgd) = ', (1/N)*(mdl_sgd.forward(Variable(torch.FloatTensor(X))) - Variable(torch.FloatTensor(Y)) ).pow(2).sum().data.numpy() )
    print(' J(f_pinv) = ',(1/N)*(np.linalg.norm(Y-np.dot( poly_feat.fit_transform(X_train) ,c_pinv))**2) )
    #print(' J(c_rls) = ',(1/N)*(np.linalg.norm(Y-(1/N)*(np.linalg.norm(Y-np.dot( poly_kernel_matrix( x_true,D_sgd-1 ),c_rls))**2) )**2) )
    #
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    print("--- %s seconds ---" % seconds )
    print("--- %s minutes ---" % minutes )
    print("--- %s hours ---" % hours )
    print('\a')
    ## plots
    if D0 == 1:
        x_horizontal = np.linspace(lb,ub,1000)
        X_plot = poly_kernel_matrix(x_horizontal,D_sgd-1)
        #plots objs
        p_sgd, = plt.plot(x_horizontal, [ float(f_sgd(x_i)[0]) for x_i in x_horizontal ])
        p_pinv, = plt.plot(x_horizontal, np.dot(X_plot,c_pinv))
        p_data, = plt.plot(x_true,Y,'ro')
        p_list = [p_sgd,p_pinv,p_data]
        plt.title('SGD vs minimum norm solution curves')
        if len(p_list) == 3:
            if len(D_layers) <= 2:
                sgd_legend_str = 'Degree model={} non linear-layers={}'.format(str(D_sgd-1),1)
            else:
                nb_non_linear_layers = len(D_layers)-2
                degree_sgd = adegree**(len(D_layers)-2)
                sgd_legend_str = 'Degree model={} non linear-layers={}'.format(degree_sgd,nb_non_linear_layers)
            plt.legend(p_list,['SGD solution {}, param count={}, batch-size={}, iterations={}, step size={}'.format(
                sgd_legend_str,nb_params,M,nb_iter,eta),
                'minimum norm solution Degree model='+str(D_pinv-1),
                'data points'])
            plt.ylabel('f(x)')
        else:
            plt.legend(p_list,['sgd curve Degree_mdl={}, batch-size= {}, iterations={}, step size={}'.format(
            str(D_sgd-1),M,nb_iter,eta),'min norm (pinv) Degree_mdl='+str(D_pinv-1), 'data points'])
        #plt.legend(p_list,['average sgd model Degree_mdl={}'.format( str(D_sgd-1) ),'sgd curve Degree_mdl={}, batch-size= {}, iterations={}, eta={}'.format(str(D_sgd-1),M,nb_iter,eta),'min norm (pinv) Degree_mdl='+str(D_pinv-1),'data points'])
        #plt.legend(p_list,['min norm (pinv) Degree_mdl='+str(D_pinv-1),'data points'])
        plt.ylabel('f(x)')
        ##
        fig1 = plt.figure()
        p_loss, = plt.plot(np.arange(len(loss_list)), loss_list,color='m')
        plt.legend([p_loss],['plot loss'])
        plt.title('Loss vs Iterations')
        ##
        for i in range(len(grad_list)):
            fig2 = plt.figure()
            current_grad_list = grad_list[i]
            #pdb.set_trace()
            p_grads, = plt.plot(np.arange(len(current_grad_list)), current_grad_list,color='g')
            plt.legend([p_grads],['plot grads'])
            plt.title('Gradient vs Iterations: # {}'.format(i))
        ##
        plot_activation_func(act)
        ##
    else:
        #
        nb_non_linear_layers = len(D_layers)-2
        degree_sgd = adegree**(len(D_layers)-2)
        sgd_legend_str = 'Degree model={} non linear-layers={}'.format(degree_sgd,nb_non_linear_layers)
        #
        X_data, Y_data = X_test,Y_test
        Xp,Yp,Zp = make_meshgrid_data_from_training_data(X_data=X_data, Y_data=Y_data) # meshgrid for visualization
        #visualize(Xp,Yp,Zp,title_name='Test function') # visualize function
        Xp_train,Yp_train,Zp_train = make_meshgrid_data_from_training_data(X_data=X_train, Y_data=Y_train) # meshgrid for trainign points
        #pdb.set_trace()
        ## plot data PINV
        Y_pinv = np.dot(poly_feat.fit_transform(X_data),c_pinv)
        _,_,Zp_pinv = make_meshgrid_data_from_training_data(X_data=X_data, Y_data=Y_pinv)
        ## plot data SGD
        Y_sgd = mdl_sgd.forward(Variable(torch.FloatTensor(X_data))).data.numpy()
        _,_,Zp_sgd = make_meshgrid_data_from_training_data(X_data=X_data, Y_data=Y_sgd)

        ## FIG PINV
        fig1 = plt.figure()
        ax1 = Axes3D(fig1)
        ## plot data points fig1
        points_scatter = ax1.scatter(Xp_train,Yp_train,Zp_train, marker='D')
        ## surf PINV
        surf = ax1.plot_surface(Xp,Yp,Zp_pinv,color='y',cmap=cm.coolwarm)
        ##
        ax1.set_xlabel('x1'),ax1.set_ylabel('x2'),ax1.set_zlabel('f(x)')
        surf_proxy = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker ='_')
        ax1.legend([surf_proxy,points_scatter],[
            'minimum norm solution Degree model={}, number of monomials={}'.format(str(D_pinv-1),nb_monomials),
            'data points'])
        #plt.title('minimum norm solution')
        #minimum norm solution Degree model='+str(D_pinv-1)
        # ## FIG SGD
        fig2 = plt.figure()
        ax2 = Axes3D(fig2)
        ## plot data points fig1
        data_pts = ax2.scatter(Xp_train,Yp_train,Zp_train, marker='D')
        ## surf SGD
        surf = ax2.plot_surface(Xp,Yp,Zp_sgd,cmap=cm.coolwarm)
        ##
        ax2.set_xlabel('x1'),ax2.set_ylabel('x2'),ax2.set_zlabel('f(x)')
        surf_proxy = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker = '_')
        ax2.legend([surf_proxy,data_pts],[
            'SGD solution {}, param count={}, batch-size={}, iterations={}, step size={}'.format(sgd_legend_str,nb_params,M,nb_iter,eta),
            'data points'])
        #plt.title('SGD solution')
        ##
        fig = plt.figure()
        ax = Axes3D(fig)
        #train
        points_scatter = ax.scatter(Xp_train,Yp_train,Zp_train, marker='D')
        surf = ax.plot_surface(Xp_train,Yp_train,Zp_train, cmap=cm.coolwarm)
        #Test
        #points_scatter = ax.scatter(Xp,Yp,Zp, marker='D')
        #surf = ax.plot_surface(Xp,Yp,Zp, cmap=cm.coolwarm)
        plt.title('Test function')
    plt.show()

if __name__ == '__main__':
    print('main started')
    main()
    #print('End')
    print('\a')
