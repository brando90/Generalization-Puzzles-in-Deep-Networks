import time
import numpy as np

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

import matplotlib.pyplot as plt
import scipy.integrate as integrate

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

def poly_kernel_matrix( x,D ):
    '''
    x = single rela number data value
    D = largest degree of monomial

    maps x to a kernel with each row being monomials of up to degree=D.
    [1, x^1, ..., x^D]
    '''
    N = len(x)
    Kern = np.zeros( (N,D+1) )
    for n in range(N):
        for d in range(D+1):
            Kern[n,d] = x[n]**d;
    return Kern

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

def main(argv=None):
    dtype = torch.FloatTensor
    # dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
    #pdb.set_trace()
    start_time = time.time()
    debug = True
    ##
    np.set_printoptions(suppress=True)
    lb, ub = -1, 1
    ## true facts of the data set
    N = 10
    ## mdl degree and D
    Degree_mdl = 8
    D_sgd = Degree_mdl+1
    D_pinv = Degree_mdl+1
    D_rls = D_pinv
    ## sgd
    M = 10
    eta = 0.00001 # eta = 1e-6
    A = 0.0
    nb_iter = int(1*1000)
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

    #### 2-layered mdl
    act = lambda x: x**2 # squared act
    #act = lambda x: F.relu(x) # relu act

    H1 = 2
    D0,D1,D2 = 1,H1,1
    D_layers,act = [D0,D1,D2], act

    #H1,H2 = 2,2
    #D0,D1,D2,D3 = 1,H1,H2,1
    #D_layers,act = [D0,D1,D2,D3], act

    #H1,H2,H3 = 3,3,3
    #D0,D1,D2,D3,D4 = 1,H1,H2,H3,1
    #D_layers,act = [D0,D1,D2,D3,D4], act

    bias = True
    init_config = Maps( {'w_init':'w_init_normal','mu':0.0,'std':1.0, 'bias_init':'b_fill','bias_value':0.01,'bias':bias ,'nb_layers':len(D_layers)} )
    w_inits_sgd, b_inits_sgd = get_initialization(init_config)
    #### Get Data set
    ## Get input variables X
    #data_set_name = 'sine'
    data_set_name = 'similar_nn'
    data_set_name = 'from_file'
    init_config_data = Maps({})
    f_true = None
    if data_set_name == 'sine':
        x_true = np.linspace(lb,ub,N) # the real data points
        Y = np.sin(2*np.pi*x_true)
        f_true = lambda x: np.sin(2*np.pi*x)
    elif data_set_name == 'similar_nn':
        ## Get data values from some net itself
        x_true = np.linspace(lb,ub,N)
        x_true.shape = x_true.shape[0],1
        #
        init_config_data = Maps( {'w_init':'w_init_normal','mu':0.0,'std':2.0, 'bias_init':'b_fill','bias_value':0.1,'bias':bias ,'nb_layers':len(D_layers)} )
        w_inits_data, b_inits_data = get_initialization(init_config_data)
        data_generator = NN(D_layers=D_layers,act=act,w_inits=w_inits_data,b_inits=b_inits_data,bias=bias)
        Y = get_Y_from_new_net(data_generator=data_generator, X=x_true,dtype=dtype)
        f_true = lambda x: f_mdl_eval(x,data_generator,dtype)
    elif data_set_name == 'from_file':
        data = np.load( './data/{}'.format('data_numpy_nb_layers3_biasTrue_mu0.0_std2.0.npz') )
        x_true, Y = data['X_train'], data['Y_train']
        X_test, Y_test = data['X_test'], data['Y_test']
    ## reshape
    Y.shape = (N,1) # TODO why do I need this?
    ## LA models
    Kern = poly_kernel_matrix(x_true,Degree_mdl)
    c_pinv = np.dot(np.linalg.pinv( Kern ),Y) # [D_pinv,1]
    c_rls = get_RLS_soln(Kern,Y,lambda_rls) # [D_pinv,1]
    ## data to TORCH
    print('len(D_layers) ', len(D_layers))
    #pdb.set_trace()
    if len(D_layers) == 2:
        X = poly_kernel_matrix(x_true,Degree_mdl) # maps to the feature space of the model
        #pdb.set_trace()
    else:
        X = x_true
        X.shape = X.shape[0],1
    print('X ', X)
    X = Variable(torch.FloatTensor(X).type(dtype), requires_grad=False)
    Y = Variable(torch.FloatTensor(Y).type(dtype), requires_grad=False)
    ## SGD model
    mdl_sgd = NN(D_layers=D_layers,act=act,w_inits=w_inits_sgd,b_inits=b_inits_sgd,bias=bias)
    # mdl_sgd = torch.nn.Sequential(
    #     torch.nn.Linear(D_sgd,1)
    # )
    # loss funtion
    #loss_fn = torch.nn.MSELoss(size_average=False)
    ## GPU
    #mdl_sgd.to_gpu() if (dtype == torch.cuda.FloatTensor) else 1

    ## check if deep net can equal
    #compare_first_layer(data_generator,mdl_sgd)
    #check_coeffs_poly(tmdl=mdl_sgd,act=sQuad,c_pinv=c_pinv,debug=True)

    #
    nb_module_params = len( list(mdl_sgd.parameters()) )
    loss_list = [ ]
    grad_list = [ [] for i in range(nb_module_params) ]
    #Ws = [W]
    #W_avg = Variable(torch.FloatTensor(W.data).type(dtype), requires_grad=False)
    #pdb.set_trace()
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
            #W = W - eta*W.grad
            #W.data = W.data - eta*W.grad.data + A*gdl_eps
            #W.data.copy_(W.data - eta*W.grad.data)
            W.data.copy_(W.data - eta*W.grad.data + A*gdl_eps) # W - eta*g + A*gdl_eps
        #pdb.set_trace()
        ## TRAINING STATS
        if i % 100 == 0 or i == 0:
            current_loss = loss.data.numpy()[0]
            loss_list.append(current_loss)
            if not np.isfinite(current_loss) or np.isinf(current_loss) or np.isnan(current_loss):
                print('loss: {} \n >>>>> BREAK HAPPENED'.format(current_loss) )
                break
            #for W in mdl_sgd.parameters():
            for i, W in enumerate(mdl_sgd.parameters()):
                grad_norm = W.grad.data.norm(2)
                #print('grad_norm: ',grad_norm)
                grad_list[i].append( W.grad.data.norm(2) )
                if not np.isfinite(grad_norm) or np.isinf(grad_norm) or np.isnan(grad_norm):
                    print('current_loss: {}, grad_norm: {},\n >>>>> BREAK HAPPENED'.format(current_loss,grad_norm) )
                    break
        ## Manually zero the gradients after updating weights
        mdl_sgd.zero_grad()
        ## COLLECT MOVING AVERAGES
        # for i in range(len(Ws)):
        #     W, W_avg = Ws[i], W_avgs[i]
        #     W_avgs[i] = (1/nb_iter)*W + W_avg
    ########################################################################################################################################################
    print('\a')
    ##
    X, Y = X.data.numpy(), Y.data.numpy()
    #
    if len(D_layers) <= 2:
        c_sgd = list(mdl_sgd.parameters())[0].data.numpy()
        c_sgd = c_sgd.transpose()
    else:
        ## tmdl
        tmdl = mdl_sgd
        ## sNN
        act = sQuad
        smdl = sNN(act,mdl=tmdl)
        ## get simplification
        x = symbols('x')
        expr = smdl.forward(x)
        s_expr = poly(expr,x)
        #c_sgd = s_expr.coeffs()
        #c_sgd.reverse()
        #c_sgd = np.array( c_sgd ) # first coeff is lowest degree
        c_sgd = np.array( s_expr.coeffs()[::-1] )
        c_sgd = [ np.float64(num) for num in c_sgd]
        #c_sgd = [ float(num) for num in c_sgd]
        #pdb.set_trace()
    if debug:
        print('X = ', X)
        print('Y = ', Y)
        print(mdl_sgd)
        if len(D_layers) > 2:
            print('structured poly: ', s_expr)
        print('c_sgd = ', c_sgd)
        print('c_pinv: ', c_pinv)
    #
    print('\n--> data set stats: data_set_name={}, init_config_data={}'.format(data_set_name,init_config_data) )
    print('---- Learning params')
    print('Degree_mdl = {}, N = {}, M = {}, eta = {}, nb_iter = {}'.format(Degree_mdl,N,M,eta,nb_iter))
    print('init_config: ', init_config)
    print('D_layers,act: ', D_layers,act)
    print('number of layers = {}'.format(nb_module_params))
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
        print('||c_sgd - c_pinv||_2 = ', np.linalg.norm(c_sgd - c_pinv,2))
        #print('||c_sgd - c_avg||_2 = ', np.linalg.norm(c_sgd - c_pinv,2))
        #print('||c_avg - c_pinv||_2 = ', np.linalg.norm(c_avg - c_pinv,2))
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
        print('J_gen(f_pinv) = ', (1/N)*(np.linalg.norm(Y_test-np.dot( poly_kernel_matrix( X_test,D_sgd-1 ),c_pinv))**2) )
    else:
        print('||f_sgd - f_true||^2_2 = ', L2_norm_2(f=f_sgd,g=f_true,lb=lb,ub=ub))
        print('||f_pinv - f_true||^2_2 = ', L2_norm_2(f=f_pinv,g=f_true,lb=lb,ub=ub))
    #
    print('-- Train Error')
    print(' J(c_sgd) = ', (1/N)*(mdl_sgd.forward(Variable(torch.FloatTensor(X))) - Variable(torch.FloatTensor(Y)) ).pow(2).sum().data.numpy() )
    print( ' J(c_pinv) = ',(1/N)*(np.linalg.norm(Y-np.dot( poly_kernel_matrix( x_true,D_sgd-1 ),c_pinv))**2) )
    print( ' J(c_rls) = ',(1/N)*(np.linalg.norm(Y-(1/N)*(np.linalg.norm(Y-np.dot( poly_kernel_matrix( x_true,D_sgd-1 ),c_rls))**2) )**2) )
    #
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    print("--- %s seconds ---" % seconds )
    print("--- %s minutes ---" % minutes )
    print("--- %s hours ---" % hours )
    print('\a')
    ## plots
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
            degree_sgd = 2**(len(D_layers)-2)
            sgd_legend_str = 'Degree model={} non linear-layers={}'.format(degree_sgd,nb_non_linear_layers)
        plt.legend(p_list,['SGD solution {}, batch-size={}, iterations={}, step size={}'.format(
        sgd_legend_str,M,nb_iter,eta),
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
    fig2 = plt.figure()
    for i in range(1):
        current_grad_list = grad_list[i]
        #pdb.set_trace()
        p_grads, = plt.plot(np.arange(len(current_grad_list)), current_grad_list,color='g')
        plt.legend([p_grads],['plot grads'])
        plt.title('Gradient vs Iterations: # {}'.format(i))
    ##
    plt.show()

if __name__ == '__main__':
    main()
    print('\a')
