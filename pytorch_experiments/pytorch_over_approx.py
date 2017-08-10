import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import maps
import pdb

from models_pytorch import *
from inits import *

def f_mdl_LA(x,c):
    D,_ = c.shape
    X = poly_kernel_matrix( [x],D-1 )
    return np.dot(X,c)

def f_mdl_eval(x,mdl_eval,dtype):
    _,D = list(mdl_eval.parameters())[0].data.numpy().shape
    if len(list(mdl_eval.parameters())) == 2:
        x = poly_kernel_matrix( [x],D-1 )
        x = Variable(torch.FloatTensor([x]).type(dtype))
    else:
        x = Variable(torch.FloatTensor([x]).type(dtype))
        x = x.view(1,1)
    y_pred = mdl_eval.forward(x)
    return y_pred.data.numpy()

def L2_norm_2(f,g,lb=0,ub=1):
    f_g_2 = lambda x: (f(x) - g(x))**2
    result = integrate.quad(func=f_g_2, a=lb,b=ub)
    integral_val = result[0]
    return integral_val

def get_c_true(Degree_true,lb=0,ub=1):
    x = np.linspace(lb,ub,5)
    Y = np.array([0,1,0,-1,0])
    Y.shape = (5,1)
    X = poly_kernel_matrix( x,Degree_true ) # [N, D] = [N, Degree_mdl+1]
    c_true = np.dot(np.linalg.pinv(X),Y) # [N,1]
    return c_true

def get_data_set_points(c_true,x,Degree_truth=4):
    N = len(x)
    X = poly_kernel_matrix(x,Degree_truth)
    Y = np.dot(X,c_true)
    Y.shape = (N,1)
    return X,Y

def poly_kernel_matrix( x,D ):
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

def get_old():
    x = np.linspace(0,1,5)
    Y = np.array([0,1,0,-1,0])
    Y.shape = (N,1)
    X_true = poly_kernel_matrix( x,Degree ) # [N, D] = [N, Degree+1]
    #c_true = get_c_true(Degree_true,lb,ub)
    #X_true,Y = get_data_set_points(c_true,x_true) # maps to the real feature space
    return X_true, Y

def main(argv=None):
    #pdb.set_trace()
    start_time = time.time()
    debug = True
    ##
    np.set_printoptions(suppress=True)
    lb, ub = 0, 1
    ## true facts of the data set
    N = 5
    Degree_true = 4
    D_true = Degree_true+1
    ## mdl degree and D
    Degree_mdl = 10
    D_sgd = Degree_mdl+1
    D_pinv = Degree_mdl+1
    D_rls = D_pinv
    ## sgd
    M = 5
    eta = 0.0001 # eta = 1e-6
    A = 0.0
    nb_iter = int(2*10000)
    # RLS
    lambda_rls = 0.005
    ## one layered mdl
    identity_act = lambda x: x
    D_1,D_2 = D_sgd,1 # note D^(0) is not present cuz the polyomial is explicitly constructed by me
    D_layers,act = [D_1,D_2], identity_act
    w_inits = [None]+[lambda x: w_init_zero(x) for i in range(len(D_layers)) ]
    #w_inits = [None]+[lambda x: w_init_normal(x,mu=0.0,std=1.0) for i in range(len(D_layers)) ]
    b_inits = [None]+[lambda x: b_fill(x,value=0.1) for i in range(len(D_layers)) ]
    ## two layered mdl
    # act = lambda x: x**2 # squared act
    # #act = lambda x: F.relu(x) # relu act
    # D0,D1,D2,D3 = 1,2,2,1
    # D_layers,act = [D0,D1,D2,D3], act
    # #w_inits, b_inits = lambda x: w_init_normal(x,mu=0.0,1.0), lambda x: b_fill(x,value=0.1)
    # w_inits = [None]+[lambda x: torch.nn.init.xavier_normal(x, gain=1) for i in range(len(D_layers)) ]
    # b_inits = [None]+[lambda x: b_fill(x,value=0.1) for i in range(len(D_layers)) ]
    #### Get Data set
    ## Get input variables X
    x_true = np.linspace(lb,ub,N) # the real data points
    ## Get target variables Y
    Y = np.sin(2*np.pi*x_true)
    Y.shape = (N,1)
    ## data to TORCH
    dtype = torch.FloatTensor
    # dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
    if len(D_layers) == 2:
        X = poly_kernel_matrix(x_true,Degree_mdl) # maps to the feature space of the model
    else:
        X = x_true
    X = Variable(torch.FloatTensor(X).type(dtype), requires_grad=False)
    Y = Variable(torch.FloatTensor(Y).type(dtype), requires_grad=False)
    #### Get models
    ## LA mdls
    c_pinv = np.dot(np.linalg.pinv(X.data.numpy()),Y.data.numpy()) # [D_pinv,1]
    c_rls = get_RLS_soln(X.data.numpy(),Y.data.numpy(),lambda_rls) # [D_pinv,1]
    ## SGD model
    mdl_sgd = NN(D_layers=D_layers,act=act,w_inits=w_inits,b_inits=b_inits)
    # loss funtion
    #loss_fn = torch.nn.MSELoss(size_average=False)
    ## GPU
    mdl_sgd.to_gpu() if (dtype == torch.cuda.FloatTensor) else 1
    ## debug print statements
    #print('Y: ',Y)
    print('>>norm(Y): ', ((1/N)*torch.norm(Y)**2).data.numpy()[0] )
    print('>>l2_loss_torch: ', (1/N)*( Y - mdl_sgd.forward(X)).pow(2).sum() )
    #
    nb_params = len( list(mdl_sgd.parameters()) )
    loss_list = [ ]
    grad_list = [ [] for i in range(nb_params) ]
    #Ws = [W]
    #W_avg = Variable(torch.FloatTensor(W.data).type(dtype), requires_grad=False)
    for i in range(nb_iter):
        # Forward pass: compute predicted Y using operations on Variables
        batch_xs, batch_ys = get_batch2(X,Y,M,dtype) # [M, D], [M, 1]
        ## FORWARD PASS
        y_pred = mdl_sgd.forward(X)
        ## LOSS
        loss = (1/N)*(y_pred - batch_ys).pow(2).sum()
        ## BACKARD PASS
        loss.backward() # Use autograd to compute the backward pass. Now w will have gradients
        ## SGD update
        for W in mdl_sgd.parameters():
            gdl_eps = torch.randn(W.data.size()).type(dtype)
            W.data = W.data - eta*W.grad.data + A*gdl_eps # W - eta*g + A*gdl_eps % B
        ## Manually zero the gradients after updating weights
        mdl_sgd.zero_grad()
        ## TRAINING STATS
        if i % 500 == 0 or i == 0:
            current_loss = loss.data.numpy()[0]
            loss_list.append(current_loss)
            if not np.isfinite(current_loss) or np.isinf(current_loss) or np.isnan(current_loss):
                print('loss: {} \n >>>>> BREAK HAPPENED'.format(current_loss) )
                break
            # get grads
            #for W in mdl_sgd.parameters():
            for i, W in enumerate(mdl_sgd.parameters()):
                grad_norm = W.grad.data.norm(2)
                grad_list[i].append( W.grad.data.norm(2) )
                if not np.isfinite(grad_norm) or np.isinf(grad_norm) or np.isnan(grad_norm):
                    print('current_loss: {}, grad_norm: {},\n >>>>> BREAK HAPPENED'.format(current_loss,grad_norm) )
                    break
        ## COLLECT MOVING AVERAGES
        # for i in range(len(Ws)):
        #     W, W_avg = Ws[i], W_avgs[i]
        #     W_avgs[i] = (1/nb_iter)*W + W_avg
    ##
    X = X.data.numpy()
    Y = Y.data.numpy()
    if len(D_layers) == 2:
        print('list(mdl_sgd.parameters()) = ', list(mdl_sgd.parameters()))
        c_sgd = list(mdl_sgd.parameters())[0].data.numpy()
        c_sgd = c_sgd.transpose()
    #
    if len(D_layers) == 2:
        c_sgd = list(mdl_sgd.parameters())[0].data.numpy()
    if debug:
        if len(D_layers) == 2:
            print('c_sgd = ', c_sgd)
        print('c_pinv: ', c_pinv)
    #
    print('\n---- Learning params')
    print('Degree_mdl = {}, N = {}, M = {}, eta = {}, nb_iter = {}'.format(Degree_mdl,N,M,eta,nb_iter))
    #
    print('\n---- statistics about learned params')
    print('--L1')
    print('||c_pinv||_1 = {} '.format(np.linalg.norm(c_pinv,1)) )
    #print('||c_avg||_1 = {} '.format(np.linalg.norm(c_avg,1)) )
    print('||c_sgd||_1 = {} '.format(np.linalg.norm(c_sgd,1)) )
    print('--L2')
    print('||c_sgd||_2 = ', np.linalg.norm(c_sgd,2))
    #print('||c_avg||_2 = {} '.format(np.linalg.norm(c_avg,2))
    print('||c_pinv||_2 = ', np.linalg.norm(c_pinv,2))
    print('---- parameters differences')
    if len(D_layers) == 2:
        print('||c_sgd - c_pinv||_2 = ', np.linalg.norm(c_sgd - c_pinv,2))
        #print('||c_sgd - c_avg||_2 = ', np.linalg.norm(c_sgd - c_pinv,2))
        #print('||c_avg - c_pinv||_2 = ', np.linalg.norm(c_avg - c_pinv,2))
    f_sgd = lambda x: f_mdl_eval(x,mdl_sgd,dtype)
    f_pinv = lambda x: f_mdl_LA(x,c_pinv)
    print('-- L2 functional norm difference')
    print('||f_sgd - f_pinv||^2_2 = ', L2_norm_2(f=f_sgd,g=f_pinv,lb=0,ub=1))
    #print('||f_avg - f_pinv||^2_2 = ', L2_norm_2(f=f_avg,g=f_pinv,lb=0,ub=1))
    #
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
    plots = {}
    plots['p_sgd'] = plt.plot(x_horizontal, [ float(f_sgd(x_i)[0]) for x_i in x_horizontal ])
    plots['p_pinv'] = plt.plot(x_horizontal, np.dot(X_plot,c_pinv))
    plots['p_rls'] = plt.plot(x_horizontal, np.dot(X_plot,c_rls))
    plots['p_data'] = plt.plot(x_true,Y,'ro')
    p_list = list(plots.keys())
    if 'p_rls' in plots:
        plt.legend(p_list,['sgd curve Degree_mdl='+str(D_sgd-1),
        'min norm (pinv) Degree_mdl='+str(D_pinv-1),
        'rls regularization lambda={} Degree_mdl={}'.format(lambda_rls,D_rls-1),
        'data points'])
    else:
        plt.legend(p_list,['sgd curve Degree_mdl={}, batch-size= {}, iterations={}, eta={}'.format(
        str(D_sgd-1),M,nb_iter,eta),
        'min norm (pinv) Degree_mdl='+str(D_pinv-1),
        'data points'])
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

def plot_pts():
    fig2 = plt.figure()
    p_grads, = plt.plot([200,375,500], [0.45283,0.1125,0.02702],color='g')
    plt.legend([p_grads],['L2 norm'])
    plt.title('How SGD approaches pseudoinverse')
    plt.xlabel('iterations (thousands)')
    plt.ylabel('L2 norm between SGD solution and pinv')
    plt.show()

if __name__ == '__main__':
    #tf.app.run()
    #plot_pts()
    main()
    print('\a')
