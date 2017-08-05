import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

import torch
from torch.autograd import Variable

import maps
import pdb

from models_pytorch import *
from inits import *

def L2_norm_2(f,g,lb=0,ub=1):
    f_g_2 = lambda x: (f(x) - g(x))**2
    result = integrate.quad(func=f_g_2, a=lb,b=ub)
    integral_val = result[0]
    return integral_val

def get_c_true(Degree_true,lb=0,ub=1):
    x = np.linspace(lb,ub,5)
    y = np.array([0,1,0,-1,0])
    y.shape = (5,1)
    X_mdl = poly_kernel_matrix( x,Degree_true ) # [N, D] = [N, Degree_mdl+1]
    c_true = np.dot(np.linalg.pinv(X_mdl),y) # [N,1]
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
    X,Y = X.data.numpy(), Y.data.numpy()
    N = len(Y)
    valid_indices = np.array( range(N) )
    batch_indices = np.random.choice(valid_indices,size=M,replace=False)
    batch_xs = index_batch(X,batch_indices,dtype)
    batch_ys = index_batch(Y,batch_indices,dtype)
    return Variable(batch_xs, requires_grad=False), Variable(batch_ys, requires_grad=False)

def get_old():
    x = np.linspace(0,1,5)
    y = np.array([0,1,0,-1,0])
    y.shape = (N,1)
    X_true = poly_kernel_matrix( x,Degree ) # [N, D] = [N, Degree+1]
    #c_true = get_c_true(Degree_true,lb,ub)
    #X_true,y = get_data_set_points(c_true,x_true) # maps to the real feature space
    return X_true, y

def main(argv=None):
    #pdb.set_trace()
    start_time = time.time()
    ##
    np.set_printoptions(suppress=True)
    lb, ub = 0, 1
    ## true facts of the data set
    N = 5
    Degree_true = 4
    D_true = Degree_true+1
    ## mdl degree and D
    Degree_mdl = 4
    D_sgd = Degree_mdl+1
    D_pinv = Degree_mdl+1
    D_rls = D_pinv
    ## sgd
    M = 5
    eta = 0.0001 # eta = 1e-6
    A = 0.25
    nb_iter = int(1000000)
    # RLS
    lambda_rls = 0.0001
    ##
    x_true = np.linspace(lb,ub,N) # the real data points
    y = np.sin(2*np.pi*x_true)
    y.shape = (N,1)
    print('y: ',y)
    ## get linear algebra mdls
    X_mdl = poly_kernel_matrix(x_true,Degree_mdl) # maps to the feature space of the model
    c_pinv = np.dot(np.linalg.pinv(X_mdl),y) # [D_pinv,1]
    c_rls = get_RLS_soln(X_mdl,y,lambda_rls) # [D_pinv,1]
    ## TORCH
    dtype = torch.FloatTensor
    # dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
    X_mdl = Variable(torch.FloatTensor(x_true).type(dtype), requires_grad=False)
    X_mdl = X_mdl.view(N,1)
    y = Variable(torch.FloatTensor(y).type(dtype), requires_grad=False)
    ## SGD mdl


    mdl_sgd = NN()
    ## debug print statements
    print('>>norm(y): ', ((1/N)*torch.norm(y)**2).data.numpy()[0] )
    #print('>>l2_np: ', (1/N)*np.linalg.norm( y.data.numpy()-(np.dot(X_mdl.data.numpy(),W.data.numpy())) )**2 )
    #print('>>l2_loss_torch: ', (1/N)*(X_mdl.mm(W) - y).pow(2).sum().data.numpy()[0] )
    #print('>>(1/N)*(y_pred - y).pow(2).sum(): ', ((1/N)*(X_mdl.mm(W) - y).pow(2).sum()).data[0] )
    #pdb.set_trace()
    loss_list = []
    grad_list = []
    Ws = [W_l1,W_l2,W_out]
    #Ws = [W]
    #W_avg = Variable(torch.FloatTensor(W.data).type(dtype), requires_grad=False)
    W_avgs = [Variable(torch.FloatTensor(W_l1.data).type(dtype), requires_grad=False),Variable(torch.FloatTensor(W_l2.data).type(dtype), requires_grad=False),Variable(torch.FloatTensor(W_out.data).type(dtype), requires_grad=False)]
    # for in range(3):
    #     w_avg = Variable(torch.FloatTensor(W.data).type(dtype), requires_grad=False)
    #     W_avgs.append()
    for i in range(nb_iter):
        # Forward pass: compute predicted y using operations on Variables
        batch_xs, batch_ys = get_batch2(X_mdl,y,M,dtype) # [M, D], [M, 1]
        ## FORWARD PASS
        #y_pred = batch_xs.mm(W)
        a_l1 = batch_xs.mm(W_l1)**2 # [M,H^(1)] = [M,D]x[D,H^(1)]
        a_l2 = a_l1.mm(W_l2)**2 # [M,H^(2)] = [M,H^(1)]x[H^(1),H^(2)]
        y_pred = a_l2.mm(W_out) # [M,1] = [M,H^(2)]x[M^(2),1]
        ## LOSS
        loss = (1/N)*(y_pred - batch_ys).pow(2).sum()
        ## BACKARD PASS
        loss.backward() # Use autograd to compute the backward pass. Now w will have gradients
        ## SGD
        for W in Ws:
            gdl_eps = torch.randn(W.data.size()).type(dtype)
            W.data = W.data - eta*W.grad.data + A*gdl_eps # W - eta*g + A*gdl_eps, B
        ## TRAINING STATS
        if i % 500 == 0 or i == 0:
            current_loss, current_norm_grad_2 = loss.data.numpy()[0], torch.norm(W.grad).data.numpy()[0]
            print('current_norm_grad_2: ', current_norm_grad_2)
            loss_list.append(current_loss)
            grad_list.append( current_norm_grad_2 )
            if not np.isfinite(current_loss) or np.isinf(current_loss) or np.isnan(current_loss):
                print('loss: ',current_loss)
                print('>>>>> BREAK HAPPENED')
                break
        ## Manually zero the gradients after updating weights
        for W in Ws:
            W.grad.data.zero_()
        ## COLLECT MOVING AVERAGES
        for i in range(len(Ws)):
            W, W_avg = Ws[i], W_avgs[i]
            W_avgs[i] = (1/nb_iter)*W + W_avg
    #
    #c_avg = W_avg.data.numpy()
    #c_sgd = W.data.numpy()
    X_mdl = X_mdl.data.numpy()
    y = y.data.numpy()
    #
    print('c_pinv: ', c_pinv)
    #
    print('\n---- Learning params')
    print('Degree_mdl = {}, N = {}, M = {}, eta = {}, nb_iter = {}'.format(Degree_mdl,N,M,eta,nb_iter))
    #
    print('\n---- statistics about learned params')
    #print('||c_sgd - c_pinv|| = ', np.linalg.norm(c_sgd - c_pinv,2))
    #print('||c_sgd - c_avg|| = ', np.linalg.norm(c_sgd - c_pinv,2))
    #print('||c_avg - c_pinv|| = ', np.linalg.norm(c_avg - c_pinv,2))
    # def f_avg(x):
    #     X = poly_kernel_matrix( [x],D_sgd-1 )
    #     return np.dot(X,c_avg)
    def f_avg(x):
        W_l1 = W_avgs[0]
        W_l2 = W_avgs[1]
        W_out = W_avgs[2]
        x = torch.FloatTensor([x])
        x = Variable(torch.FloatTensor(x).type(dtype))
        x = x.view(1,1)
        a_l1 = x.mm(W_l1)**2 # [M,H^(1)] = [M,D]x[D,H^(1)]
        a_l2 = a_l1.mm(W_l2)**2 # [M,H^(2)] = [M,H^(1)]x[H^(1),H^(2)]
        y_pred = a_l2.mm(W_out) # [M,1] = [M,H^(2)]x[M^(2),1]
        return y_pred.data.numpy()
    # def f_sgd(x):
    #     X = poly_kernel_matrix( [x],D_sgd-1 )
    #     return np.dot(X,c_sgd)
    def f_sgd(x):
        W_l1 = Ws[0]
        W_l2 = Ws[1]
        W_out = Ws[2]
        #pdb.set_trace()
        x = torch.FloatTensor([x])
        x = Variable(torch.FloatTensor(x).type(dtype))
        x = x.view(1,1)
        a_l1 = x.mm(W_l1)**2 # [M,H^(1)] = [M,D]x[D,H^(1)]
        a_l2 = a_l1.mm(W_l2)**2 # [M,H^(2)] = [M,H^(1)]x[H^(1),H^(2)]
        y_pred = a_l2.mm(W_out) # [M,1] = [M,H^(2)]x[M^(2),1]
        return y_pred.data.numpy()
    def g_pinv(x):
        X = poly_kernel_matrix( [x],D_sgd-1 )
        return np.dot(X,c_pinv)
    print('||f_sgd - f_pinv||^2 = ', L2_norm_2(f=f_sgd,g=g_pinv,lb=0,ub=1))
    print('||f_avg - f_pinv||^2 = ', L2_norm_2(f=f_avg,g=g_pinv,lb=0,ub=1))
    #
    #print('c_sgd.shape: ', c_sgd.shape)
    print('c_pinv.shape: ', c_pinv.shape)
    print('c_rls.shape: ', c_rls.shape)
    #print('norm(c_sgd): ', np.linalg.norm(c_sgd))
    print('norm(c_pinv): ', np.linalg.norm(c_pinv))
    print('norm(c_rls): ', np.linalg.norm(c_rls))

    #
    #Xc_sdg = np.dot(X_mdl,c_sgd)
    #print(' J(c_sgd) = ', (1/N)*(np.linalg.norm(y-Xc_sdg)**2) )
    a_l1 = Variable(torch.FloatTensor(X_mdl)).mm(W_l1)**2 # [M,H^(1)] = [M,D]x[D,H^(1)]
    a_l2 = a_l1.mm(W_l2)**2 # [M,H^(2)] = [M,H^(1)]x[H^(1),H^(2)]
    y_pred = a_l2.mm(W_out) # [M,1] = [M,H^(2)]x[M^(2),1]
    print(' J(c_sgd) = ', (1/N)*(y_pred - Variable(torch.FloatTensor(y)) ).pow(2).sum().data.numpy() )
    Xc_pinv = np.dot( poly_kernel_matrix( x_true,D_sgd-1 ),c_pinv)
    print( ' J(c_pinv) = ',(1/N)*(np.linalg.norm(y-Xc_pinv)**2) )
    #Xc_rls = np.dot(X_mdl,c_rls)
    #print( ' J(c_rls) = ',(1/N)*(np.linalg.norm(y-Xc_rls)**2) )
    #Xc_W_avg = np.dot(X_mdl,c_avg)
    #print( ' J(c_avg) = ',(1/N)*(np.linalg.norm(y-Xc_W_avg)**2) )
    #
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    print("--- %s seconds ---" % seconds )
    print("--- %s minutes ---" % minutes )
    print("--- %s hours ---" % hours )
    print('\a')
    ##
    x_horizontal = np.linspace(lb,ub,1000)
    X_plot = poly_kernel_matrix(x_horizontal,D_sgd-1)
    #plots
    #p_sgd, = plt.plot(x_horizontal, np.dot(X_plot,c_sgd))
    p_sgd, = plt.plot(x_horizontal, [ float(f_sgd(x_i)[0]) for x_i in x_horizontal ])
    p_pinv, = plt.plot(x_horizontal, np.dot(X_plot,c_pinv))
    #p_rls, = plt.plot(x_horizontal, np.dot(X_plot,c_rls))
    #p_avg, = plt.plot(x_horizontal, np.dot(X_plot,c_avg))
    #p_avg, = plt.plot(x_horizontal, [ float(f_avg(x_i)[0]) for x_i in x_horizontal ])
    p_data, = plt.plot(x_true,y,'ro')
    #
    #p_list=[p_sgd,p_pinv,p_rls,p_data]
    #p_list=[p_data]
    p_list=[p_sgd,p_pinv,p_data]
    #p_list = [p_avg,p_sgd,p_pinv,p_data]
    #p_list=[p_pinv,p_data]
    #plt.legend(p_list,['sgd curve Degree_mdl='+str(D_sgd-1),'min norm (pinv) Degree_mdl='+str(D_pinv-1),'rls regularization lambda={} Degree_mdl={}'.format(lambda_rls,D_rls-1),'data points'])
    plt.legend(p_list,['sgd curve Degree_mdl={}, batch-size= {}, iterations={}, eta={}'.format(str(D_sgd-1),M,nb_iter,eta),'min norm (pinv) Degree_mdl='+str(D_pinv-1),'data points'])
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
    p_grads, = plt.plot(np.arange(len(grad_list)), grad_list,color='g')
    plt.legend([p_grads],['plot grads'])
    plt.title('Gradient vs Iterations')
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
