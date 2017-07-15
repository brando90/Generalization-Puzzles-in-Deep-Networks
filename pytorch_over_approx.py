import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

import maps

import pdb

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

def get_batch(X,Y,M):
    # N = len(Y)
    # valid_indices = np.array( range(N) )
    # batch_indices = np.random.choice(valid_indices,size=M,replace=False)
    # batch_xs = X[batch_indices,:]
    # batch_ys = Y[batch_indices]

    return batch_xs, batch_ys

def get_batch2(X,Y,M,dtype):
    X,Y = X.data.numpy(), Y.data.numpy()
    N = len(Y)
    valid_indices = np.array( range(N) )
    batch_indices = np.random.choice(valid_indices,size=M,replace=False)
    batch_xs = torch.FloatTensor(X[batch_indices,:]).type(dtype)
    batch_ys = torch.FloatTensor(Y[batch_indices]).type(dtype)
    return Variable(batch_xs, requires_grad=False), Variable(batch_ys, requires_grad=False)

def get_old():
    x = np.linspace(0,1,5)
    y = np.array([0,1,0,-1,0])
    y.shape = (N,1)
    X_true = poly_kernel_matrix( x,Degree ) # [N, D] = [N, Degree+1]
    return X_true, y

def main(argv=None):
    start_time = time.time()
    #
    np.set_printoptions(suppress=True)
    lb, ub = 0, 1
    ## true facts of the data set
    #B=10000
    N = 5
    Degree_true = 4
    D_true = Degree_true+1
    ## mdl degree and D
    Degree_mdl = 100
    D_sgd = Degree_mdl+1
    D_pinv = Degree_mdl+1
    D_rls = D_pinv
    ## sgd
    M = 2
    eta = 0.02 # eta = 1e-6
    nb_iter = int(100000)
    lambda_rls = 0.0001
    ##
    x_true = np.linspace(lb,ub,N) # the real data points
    #y = np.array([0,1,0,-1,0])
    #c_true = get_c_true(Degree_true,lb,ub)
    #X_true,y = get_data_set_points(c_true,x_true) # maps to the real feature space
    y = np.sin(2*np.pi*x_true)
    y.shape = (N,1)
    ## get linear algebra mdls
    X_mdl = poly_kernel_matrix(x_true,Degree_mdl) # maps to the feature space of the model
    c_pinv = np.dot(np.linalg.pinv(X_mdl),y) # [D_pinv,1]
    c_rls = get_RLS_soln(X_mdl,y,lambda_rls) # [D_pinv,1]
    ## TORCH
    dtype = torch.FloatTensor
    # dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
    X_mdl = Variable(torch.FloatTensor(X_mdl).type(dtype), requires_grad=False)
    y = Variable(torch.FloatTensor(y).type(dtype), requires_grad=False)
    ## SGD mdl
    #w_init = torch.randn(D_sgd,1).type(dtype)
    w_init = torch.zeros(D_sgd,1).type(dtype)
    W = Variable(w_init, requires_grad=True)
    ## debug print statements
    print('>>norm(y): ', ((1/N)*torch.norm(y)**2).data.numpy()[0] )
    print('>>l2_np: ', (1/N)*np.linalg.norm( y.data.numpy()-(np.dot(X_mdl.data.numpy(),W.data.numpy())) )**2 )
    print('>>l2_loss_torch: ', (1/N)*(X_mdl.mm(W) - y).pow(2).sum().data.numpy()[0] )
    print('>>(1/N)*(y_pred - y).pow(2).sum(): ', ((1/N)*(X_mdl.mm(W) - y).pow(2).sum()).data[0] )
    #pdb.set_trace()
    loss_list = []
    grad_list = []
    for i in range(nb_iter):
        #pdb.set_trace()
        #valid_indices = torch.arange(0,N).numpy()
        #valid_indices = np.array( range(N) )
        #batch_indices = np.random.choice(valid_indices,size=M,replace=False)
        #indices = torch.LongTensor(batch_indices)
        #batch_xs, batch_ys = torch.index_select(X_mdl, 0, indices), torch.index_select(y, 0, indices)
        #batch_xs,batch_ys = torch.index_select(X_mdl, 0, indices), torch.index_select(y, 0, indices)
        batch_xs, batch_ys = get_batch2(X_mdl,y,M,dtype)
        # Forward pass: compute predicted y using operations on Variables
        y_pred = batch_xs.mm(W)
        # Compute and print loss using operations on Variables. Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape (1,); loss.data[0] is a scalar value holding the loss.
        #loss = (1/N)*(y_pred - y).pow(2).sum()
        loss = (1/N)*(y_pred - batch_ys).pow(2).sum()
        # Use autograd to compute the backward pass. Now w will have gradients
        loss.backward()
        # Update weights using gradient descent; w1.data are Tensors,
        # w.grad are Variables and w.grad.data are Tensors.
        W.data -= eta * W.grad.data
        #
        #pdb.set_trace()
        if i % 500 == 0 or i == 0:
            loss_list.append(loss.data.numpy())
            grad_list.append( torch.norm(W.grad) )
        # Manually zero the gradients after updating weights
        W.grad.data.zero_()
    #
    c_sgd = W.data.numpy()
    X_mdl = X_mdl.data.numpy()
    y = y.data.numpy()
    #
    print('\n---- Learning params')
    print('Degree_mdl = {}, N = {}, M = {}, eta = {}, nb_iter = {}'.format(Degree_mdl,N,M,eta,nb_iter))
    #
    print('\n---- statistics about learned params')
    print('||c_sgd - c_pinv|| = ', np.linalg.norm(c_sgd - c_pinv,2))
    #
    print('c_sgd.shape: ', c_sgd.shape)
    print('c_pinv.shape: ', c_pinv.shape)
    print('c_rls.shape: ', c_rls.shape)
    print('norm(c_sgd): ', np.linalg.norm(c_sgd))
    print('norm(c_pinv): ', np.linalg.norm(c_pinv))
    print('norm(c_rls): ', np.linalg.norm(c_rls))

    #
    Xc_pinv = np.dot(X_mdl,c_sgd)
    print(' J(c_sgd) = ', (1/N)*(np.linalg.norm(y-Xc_pinv)**2) )
    Xc_pinv = np.dot(X_mdl,c_pinv)
    print( ' J(c_pinv) = ',(1/N)*(np.linalg.norm(y-Xc_pinv)**2) )
    Xc_rls = np.dot(X_mdl,c_rls)
    print( ' J(c_rls) = ',(1/N)*(np.linalg.norm(y-Xc_rls)**2) )
    #
    seconds = (time.time() - start_time)
    print('\a')
    ##
    x_horizontal = np.linspace(lb,ub,1000)
    X_plot = poly_kernel_matrix(x_horizontal,D_sgd-1)
    #plots
    p_sgd, = plt.plot(x_horizontal, np.dot(X_plot,c_sgd))
    p_pinv, = plt.plot(x_horizontal, np.dot(X_plot,c_pinv))
    #p_rls, = plt.plot(x_horizontal, np.dot(X_plot,c_rls))
    p_data, = plt.plot(x_true,y,'ro')
    #
    #p_list=[p_sgd,p_pinv,p_rls,p_data]
    #p_list=[p_data]
    p_list=[p_sgd,p_pinv,p_data]
    #p_list=[p_pinv,p_data]
    #plt.legend(p_list,['sgd curve Degree_mdl='+str(D_sgd-1),'min norm (pinv) Degree_mdl='+str(D_pinv-1),'rls regularization lambda={} Degree_mdl={}'.format(lambda_rls,D_rls-1),'data points'])
    plt.legend(p_list,['sgd curve Degree_mdl='+str(D_sgd-1),'min norm (pinv) Degree_mdl='+str(D_pinv-1),'data points'])
    #plt.legend(p_list,['min norm (pinv) Degree_mdl='+str(D_pinv-1),'data points'])
    plt.ylabel('f(x)')
    plt.show()
    ##
    minutes = seconds/ 60
    hours = minutes/ 60
    print("--- %s seconds ---" % seconds )
    print("--- %s minutes ---" % minutes )
    print("--- %s hours ---" % hours )
    print('\a')

if __name__ == '__main__':
    #tf.app.run()
    main()
    print('\a')
