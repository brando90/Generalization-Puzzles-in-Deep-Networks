import numpy as np
from sklearn.preprocessing import PolynomialFeatures

import torch
from torch.autograd import Variable

from maps import NamedDict

from plotting_utils import *

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

def get_sequential_lifted_mdl(nb_monomials,D_out, bias=False):
    return torch.nn.Sequential(torch.nn.Linear(nb_monomials,D_out,bias=bias))

def train_SGD(mdl, M,eta,nb_iter,logging_freq ,dtype, X_train,Y_train):
    ##
    N_train,_ = tuple( X_train.size() )
    #print(N_train)
    for i in range(nb_iter):
        # Forward pass: compute predicted Y using operations on Variables
        batch_xs, batch_ys = get_batch2(X_train,Y_train,M,dtype) # [M, D], [M, 1]
        ## FORWARD PASS
        y_pred = mdl.forward(batch_xs)
        ## LOSS + Regularization
        batch_loss = (1/M)*(y_pred - batch_ys).pow(2).sum()
        ## BACKARD PASS
        batch_loss.backward() # Use autograd to compute the backward pass. Now w will have gradients
        ## SGD update
        for W in mdl.parameters():
            delta = eta*W.grad.data
            W.data.copy_(W.data - delta)
        ## Manually zero the gradients after updating weights
        mdl.zero_grad()
        ## train stats
        if i % (nb_iter/10) == 0 or i == 0:
            current_train_loss = (1/N_train)*(mdl.forward(X_train) - Y_train).pow(2).sum().data.numpy()
            print('i = {}, current_loss = {}'.format(i, current_train_loss ) )
##
logging_freq = 100
dtype = torch.FloatTensor
## SGD params
M = 3
eta = 0.002
nb_iter = 20*1000
##
lb,ub = 0,1
f_target = lambda x: np.sin(2*np.pi*x)
N_train = 5
X_train = np.linspace(lb,ub,N_train)
Y_train = f_target(X_train)
## degree of mdl
Degree_mdl = 4
## pseudo-inverse solution
c_pinv = np.polyfit( X_train, Y_train , Degree_mdl )[::-1]
## linear mdl to train with SGD
nb_terms = c_pinv.shape[0]
mdl_sgd = get_sequential_lifted_mdl(nb_monomials=nb_terms,D_out=1, bias=False)
mdl_sgd[0].weight.data.normal_(mean=0,std=0.001)
## Make polynomial Kernel
poly_feat = PolynomialFeatures(degree=Degree_mdl)
Kern_train = poly_feat.fit_transform(X_train.reshape(N_train,1))
Kern_train_pt, Y_train_pt = Variable(torch.FloatTensor(Kern_train).type(dtype), requires_grad=False), Variable(torch.FloatTensor(Y_train).type(dtype), requires_grad=False)
train_SGD(mdl_sgd, M,eta,nb_iter,logging_freq ,dtype, Kern_train_pt,Y_train_pt)

#### PLOTTING
x_horizontal = np.linspace(lb,ub,1000).reshape(1000,1)
X_plot = poly_feat.fit_transform(x_horizontal)
X_plot_pytorch = Variable( torch.FloatTensor(X_plot), requires_grad=False)
##
fig1 = plt.figure()
#plots objs
p_sgd, = plt.plot(x_horizontal, [ float(f_val) for f_val in mdl_sgd.forward(X_plot_pytorch).data.numpy() ])
p_pinv, = plt.plot(x_horizontal, np.dot(X_plot,c_pinv))
p_data, = plt.plot(X_train,Y_train,'ro')
## legend
nb_terms = c_pinv.shape[0]
legend_mdl = f'SGD solution standard parametrization, number of monomials={nb_terms}, batch-size={M}, iterations={nb_iter}, step size={eta}'
plt.legend(
        [p_sgd,p_pinv,p_data],
        [legend_mdl,f'linear algebra soln, number of monomials={nb_terms}',f'data points = {N_train}']
    )
##
plt.xlabel('x'), plt.ylabel('f(x)')
plt.show()
