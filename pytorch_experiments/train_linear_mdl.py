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

def train_SGD(mdl, M,eta,nb_iter,logging_freq ,dtype, X_train,Y_train, X_test,Y_test):
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
        ## train stats
        if i % (nb_iter/10) == 0 or i == 0:
            current_train_loss = (1/N_train)*(mdl.forward(X_train) - Y_train).pow(2).sum().data.numpy()
            print('i = {}, current_loss = {}'.format(i, current_train_loss ) )
        ## Manually zero the gradients after updating weights
        mdl.zero_grad()
##
logging_freq = 100
dtype = torch.FloatTensor
## SGD params
M = 3
eta = 0.0002
nb_iter = 20*1000
##
f_target = lambda x: np.sin(2*np.pi*x)
N_train = 5
X_train = np.linspace(0,1,N_train)
Y_train = f_target(X_train)
N_test = 50
X_test = np.linspace(0,1,N_test)
Y_test = f_target(X_test)
## degree of mdl
Degree_mdl = 5
## pseudo-inverse solution
c_pinv = np.polyfit( X_train, Y_train , Degree_mdl )[::-1]
## linear mdl to train with SGD
nb_terms = c_pinv.shape[0]
mdl_sgd = get_sequential_lifted_mdl(nb_monomials=nb_terms,D_out=1, bias=False)
## Make polynomial Kernel
poly_feat = PolynomialFeatures(degree=Degree_mdl)
Kern_train, Kern_test = poly_feat.fit_transform(X_train.reshape(N_train,1)), poly_feat.fit_transform(X_test.reshape(N_test,1))
Kern_train_pt, Y_train_pt = Variable(torch.FloatTensor(Kern_train).type(dtype), requires_grad=False), Variable(torch.FloatTensor(Y_train).type(dtype), requires_grad=False)
Kern_test_pt, Y_test_pt = Variable(torch.FloatTensor(Kern_test).type(dtype), requires_grad=False ), Variable(torch.FloatTensor(Y_test).type(dtype), requires_grad=False)
train_SGD(mdl_sgd, M,eta,nb_iter,logging_freq ,dtype, Kern_train_pt,Y_train_pt, Kern_test_pt,Y_test_pt)
##
legend_mdl = f'SGD solution standard parametrization, number of monomials={nb_terms}, batch-size={M}, iterations={nb_iter}, step size={eta}'
data = NamedDict(X_train=X_train,Y_train=Y_train_pt)
arg = NamedDict(mdl_sgd=mdl_sgd,data_lb=0,data_ub=1,c_pinv=c_pinv,X_train=X_train,poly_feat=poly_feat,data=data,legend_mdl=legend_mdl)
plot_1D_stuff(arg)
plt.show()
