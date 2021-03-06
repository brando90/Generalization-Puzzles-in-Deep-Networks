import time
start_time = time.time()

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial.hermite import hermvander

import random

import torch
from torch.autograd import Variable

from maps import NamedDict

from plotting_utils import *

def get_hermite_poly(x,degree):
    #scipy.special.hermite()
    N, = x.shape
    ##
    X = np.zeros( (N,degree+1) )
    for n in range(N):
        for deg in range(degree+1):
            #X[n,deg] = hermite( n=deg, z=float(x[deg]) )
            X[n,deg] = legendre( n=deg, x=float(x[deg]) )
            #X[n,deg] = chebyt( n=deg, x=float(x[deg]) )
            #X[n,deg] = chebyu( n=deg, x=float(x[deg]) )
    return X

def get_chebyshev_nodes(lb,ub,N):
    k = np.arange(1,N+1)
    chebyshev_nodes = 0.5*(lb+ub)+0.5*(ub-lb)*np.cos((np.pi*2*k-1)/(2*N))
    return chebyshev_nodes

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

def get_batch3(X,Y,M,dtype):
    '''
    get batch for pytorch model
    '''
    # TODO fix and make it nicer, there is pytorch forum question
    #X,Y = X.data.numpy(), Y.data.numpy()
    X,Y = X, Y
    N = X.size()[0]
    #indices = np.random.randint(0,N,size=M)
    #indices = [ random.randint(0,N) for in i range(M)]
    indices = np.random.random_integers(N,size=(M,))
    if dtype ==  torch.cuda.FloatTensor:
        batch_indices = torch.cuda.LongTensor( indices )# without replacement
    else:
        batch_indices = torch.LongTensor( indices ).type(dtype)  # without replacement
    batch_xs = torch.index_select(X,0,batch_indices)
    batch_ys = torch.index_select(Y,0,batch_indices)
    return Variable(batch_xs, requires_grad=False), Variable(batch_ys, requires_grad=False)

def get_sequential_lifted_mdl(nb_monomials,D_out, bias=False):
    return torch.nn.Sequential(torch.nn.Linear(nb_monomials,D_out,bias=bias))

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

def train_SGD(mdl, M,eta,nb_iter,logging_freq ,dtype, X_train,Y_train):
    ##
    #pdb.set_trace()
    N_train,_ = tuple( X_train.size() )
    #print(N_train)
    for i in range(1,nb_iter+1):
        # Forward pass: compute predicted Y using operations on Variables
        batch_xs, batch_ys = get_batch2(X_train,Y_train,M,dtype) # [M, D], [M, 1]
        ## FORWARD PASS
        y_pred = mdl.forward(batch_xs)
        ## Check vectors have same dimension
        if vectors_dims_dont_match(batch_ys,y_pred):
            raise ValueError('You vectors don\'t have matching dimensions. It will lead to errors.')
        ## LOSS + Regularization
        batch_loss = (1/M)*(y_pred - batch_ys).pow(2).sum()
        ## BACKARD PASS
        batch_loss.backward() # Use autograd to compute the backward pass. Now w will have gradients
        ## SGD update
        for W in mdl.parameters():
            eta = 0.1/(i**0.6)
            delta = eta*W.grad.data
            W.data.copy_(W.data - delta)
        ## train stats
        if i % (nb_iter/10) == 0 or i == 0:
            #X_train_, Y_train_ = Variable(X_train), Variable(Y_train)
            X_train_, Y_train_ = X_train, Y_train
            current_train_loss = (1/N_train)*(mdl.forward(X_train_) - Y_train_).pow(2).sum().data.numpy()
            print('\n-------------')
            print(f'i = {i}, current_train_loss = {current_train_loss}\n')
            print(f'eta*W.grad.data = {eta*W.grad.data}')
            print(f'W.grad.data = {W.grad.data}')
        ## Manually zero the gradients after updating weights
        mdl.zero_grad()
    final_sgd_error = current_train_loss
    return final_sgd_error
##
D0=1
logging_freq = 100
#dtype = torch.cuda.FloatTensor
dtype = torch.FloatTensor
## SGD params
M = 5
eta = 0.01
nb_iter = 500*10
##
lb,ub = 0,1
freq_sin = 4 # 2.3
f_target = lambda x: np.sin(2*np.pi*freq_sin*x)
N_train = 10
#X_train = np.linspace(lb,ub,N_train)
X_train = get_chebyshev_nodes(lb,ub,N_train).reshape(N_train,D0)
Y_train = f_target(X_train).reshape(N_train,1)
x_horizontal = np.linspace(lb,ub,1000).reshape(1000,1)
## degree of mdl
Degree_mdl = N_train-1
## pseudo-inverse solution
## Standard
poly_feat = PolynomialFeatures(degree=Degree_mdl)
Kern_train = poly_feat.fit_transform(X_train.reshape(N_train,1))
Kern_train,_ = np.linalg.qr(Kern_train)
X_plot = poly_feat.fit_transform(x_horizontal)
X_plot,_ = np.linalg.qr(X_plot)
## Hermite
#Kern_train = hermvander(X_train,Degree_mdl)
print(f'Kern_train.shape={Kern_train.shape}')
#Kern_train = Kern_train.reshape(N_train,Kern_train.shape[2])
#X_plot = hermvander(x_horizontal,Degree_mdl)
#X_plot = X_plot.reshape(1000,X_plot.shape[2])
#Kern_train = get_hermite_poly(X_train,Degree_mdl)
#Kern_train,_ = np.linalg.qr(Kern_train)
##
Kern_train_pinv = np.linalg.pinv( Kern_train )
c_pinv = np.dot(Kern_train_pinv, Y_train)
##
condition_number_hessian = np.linalg.cond( np.dot(Kern_train.T,Kern_train))
## linear mdl to train with SGD
nb_terms = c_pinv.shape[0]
mdl_sgd = get_sequential_lifted_mdl(nb_monomials=nb_terms,D_out=1, bias=False)
mdl_sgd[0].weight.data.normal_(mean=0,std=0.001)
mdl_sgd[0].weight.data.fill_(0)
## Make polynomial Kernel
Kern_train_pt, Y_train_pt = Variable(torch.FloatTensor(Kern_train).type(dtype), requires_grad=False), Variable(torch.FloatTensor(Y_train).type(dtype), requires_grad=False)
#Kern_train_pt, Y_train_pt = torch.FloatTensor(Kern_train).type(dtype), torch.FloatTensor(Y_train).type(dtype)
final_sgd_error = train_SGD(mdl_sgd, M,eta,nb_iter,logging_freq ,dtype, Kern_train_pt,Y_train_pt)
## PRINT ERRORS
train_error_pinv = (1/N_train)*(np.linalg.norm(Y_train-np.dot(Kern_train,c_pinv))**2)
print('\n-----------------')
print(f'N_train={N_train}')
print(f'train_error_pinv = {train_error_pinv}')
print(f'final_sgd_error = {final_sgd_error}')

print(f'condition_number_hessian = np.linalg.cond( np.dot(Kern_train.T,Kern_train)) = {condition_number_hessian}')
print('\a')
#### PLOTTING
X_plot_pytorch = Variable( torch.FloatTensor(X_plot), requires_grad=False)
##
fig1 = plt.figure()
##plots objs
p_sgd, = plt.plot(x_horizontal, [ float(f_val) for f_val in mdl_sgd.forward(X_plot_pytorch).data.numpy() ])
p_pinv, = plt.plot(x_horizontal, np.dot(X_plot,c_pinv))
p_data, = plt.plot(X_train,Y_train,'ro')
# legend
nb_terms = c_pinv.shape[0]
legend_mdl = f'SGD solution standard parametrization, number of monomials={nb_terms}, batch-size={M}, iterations={nb_iter}, step size={eta}'
plt.legend(
       [p_sgd,p_pinv,p_data],
       [legend_mdl,f'linear algebra soln, number of monomials={nb_terms}',f'data points = {N_train}']
   )
#
plt.xlabel('x'), plt.ylabel('f(x)')
plt.show()
## REPORT TIMES
seconds = (time.time() - start_time)
minutes = seconds/ 60
hours = minutes/ 60
print("--- %s seconds ---" % seconds )
print("--- %s minutes ---" % minutes )
print("--- %s hours ---" % hours )
#plt.show()
