import time
start_time = time.time()

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial.hermite import hermvander

import torch
from torch.autograd import Variable

from maps import NamedDict

from plotting_utils import *

def get_batch2(X,Y,M,dtype):
    '''
    get batch for pytorch model
    '''
    # TODO fix and make it nicer, there is pytorch forum question
    #X,Y = X.data.numpy(), Y.data.numpy()
    X,Y = X, Y
    N = X.size()[0]
    if dtype ==  torch.cuda.FloatTensor:
        batch_indices = torch.cuda.LongTensor( np.random.randint(0,N,size=M) ).type(dtype) # without replacement
    else:
        batch_indices = torch.LongTensor( np.random.randint(0,N,size=M) ) # without replacement
    pdb.set_trace()
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
    for i in range(nb_iter):
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
            delta = eta*W.grad.data
            W.data.copy_(W.data - delta)
        ## train stats
        if i % (nb_iter/10) == 0 or i == 0:
            X_train_, Y_train_ = Variable(X_train), Variable(Y_train)
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
logging_freq = 100
dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor
## SGD params
M = 8
eta = 0.1
nb_iter = 200*10
##
lb,ub = -1,1
freq_sin = 2
f_target = lambda x: np.sin(2*np.pi*freq_sin*x)
N_train = 12
X_train = np.linspace(lb,ub,N_train)
Y_train = f_target(X_train).reshape(N_train,1)
x_horizontal = np.linspace(lb,ub,1000).reshape(1000,1)
## degree of mdl
Degree_mdl = N_train-1
## pseudo-inverse solution
## Standard
poly_feat = PolynomialFeatures(degree=Degree_mdl)
Kern_train = poly_feat.fit_transform(X_train.reshape(N_train,1))
X_plot = poly_feat.fit_transform(x_horizontal)
## Hermite
# Kern_train = hermvander(X_train,Degree_mdl)
# Kern_train = Kern_train.reshape(N_train,Kern_train.shape[1])
# X_plot = hermvander(x_horizontal,Degree_mdl)
# X_plot = X_plot.reshape(1000,X_plot.shape[2])
##
Kern_train_pinv = np.linalg.pinv( Kern_train )
c_pinv = np.dot(Kern_train_pinv, Y_train)
##
condition_number_hessian = np.linalg.cond(Kern_train)
## linear mdl to train with SGD
nb_terms = c_pinv.shape[0]
mdl_sgd = get_sequential_lifted_mdl(nb_monomials=nb_terms,D_out=1, bias=False)
mdl_sgd[0].weight.data.normal_(mean=0,std=0.001)
mdl_sgd[0].weight.data.fill_(0)
## Make polynomial Kernel
Kern_train_pt, Y_train_pt = Variable(torch.FloatTensor(Kern_train).type(dtype), requires_grad=False), Variable(torch.FloatTensor(Y_train).type(dtype), requires_grad=False)
Kern_train_pt, Y_train_pt = torch.FloatTensor(Kern_train).type(dtype), torch.FloatTensor(Y_train).type(dtype)
final_sgd_error = train_SGD(mdl_sgd, M,eta,nb_iter,logging_freq ,dtype, Kern_train_pt,Y_train_pt)
## PRINT ERRORS
train_error_pinv = (1/N_train)*(np.linalg.norm(Y_train-np.dot(Kern_train,c_pinv))**2)
print('\n-----------------')
print(f'train_error_pinv = {train_error_pinv}')
print(f'final_sgd_error = {final_sgd_error}')

print(f'condition_number_hessian = {condition_number_hessian}')
print('\a')
#### PLOTTING
X_plot_pytorch = Variable( torch.FloatTensor(X_plot), requires_grad=False)
##
#fig1 = plt.figure()
#plots objs
#p_sgd, = plt.plot(x_horizontal, [ float(f_val) for f_val in mdl_sgd.forward(X_plot_pytorch).data.numpy() ])
#p_pinv, = plt.plot(x_horizontal, np.dot(X_plot,c_pinv))
#p_data, = plt.plot(X_train,Y_train,'ro')
## legend
#nb_terms = c_pinv.shape[0]
#legend_mdl = f'SGD solution standard parametrization, number of monomials={nb_terms}, batch-size={M}, iterations={nb_iter}, step size={eta}'
#plt.legend(
#        [p_sgd,p_pinv,p_data],
#        [legend_mdl,f'linear algebra soln, number of monomials={nb_terms}',f'data points = {N_train}']
#    )
##
#plt.xlabel('x'), plt.ylabel('f(x)')
#plt.show()
## REPORT TIMES
seconds = (time.time() - start_time)
minutes = seconds/ 60
hours = minutes/ 60
print("--- %s seconds ---" % seconds )
print("--- %s minutes ---" % minutes )
print("--- %s hours ---" % hours )
#plt.show()
