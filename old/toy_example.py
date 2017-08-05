import numpy as np

import torch
from torch.autograd import Variable

import pdb

def get_batch2(X,Y,M,dtype):
    X,Y = X.data.numpy(), Y.data.numpy()
    N = len(Y)
    valid_indices = np.array( range(N) )
    batch_indices = np.random.choice(valid_indices,size=M,replace=False)
    batch_xs = torch.FloatTensor(X[batch_indices,:]).type(dtype)
    batch_ys = torch.FloatTensor(Y[batch_indices]).type(dtype)
    return Variable(batch_xs, requires_grad=False), Variable(batch_ys, requires_grad=False)

def poly_kernel_matrix( x,D ):
    N = len(x)
    Kern = np.zeros( (N,D+1) )
    for n in range(N):
        for d in range(D+1):
            Kern[n,d] = x[n]**d;
    return Kern

## data params
N=5 # data set size
Degree=4 # number dimensions/features
D_sgd = Degree+1
##
x_true = np.linspace(0,1,N) # the real data points
y = np.sin(2*np.pi*x_true)
y.shape = (N,1)
## TORCH
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
X_mdl = poly_kernel_matrix( x_true,Degree )
X_mdl = Variable(torch.FloatTensor(X_mdl).type(dtype), requires_grad=False)
y = Variable(torch.FloatTensor(y).type(dtype), requires_grad=False)
## SGD mdl
w_init = torch.zeros(D_sgd,1).type(dtype)
W = Variable(w_init, requires_grad=True)
M = 5 # mini-batch size
eta = 0.1 # step size
for i in range(500):
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
    loss = (1/N)*(y_pred - batch_ys).pow(2).sum()
    # Use autograd to compute the backward pass. Now w will have gradients
    loss.backward()
    # Update weights using gradient descent; w1.data are Tensors,
    # w.grad are Variables and w.grad.data are Tensors.
    W.data -= eta * W.grad.data
    # Manually zero the gradients after updating weights
    W.grad.data.zero_()

#
c_sgd = W.data.numpy()
X_mdl = X_mdl.data.numpy()
y = y.data.numpy()
#
Xc_pinv = np.dot(X_mdl,c_sgd)
print('J(c_sgd) = ', (1/N)*(np.linalg.norm(y-Xc_pinv)**2) )
print('loss = ',loss.data[0])
