import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

import pdb

## Activations

def quadratic(x):
    return x**2

def quad_ax2_bx_c(x,a,b,c):
    return a*x**2+b*x+c

def get_relu_poly_act(degree=2,lb=-1,ub=1,N=100):
    X = np.linspace(lb,ub,N)
    Kern = poly_kernel_matrix(X,degree) #[1, x^1, ..., x^D]
    Y = np.maximum(0,X)
    c_pinv = np.dot(np.linalg.pinv( Kern ),Y)
    if degree==2:
        a,b,c = [ float(x) for x in c_pinv ]
        f = lambda x: quad_ax2_bx_c(x,a,b,x)
        f.__name__ = 'quad_ax2_bx_c'
        return f
    #
    def poly_act(x):
        #print('poly_act')
        #print('degree ', degree)
        a = x**0
        for i in range(1,len(c_pinv)):
            coeff = float(c_pinv[i])
            a += coeff*x**i
        #W = Variable( torch.FloatTensor(c_pinv),requires_grad=False)
        #activation = W.mm(X)
        #print(activation)
        return a
    poly_act.__name__ = 'poly_act_degree{}'.format(degree)
    return poly_act

## Kernel methods

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

##

class NN(torch.nn.Module):
    # http://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html#sphx-glr-beginner-examples-nn-two-layer-net-module-py
    # http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-nn
    def __init__(self, D_layers,act,w_inits,b_inits,bias=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.

        D_layers = [D^(0),D^(1),...,D^(L)]
        w_inits = [None,W_f1,...,W_fL]
        b_inits = [None,b_f1,...,b_fL]
        bias = True
        """
        super(type(self), self).__init__()
        # if bias is false then we don't need any init for it (if we do have an init for it and bias=False throw an error)
        #if not bias and (b_inits != [] or b_inits != None):
        #    raise ValueError('bias is {} but b_inits is not empty nor None but isntead is {}'.format(bias,b_inits))
        self.bias = bias
        # actiaction func
        self.act = act
        #create linear layers
        self.linear_layers = torch.nn.ModuleList([None])
        #self.linear_layers = torch.nn.ParameterList([None])
        for d in range(1,len(D_layers)):
            linear_layer = torch.nn.Linear(D_layers[d-1], D_layers[d],bias=bias)
            self.linear_layers.append(linear_layer)
        # initialize model
        for d in range(1,len(D_layers)):
            weight_init = w_inits[d]
            m = self.linear_layers[d]
            weight_init(m)
            if bias:
                bias_init = b_inits[d]
                bias_init(m)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        a = x
        for d in range(1,len(self.linear_layers)-1):
            W_d = self.linear_layers[d]
            z = W_d(a)
            a = self.act(z)
        d = len(self.linear_layers)-1
        y_pred = self.linear_layers[d](a)
        return y_pred

    def numpy_forward(self,x,dtype):
        if type(x) == np.ndarray:
            X = Variable(torch.FloatTensor(X).type(dtype), requires_grad=False)
        y_pred = self.forward(x)
        return y_pred.data.numpy()

    def to_gpu(self,device_id=None):
        torch.nn.Module.cuda(device_id=device_id)

    def get_parameters(self):
        return list(self.parameters())

    def get_nb_params(self):
        return sum(p.numel() for p in model.parameters())

#

def get_all_params(var, all_params):
    if isinstance(var, Parameter):
        all_params[id(var)] = var.nelement()
    elif hasattr(var, "creator") and var.creator is not None:
        if var.creator.previous_functions is not None:
            for j in var.creator.previous_functions:
                get_all_params(j[0], all_params)
    elif hasattr(var, "previous_functions"):
        for j in var.previous_functions:
            get_all_params(j[0], all_params)


##
