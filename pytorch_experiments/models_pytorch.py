import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

import unittest

import pdb

## Activations

def relu(x):
    if type(x) == np.ndarray:
        return np.maximum(0,x)
    else:
        #x = torch.FloatTensor(x)
        return F.relu(x)

def quadratic(x):
    return x**2

def quad_ax2_bx_c(x,a,b,c):
    return a*x**2+b*x+c


def get_relu_poly_act2(X,degree=2):
    #Kern = poly_kernel_matrix(X,degree) #[1, x^1, ..., x^D]
    Y = np.maximum(0,X)
    #c_pinv = np.dot(np.linalg.pinv( Kern ),Y)
    c_pinv = np.polyfit(X, Y, degree)[::-1]
    #
    def poly_act(x):
        #print('poly_act')
        #print('degree ', degree)
        a = float(c_pinv[0]) * (x**0)
        for i in range(1,len(c_pinv)):
            coeff = float(c_pinv[i])
            a += coeff * (x**i)
        #W = Variable( torch.FloatTensor(c_pinv),requires_grad=False)
        #activation = W.mm(X)
        #print(activation)
        return a
    poly_act.__name__ = 'poly_act_degree{}'.format(degree)
    return poly_act, c_pinv

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
    x = single real number data value
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

def generate_all_tuples_for_monomials(N,D):
    if D == 0:
        print('\n---')
        print('=> D ', D)
        seq0 = N*[0]
        sequences_degree_0 = [seq0]
        S_0 = {0:sequences_degree_0}
        print('S_0 ', S_0)
        return S_0
    else:
        # S_all = [ k->S_D ] ~ [ k->[seq0,...,seqK]]
        S_all = generate_all_tuples_for_monomials(N,D-1)# S^* = (S^*_D-1) U S_D
        print('\n---')
        print('=> D ', D)
        print(S_all)
        #
        S_D_current = []
        # for every prev set of degree tuples
        #for d in range(len(S_all.items())): # d \in [0,...,D_current]
        #print('d ', d)
        d = D-1
        d_new = D - d # get new valid degree number
        print('>d_new ', d_new)
        # for each sequences, create the new valid degree tuple
        S_all_seq_for_deg_d = S_all[d]
        print('S_all_seq_for_deg_d ', S_all_seq_for_deg_d)
        for seq in S_all[d]:
            print('seq ', seq)
            for pos in range(N):
                seq_new = seq[:]
                seq_new[pos] = seq_new[pos] + d_new # seq elements dd to D
                print('seq_new ', seq_new)
                if seq_new not in S_D_current:
                    S_D_current.append(seq_new)
                    print('S_D_current after addition: ',S_D_current)
                print()
        #
        print('before appending D ', S_all)
        S_all[D] = S_D_current
        print('after appending D ', S_all)
        print('+++')
        return S_all

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

class TestStringMethods(unittest.TestCase):

    def test_degree_zero(self):
        S_0 = generate_all_tuples_for_monomials(N=1,D=0)
        self.assertEqual(S_0,{0:[ [0] ]})
        S_0 = generate_all_tuples_for_monomials(N=2,D=0)
        self.assertEqual(S_0,{0:[ [0,0] ]})
        S_0 = generate_all_tuples_for_monomials(N=3,D=0)
        self.assertEqual(S_0,{0:[ [0,0,0] ]})

    def test_degree_one(self):
        S_0 = generate_all_tuples_for_monomials(N=1,D=1)
        self.assertEqual(S_0,{0:[ [0] ], 1:[ [1] ]})
        S_0 = generate_all_tuples_for_monomials(N=2,D=1)
        self.assertEqual(S_0,{0:[ [0,0] ], 1:[ [1,0], [0,1] ]})
        S_0 = generate_all_tuples_for_monomials(N=3,D=1)
        self.assertEqual(S_0,{0:[ [0,0,0] ], 1:[ [1,0,0], [0,1,0], [0,0,1] ]})

    def test_degree_two(self):
        print('\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        S_0 = generate_all_tuples_for_monomials(N=1,D=2)
        self.assertEqual(S_0,{0:[ [0] ], 1:[ [1] ], 2:[ [2] ]})
        S_0 = generate_all_tuples_for_monomials(N=2,D=2)
        print(S_0)
        self.assertEqual(S_0,{0:[ [0,0] ], 1:[ [1,0], [0,1] ], 2:[ [2,0], [1,1], [0,2] ]})
        S_0 = generate_all_tuples_for_monomials(N=3,D=2)
        self.assertEqual(S_0,{0:[ [0,0,0,0] ], 1:[ [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] ],2:[ [2,0,0,0], [1,1,0,0] ]})

if __name__ == '__main__':
    unittest.main()
