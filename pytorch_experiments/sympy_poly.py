import torch
from torch.autograd import Variable
import torch.nn.functional as F

from models_pytorch import *
from inits import *

from sympy import *
#from sympy import Matrix
import numpy as np

from maps import NamedDict as Maps

import pdb

'''
-do sNN = implement NN in sympy (with quadratic act func) call it sNN
-[tNN->sNN] = given torch NN (tNN) transform it to sympy NN (sNN)
-fsNN := simplify(sNN) = given sympy NN (sNN) simplify it symbolically to a "flat expression" flat sympy NN (fsNN)
-get_coeffs(fsNN) = extract the coefficients of the flat sympy NN.

https://discuss.pytorch.org/t/why-does-the-linear-module-seems-to-do-unnecessary-transposing/6277
'''

def sReLU(x,threshold=0):
    return Max(threshold,x)

def sQuad(x):
    return sPow(x,2)

def sPow(x,p):
    return Pow(x,p)

class sNN:

    def __init__(self,mdl,act):
        '''
        '''
        self.mdl = mdl
        # params
        self.weights = [None]
        self.biases = [None]
        # init weights
        for i in range(1,len(mdl.linear_layers)):
            l = mdl.linear_layers[i]
            self.weights.append( Matrix(l.weight) ) # [D_out, D_in]
            if l.bias:
                self.biases.append( Matrix(l.bias) ) # [D_out, D_in]

    def forward(self,x):
        '''
        lets start with x being [D^(0),N]

        for 2 hidden layered Net
        implements: Y(x) = W3*[W2*[W1*x
            [D^(3),D^(2)]x[D^(2),D^(1)]x[D^(1),D^(0)]  x  [D^(0),N]
            computation [D^(out),D^(in)]x[D^(in),N]
        {old thinking: Y(x) = x*W1]*W2]*W3] }
        where the ] or [ denote when in the the computation act func is applied
        '''
        a = x # [D^(0),N]
        for d in range(1,len(self.weights)-1):
            W_d = self.weights[d] # [D_out,D_in]
            b_d = self.biases[d] # [D_out,1]
            z = W_d*a + b_d # [D_out,N] = [D_out,D_in] x [D_in,N] .+ [D_out,1]
            a = self.Act(z)
        # last layer
        d = len(self.weights)-1
        W_d = self.weights[d] # [D_out,D_in]
        b_d = self.biases[d] # [D_out,1]
        y_pred = W_d*a + b_d # [D_out,N] = [D_out,D_in] x [D_in,N] .+ [D_out,1]
        return y_pred[0]

    def Act(self,z):
        D_out, D_in = z.shape
        for row in range(D_out):
            for col in range(D_in):
                z[row,col] = self.act(z[row,col])
        return z

def simplifies_poly_for_me(expr):
    '''
    e.g. poly(x*(x**2 + x - 1)**2)
    '''
    return poly(expr)

def substitute_and_eval_numerically():
    '''
    >>> from sympy import *
    >>> from sympy.abc import x
    >>> expr = sin(x)/x
    >>> expr.evalf(subs={x: 3.14})
    0.000507214304613640
    '''
    f = cos(x)+1
    return N(f.subs(x,1))

def check_matrix_multiply_with_polynomial_terms():
    '''
    Matrix * symbols(1,x,x^2) = ax^2 + bx + c
    '''
    x = symbols('x')
    poly_terms = Matrix([x**0,x,x**2]) # [3x1]
    matrix = Matrix([[1,2,3]]) # [1x3]
    expr = matrix*poly_terms # [1x1]
    y = expr[0] # just symbolic expression
    print(expr)
    pdb.set_trace()

def main():
    #c = np.random.rand(3,2)
    print('--main')
    ## tNN
    # H1,H2 = 1,1
    # D0,D1,D2,D3 = 1,H1,H2,1
    # D_layers = [D0,D1,D2,D3]
    # act = lambda x: x**2 # squared act
    # #act = lambda x: F.relu(x) # relu act
    # H1,H2 = 1,1
    # D0,D1,D2,D3 = 1,H1,H2,1
    # D_layers,act = [D0,D1,D2,D3], act
    # init_config = Maps( {'name':'w_init_normal','mu':0.0,'std':1.0} )
    # #init_config = Maps( {'name':'xavier_normal','gain':1} )
    # if init_config.name == 'w_init_normal':
    #     w_inits = [None]+[lambda x: w_init_normal(x,mu=init_config.mu,std=init_config.std) for i in range(len(D_layers)) ]
    # b_inits = [None]+[lambda x: b_fill(x,value=0.1) for i in range(len(D_layers)) ]
    # #b_inits = []
    # bias = True
    identity_act = lambda x: x
    D_1,D_2 = 3,1 # note D^(0) is not present cuz the polyomial is explicitly constructed by me
    D_layers,act = [D_1,D_2], identity_act
    init_config = Maps( {'name':'w_init_normal','mu':0.0,'std':1.0} )
    if init_config.name == 'w_init_normal':
        w_inits = [None]+[lambda x: w_init_normal(x,mu=init_config.mu,std=init_config.std) for i in range(len(D_layers)) ]
    elif init_config.name == 'w_init_zero':
        w_inits = [None]+[lambda x: w_init_zero(x) for i in range(len(D_layers)) ]
    ##b_inits = [None]+[lambda x: b_fill(x,value=0.1) for i in range(len(D_layers)) ]
    ##b_inits = [None]+[lambda x: b_fill(x,value=0.0) for i in range(len(D_layers)) ]
    b_inits = []
    bias = False
    ##
    tmdl = NN(D_layers=D_layers,act=act,w_inits=w_inits,b_inits=b_inits,bias=bias)
    ## sNN
    act = sQuad
    smdl = sNN(tmdl,act)
    print(smdl)



if __name__ == '__main__':
    main()
