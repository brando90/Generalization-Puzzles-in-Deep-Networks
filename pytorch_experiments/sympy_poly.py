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

def s_Poly(x,c_pinv_relu):
    '''
    Give lowest power fist.

    c_pinv_relu = [ a_0, a_1, ..., a_Deg]
    '''
    a = float(c_pinv_relu[0]) * (x**0)
    for i in range(1,len(c_pinv_relu)):
        coeff = float(c_pinv_relu[i])
        a += coeff*x**i
    return a

class sNN:

    def __init__(self,act,biases,mdl=None,D_layers=None):
        '''
        Note: bias arg is ignored if mdl is an arg.

        D_layers = [D^(0),D^(1),D^(2),D^(3)]

        note: I reversed the dimension from D_in,D_out to D_out,D_in in the internal
        representation of this class to (reluctantly) match pytorch's way of storing matrices
        in its torch.nn.linear class.

        biases = list indicating which layers have a bias
        '''
        # act
        self.act = act
        # params
        self.weights = [None]
        self.biases = [None]
        if mdl != None:
            self.mdl = mdl
            # init weights
            for i in range(1,len(mdl.linear_layers)):
                #print('--i ', i)
                l = mdl.linear_layers[i]
                self.weights.append( Matrix( l.weight.data.numpy() ) ) # [D_out, D_in]
                ##
                bias = biases[i]
                #print('l.bias ', l.bias)
                if bias:
                    #print('i ', i)
                    self.biases.append( Matrix( l.bias.data.numpy() ) ) # [D_out, D_in]
                else:
                    self.biases.append( None )
        elif D_layers != None:
            self.bias = bias
            for i in range(1,len(D_layers)):
                D_in,D_out = D_layers[i-1],D_layers[i]
                self.weights.append( symarray('W^'+str(i), (D_out,D_in)) ) # [D_out, D_in]
                if self.bias:
                    #print('i ', i)
                    self.biases.append( Matrix( symarray('b', (D_out,1)) ) ) # [D_out, D_in]
                else:
                    self.biases.append( None )


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
            #print('d ', d)
            #print('self.weights ', len(self.weights) )
            #print('self.biases ', len(self.biases) )
            W_d = self.weights[d] # [D_out,D_in]
            b_d = self.biases[d] # [D_out,1]
            if b_d != None:
                #pdb.set_trace()
                z = W_d*a + b_d # [D_out,N] = [D_out,D_in] x [D_in,N] .+ [D_out,1]
            else:
                z = W_d*a # [D_out,N] = [D_out,D_in] x [D_in,N]
            a = self.Act(z)
        # last layer
        d = len(self.weights)-1
        W_d = self.weights[d] # [D_out,D_in]
        b_d = self.biases[d] # [D_out,1]
        if b_d != None:
            z = W_d*a + b_d # [D_out,N] = [D_out,D_in] x [D_in,N] .+ [D_out,1]
        else:
            z = W_d*a # [D_out,N] = [D_out,D_in] x [D_in,N]
        y_pred = z
        return y_pred[0]

    def Act(self,z):
        D_out, D_in = z.shape
        for row in range(D_out):
            for col in range(D_in):
                z[row,col] = self.act(z[row,col])
        return z

def simplifies_poly_for_me(expr,*var_list):
    '''
    e.g. poly(x*(x**2 + x - 1)**2)
    '''
    return poly(expr,*var_list)

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

def test_tNN_2_sNN():
    print('---- test_tNN_2_sNN')
    ## tNN
    act = lambda x: x**2 # squared act
    #act = lambda x: F.relu(x) # relu act
    H1,H2 = 2,2
    D0,D1,D2,D3 = 1,H1,H2,1
    D_layers,act = [D0,D1,D2,D3], act
    init_config = Maps( {'name':'w_init_normal','mu':0.0,'std':1.0} )
    #init_config = Maps( {'name':'xavier_normal','gain':1} )
    if init_config.name == 'w_init_normal':
        w_inits = [None]+[lambda x: w_init_normal(x,mu=init_config.mu,std=init_config.std) for i in range(len(D_layers)) ]
    b_inits = [None]+[lambda x: b_fill(x,value=0.1) for i in range(len(D_layers)) ]
    #b_inits = []
    bias = True
    # identity_act = lambda x: x
    # D_1,D_2 = 5,1 # note D^(0) is not present cuz the polyomial is explicitly constructed by me
    # D_layers,act = [D_1,D_2], identity_act
    # init_config = Maps( {'name':'w_init_normal','mu':0.0,'std':1.0} )
    # if init_config.name == 'w_init_normal':
    #     w_inits = [None]+[lambda x: w_init_normal(x,mu=init_config.mu,std=init_config.std) for i in range(len(D_layers)) ]
    # elif init_config.name == 'w_init_zero':
    #     w_inits = [None]+[lambda x: w_init_zero(x) for i in range(len(D_layers)) ]
    # b_inits = [None]+[lambda x: b_fill(x,value=0.1) for i in range(len(D_layers)) ]
    # b_inits = [None]+[lambda x: b_fill(x,value=0.0) for i in range(len(D_layers)) ]
    # b_inits = []
    # bias = False
    ##
    tmdl = NN(D_layers=D_layers,act=act,w_inits=w_inits,b_inits=b_inits,bias=bias)
    ## sNN
    act = sQuad
    smdl = sNN(tmdl,act)
    print(smdl)
    #
    x = symbols('x')
    expr = smdl.forward(x)
    s_expr = poly(expr)
    print( '{} \n {} \n'.format(expr,s_expr) )
    print( 'coefs: {}'.format( s_expr.coeffs() ) )
    print( 'type(coefs): {}'.format( type(s_expr.coeffs()) ) )

def test_purely_symbolic_sNN():
    # H1,H2 = 2,2
    # D0,D1,D2,D3 = 1,H1,H2,1
    # D_layers,act = [D0,D1,D2,D3], sQuad
    H1 = 2
    D0,D1,D3 = 1,H1,1
    D_layers,act = [D0,D1,D3], sQuad
    smdl = sNN(act=act,D_layers=D_layers,bias=True)
    print(smdl)
    #
    x = symbols('x')
    expr = smdl.forward(x)
    s_expr = poly(expr,x)
    print( '\n--->>> unstructed poly \n {} \n\n--->>> structured poly \n {} \n'.format( latex(expr), latex(s_expr) ) )
    print( 'coefs: {}'.format( s_expr.coeffs() ) )
    print( 'type(coefs): {}'.format( type(s_expr.coeffs()) ) )

def main():
    print('--main')
    #test_tNN_2_sNN()
    test_purely_symbolic_sNN()


if __name__ == '__main__':
    main()
