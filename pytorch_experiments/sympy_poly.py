from sympy import *
#from sympy import Matrix
import numpy as np

#from maps import NamedDict as Maps

import pdb

'''
-do sNN = implement NN in sympy (with quadratic act func) call it sNN
-[tNN->sNN] = given torch NN (tNN) transform it to sympy NN (sNN)
-fsNN := simplify(sNN) = given sympy NN (sNN) simplify it symbolically to a "flat expression" flat sympy NN (fsNN)
-get_coeffs(fsNN) = extract the coefficients of the flat sympy NN.

https://discuss.pytorch.org/t/why-does-the-linear-module-seems-to-do-unnecessary-transposing/6277
'''

class sNN:

    def __init__(self,mdl):
        '''
        '''
        # params
        self.weights = [None]
        self.biases = [None]
        # init weights
        for l in mdl.linear_layers:
            self.weights.append( Matrix(l.weight) ) # [D_out, D_in]
            if l.bias:
                self.biases.append( Matrix(l.bias) ) # [D_out, D_in]

    def forward(self,x):
        '''
        lets start with x being [1, D^(0)]

        for 2 hidden layered Net
        implements: Y(x) = W3*[W2*[W1*x
            [D^(3),D^(2)]x[D^(2),D^(1)]x[D^(1),D^(0)]  x  [D^(0),N]
            computation [D^(out),D^(in)]x[D^(in),N]
        {old thinking: Y(x) = x*W1]*W2]*W3] }
        where the ] or [ denote when in the the computation act func is applied
        '''
        a = x
        for d in range(1,len(self.linear_layers)-1):
            W_d = self.linear_layers[d]
            z = W_d(a)
            a = self.act(z)
        d = len(self.linear_layers)-1
        y_pred = self.linear_layers[d](a)
        return y_pred

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
    check_matrix_multiply_with_polynomial_terms()



if __name__ == '__main__':
    main()
