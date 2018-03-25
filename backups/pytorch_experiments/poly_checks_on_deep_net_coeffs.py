import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from sympy_poly import *

from maps import NamedDict as Maps
import pdb

def is_sparse_poly(c_sgd,c_pinv,debug=False):
    '''
    Given a deep polynomial network, checks that its not a sparse polynomial.

    If sNN was say 3 layers (then 2^3=8 degree poly) then there was suppose to be
    a degree 8 poly. So there needs to be 9 coefficients for it not be sparse.

    if len(c_sgd) >= len(c_pinv) then poly is NOT sprase because it means pNN has
    enough terms in its polynomial expression to approximate the pinv. is_sprase = False

    if len(c_sgd) < len(c_pinv) then poly IS sparse because pNN does not have enough
    terms to approximate pinv.
    '''
    if debug:
        print( 'len(c_sgd) = ',len(c_sgd))
        print( 'len(c_pinv) = ',len(c_pinv))
        print( 'len(c_sgd) < len(c_pinv) :', len(c_sgd)<len(c_pinv) )
    return len(c_sgd)>=len(c_pinv)

def can_sNN_equal_tNN(s_expr,c_pinv,debug=False):
    '''

    Given the expression of the deep net in symbolic form, uses Grogner basis
    to check if coefficients of the deep net can equal the values of the pinv coeffs.

    Note to find out if thats true we have a non-linear system of equations:
    f(w_i) = f_i
    where f is polynomial in the weights of the deep poly net and f_i are the
    coefficients of the pinv.
    '''
    return None

def check_coeffs_poly(tmdl,act,c_pinv,debug=False):
    # get sNN
    smdl = sNN(act,mdl=tmdl)
    # get expression
    x = symbols('x')
    expr = smdl.forward(x)
    s_expr = poly(expr,x)
    # check it is NOT a sparse polynomial
    c_sgd = np.array( s_expr.coeffs()[::-1] )
    is_sparse = is_sparse_poly(c_sgd,c_pinv,debug=debug)
    print('is poly spare? ',is_sparse)
    if not is_sparse:
        raise ValueError('Polynomial is sparse! Bad, it means pNN can never approximate a fully polnomyal target function.')
    # check coeffs of deep poly can equal coeffs of pinv
    snn_match_nn = can_sNN_equal_tNN(s_expr,c_pinv,debug)
    print('can deep net coeffs equal the pinv values? ',snn_match_nn)
