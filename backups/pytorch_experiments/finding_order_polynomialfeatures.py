from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sympy import *

import pdb

# nb monomials (n+d,d), d=degree, n=# of inputs

def check(n,d,user_array=None):
    if user_array is None:
        x = np.arange(2,2+n).reshape(1,n) # e.g. array([[2, 3]])
    else:
        x = user_array.reshape(1,n)
    #x = np.arange(2,2+n).reshape(1,n) # e.g. array([[2, 3]])
    print('x = ', x)
    ##
    poly_feat = PolynomialFeatures(d)
    x_poly_feat = poly_feat.fit_transform(x)
    ##
    x_poly_feat_list = [ int(i) for i in x_poly_feat[0]]
    #print('x_poly_feat = ', x_poly_feat)
    #print('x_poly_feat = ', list(x_poly_feat[0]))
    print('x_poly_feat_list = ', x_poly_feat_list)
    return x_poly_feat_list

def check_sympy_degree():
    x3,x2,x1 = symbols('x3 x2 x1')
    poly = Poly( 125*x3**3 + 75*x2*x3**2 + 45*x2**2*x3 + 27*x2**3 + 50*x1*x3**2 + 30*x1*x2*x3 + 18*x1*x2**2 + 20*x1**2*x3 + 12*x1**2*x2
        + 8*x1**3 + 25*x3**2 + 15*x2*x3 + 9*x2**2 + 10*x1*x3 + 6*x1*x2 + 4*x1**2 + 2*x1 + 3*x2 + 5*x3 + 1,(x3,x2,x1) )
    c_grevlex = poly.coeffs(order='grevlex')[::-1]
    c_grlex = poly.coeffs(order='grlex')[::-1]
    print('poly = ',poly)
    return c_grevlex, c_grlex


if __name__ == '__main__':
    #check(n=2,d=3)
    ##
    x_poly_feat_list = check(n=3,d=3,user_array=np.array([2,3,5]))
    ##
    c_grevlex, c_grlex = check_sympy_degree()
    #c_grevlex = len(c_grevlex)*[-1]

    print('c_grevlex = ', c_grevlex)
    print('c_grlex = ', c_grlex)

    print('len(c_grevlex)',len(c_grevlex))
    print('len(c_grlex)',len(c_grlex))
    print('len(x_poly_feat_list)',len(x_poly_feat_list))

    match_grevlex_list = [ c_grevlex[i] == x_poly_feat_list[i] for i in range(len(x_poly_feat_list)) ]
    match_grlex_list = [ c_grlex[i] == x_poly_feat_list[i] for i in range(len(x_poly_feat_list)) ]

    all_match_grevlex = all( match_grevlex_list )
    all_match_grlex = all( match_grlex_list )

    print('match_grevlex_list = ',match_grevlex_list)
    print('match_grlex_list =',match_grlex_list)

    print('all_match_grevlex = ',all_match_grevlex)
    print('all_match_grlex = ',all_match_grlex)
