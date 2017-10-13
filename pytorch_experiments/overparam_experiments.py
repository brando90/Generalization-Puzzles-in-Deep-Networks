import time
import numpy as np
import sys

import ast

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from maps import NamedDict as Maps
import pdb

from models_pytorch import *
from inits import *
from sympy_poly import *
from poly_checks_on_deep_net_coeffs import *
from data_file import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy

def get_LA_error(X,Y,c,poly_feat):
    N = X.shape[0]
    return (1/N)*(np.linalg.norm(Y-np.dot( poly_feat.fit_transform(X),c) )**2)

def get_nb_monomials(nb_variables,degree):
    return int(scipy.misc.comb(nb_variables+degree,degree))

def get_errors_pinv_mdls(X_train,Y_train,X_test,Y_test,degrees):
    train_errors, test_errors = [], []
    for degree_mdl in degrees:
        poly_feat = PolynomialFeatures(degree=degree_mdl)
        # get mdl
        Kern_train = poly_feat.fit_transform(X_train)
        c_pinv = np.dot(np.linalg.pinv( Kern_train ), Y_train) # c = <K^+,Y>
        # evluate it on train and test
        train_error = get_LA_error(X_train,Y_train,c_pinv,poly_feat)
        test_error = get_LA_error(X_test,Y_test,c_pinv,poly_feat)
        #
        train_errors.append( train_error ), test_errors.append( test_error )
    return train_errors, test_errors

def plot(monomials, train_errors, test_errors, N_train, N_test):
    fig1 = plt.figure()
    p_train, = plt.plot(monomials, train_errors,color='g')
    p_test, = plt.plot(monomials, test_errors,color='b')
    plt.legend([p_train,p_test], ['Train error','Test error'])
    plt.xlabel('Number of monomials' )
    plt.ylabel('Error/loss')
    plt.title('No-overfitting on sythetic, # of training points = {}, # of test points = {} '.format(N_train,N_test))

def my_main(**kwargs):
    ##
    start_time = time.time()
    ##
    plotting = kwargs['plotting'] if 'plotting' in kwargs else False
    D0 = 3
    ## Get data set
    # get points
    Degree_data_set = 10
    nb_monomials_data = get_nb_monomials(nb_variables=D0,degree=Degree_data_set)
    print( 'nb_monomials_data = {}'.format(nb_monomials_data) )
    c_mdl = np.random.rand(nb_monomials_data,1)
    N_train, N_test = 5, nb_monomials_data*2 # N_train <= nb_monomials_data
    X_train, X_test = np.random.rand(N_train,D0), np.random.rand(N_test,D0)
    # get data points
    poly_feat = PolynomialFeatures(degree=Degree_data_set)
    Kern_train = poly_feat.fit_transform(X_train)
    Kern_test = poly_feat.fit_transform(X_test)
    Y_train = np.dot(Kern_train,c_mdl)
    Y_test = np.dot(Kern_test,c_mdl)
    ## get errors from models
    degrees = list(range(1,50))
    train_errors, test_errors = get_errors_pinv_mdls(X_train,Y_train,X_test,Y_test,degrees)
    ## plot them
    monomials = [ get_nb_monomials(nb_variables=D0,degree=d) for d in degrees ]
    if plotting:
        plot(monomials, train_errors, test_errors, N_train, N_test)
        ##
        plt.show()
    ## REPORT TIMES
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    print('\a')
    print("--- %s seconds ---" % seconds )
    print("--- %s minutes ---" % minutes )
    print("--- %s hours ---" % hours )

if __name__ == '__main__':
    ##
    start_time = time.time()
    ##
    my_main(plotting=True)
