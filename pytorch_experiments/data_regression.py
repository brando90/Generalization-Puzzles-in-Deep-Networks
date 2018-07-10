import numpy as np

import data_utils

from pdb import set_trace as st

def get_chebyshev_nodes(lb,ub,N):
    k = np.arange(1,N+1)
    chebyshev_nodes = 0.5*(lb+ub)+0.5*(ub-lb)*np.cos((np.pi*2*k-1)/(2*N))
    return chebyshev_nodes


def get_2D_regression_data_equally_spaced(N_train,N_val,N_test,lb,ub,f_target):
    ''' '''
    Xtr = np.linspace(lb,ub,N_train).reshape(N_train,1)
    Xv = np.linspace(lb,ub,N_val).reshape(N_val,1)
    Xt = np.linspace(lb,ub,N_test).reshape(N_test,1)
    ''' '''
    Ytr = f_target(Xtr)
    Yv = f_target(Xv)
    Yt = f_target(Xt)
    return Xtr,Ytr, Xv,Yv, Xt,Yt

def get_2D_regression_data_chebyshev_nodes(N_train,N_val,N_test,lb,ub,f_target):
    ''' '''
    Xtr = get_chebyshev_nodes(lb,ub,N_train).reshape(N_train,1)
    Xv = get_chebyshev_nodes(lb,ub,N_val).reshape(N_val,1)
    Xt = get_chebyshev_nodes(lb,ub,N_test).reshape(N_test,1)
    ''' '''
    Ytr = f_target(Xtr)
    Yv = f_target(Xv)
    Yt = f_target(Xt)
    return Xtr,Ytr, Xv,Yv, Xt,Yt
