import numpy as np

import data_utils

from pdb import set_trace as st

def get_2D_regression_data(N_train,N_val,N_test,lb,ub,f_target):
    ''' '''
    Xtr = np.linspace(lb,ub,N_train).reshape(N_train,1)
    Xv = np.linspace(lb,ub,N_val).reshape(N_train,1)
    Xt = np.linspace(lb,ub,N_test).reshape(N_train,1)
    ''' '''
    Ytr = f_target(Xtr)
    Yv = f_target(Xv)
    Yt = f_target(Xt)
    return Xtr,Ytr, Xv,Yv, Xt,Yt
