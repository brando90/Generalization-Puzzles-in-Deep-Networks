import numpy as np
import torch
from torch.autograd import Variable

from pdb import set_trace as st

####

def l2_np_loss(y,y_):
    N,_ = y.shape
    return (1.0/N)*np.linalg.norm(y-y_,2)**2

####

def data2FloatTensor(Xtr,Xv,Xt):
    Xtr,Xv,Xt = torch.FloatTensor(Xtr),torch.FloatTensor(Xv),torch.FloatTensor(Xt)
    return Xtr,Xv,Xt

def data2LongTensor(Ytr,Yv,Yt):
    Ytr,Yv,Yt = torch.LongTensor(Ytr),torch.LongTensor(Yv),torch.LongTensor(Yt)
    return Ytr,Yv,Yt

def data2torch_classification(Xtr,Ytr,Xv,Yv,Xt,Yt):
    Xtr,Xv,Xt = torch.FloatTensor(Xtr),torch.FloatTensor(Xv),torch.FloatTensor(Xt)
    Ytr,Yv,Yt = torch.LongTensor(Ytr),torch.LongTensor(Yv),torch.LongTensor(Yt)
    return Xtr,Ytr,Xv,Yv,Xt,Yt

def data2torch_regression(Xtr,Ytr,Xv,Yv,Xt,Yt):
    Xtr,Xv,Xt = torch.FloatTensor(Xtr),torch.FloatTensor(Xv),torch.FloatTensor(Xt)
    Ytr,Yv,Yt = torch.FloatTensor(Ytr),torch.FloatTensor(Yv),torch.FloatTensor(Yt)
    return Xtr,Ytr,Xv,Yv,Xt,Yt

def data2torch_variable(Xtr,Ytr,Xv,Yv,Xt,Yt):
    Xtr,Xv,Xt = Variable(Xtr,requires_grad=False),Variable(Xv,requires_grad=False),Variable(Xt,requires_grad=False)
    Ytr,Yv,Yt = Variable(Ytr,requires_grad=False),Variable(Yv,requires_grad=False),Variable(Yt,requires_grad=False)
    return Xtr,Ytr,Xv,Yv,Xt,Yt

####

def generate_meshgrid(N,start_val,end_val):
    '''
    returns: the grid values for the x coordiantes in X_grid and for the
             y coordiantes in Y_grid.
    '''
    sqrtN = int(np.ceil(N**0.5)) #N = sqrtN*sqrtN
    if N**0.5 != int(N**0.5): # check if N_sqrt has a fractional part
        print('WARNING: your data size is not a perfect square. Could lead data set to be of an unexpected size.')
    N = sqrtN*sqrtN
    x_range = np.linspace(start_val, end_val, sqrtN)
    y_range = np.linspace(start_val, end_val, sqrtN)
    ## make meshgrid
    (X,Y) = np.meshgrid(x_range, y_range)
    return X,Y

def make_mesh_grid_to_data_set_with_f(f_target,X, Y):
    '''
        want to make data set as:
        ( x = [x1, x2], z = f(x,y) )
        X = [N, D], Z = [Dout, N] = [1, N]
    '''
    (dim_x, dim_y) = X.shape
    N = dim_x * dim_y
    X_data = np.zeros((N,2))
    Y_data = np.zeros((N,1))
    i = 0
    for dx in range(dim_x):
        for dy in range(dim_y):
            # input val
            x = X[dx, dy]
            y = Y[dx, dy]
            x_data = np.array([x, y])
            # func val
            z = f_target( np.array([x,y]) )
            y_data = z
            # load data set
            X_data[i,:] = x_data
            Y_data[i,:] = y_data
            i=i+1;
    return X_data,Y_data

def make_mesh_grid_to_data_set(X, Y, Z=None):
    '''
        want to make data set as:
        ( x = [x1, x2], z = f(x,y) )
        X = [N, D], Z = [Dout, N] = [1, N]
    '''
    (dim_x, dim_y) = X.shape
    N = dim_x * dim_y
    X_data = np.zeros((N,2))
    Y_data = np.zeros((N,1))
    i = 0
    for dx in range(dim_x):
        for dy in range(dim_y):
            # input val
            x = X[dx, dy]
            y = Y[dx, dy]
            x_data = np.array([x, y])
            # func val
            if np.any(Z) == None:
                z = None
                y_data = None
            else:
                z = Z[dx, dy]
                y_data = z
            # load data set
            X_data[i,:] = x_data
            Y_data[i,:] = y_data
            i=i+1;
    return X_data,Y_data

def make_meshgrid_data_from_training_data(X_data, Y_data):
    N, _ = X_data.shape
    sqrtN = int(np.ceil(N**0.5))
    dim_y = sqrtN
    dim_x = dim_y
    shape = (sqrtN,sqrtN)
    X = np.zeros(shape)
    Y = np.zeros(shape)
    Z = np.zeros(shape)
    i = 0
    for dx in range(dim_x):
        for dy in range(dim_y):
            #x_vec = X_data[:,i]
            #x,y = x_vec(1),x_vec(2)
            x,y = X_data[i,:]
            #x = x_vec(1);
            #y = x_vec(2);
            z = Y_data[i,:]
            X[dx,dy] = x
            Y[dx,dy] = y
            Z[dx,dy] = z
            i = i+1;
    return X,Y,Z
