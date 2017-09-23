import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from models_pytorch import *
from inits import *
from sympy_poly import *
from poly_checks_on_deep_net_coeffs import *
from data_file import *

from maps import NamedDict as Maps
import pdb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm

def generate_meshgrid(N,start_val,end_val):
    sqrtN = int(np.ceil(N**0.5)) #N = sqrtN*sqrtN
    if N**0.5 != int(N**0.5): # check if N_sqrt has a fractional part
        print('WARNING: your data size is not a perfect squre. Could lead data set to be of an unexpected size.')
    N = sqrtN*sqrtN
    x_range = np.linspace(start_val, end_val, sqrtN)
    y_range = np.linspace(start_val, end_val, sqrtN)
    ## make meshgrid
    (X,Y) = np.meshgrid(x_range, y_range)
    return X,Y

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
    return X_data, Y_data

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

##

def get_Y_from_new_net(data_generator, X,dtype):
    '''
    Note that if the list of initialization functions simply calls the random initializers
    of the weights of the model, then the model gets bran new values (i.e. the issue
    of not actually getting a different net should NOT arise).

    The worry is that the model learning from this data set would be the exact same
    NN. Its fine if the two models come from the same function class but its NOT
    meaningful to see if the model can learn exactly itself.
    '''
    X = Variable(torch.FloatTensor(X).type(dtype), requires_grad=False)
    Y = data_generator.numpy_forward(X,dtype)
    return Y

def compare_first_layer(mdl_gen,mdl_sgd):
    W1_g = mdl_gen.linear_layers[1].weight
    W1 = mdl_sgd.linear_layers[1].weight
    print(W1)
    print(W1_g)
    pdb.set_trace()

####

def save_data_set(path, D_layers,act, biases,mu=0.0,std=5.0, lb=-1,ub=1,N_train=10,N_test=1000,msg='',visualize=False,save_data=True):
    dtype = torch.FloatTensor
    #
    adegree = act.adegree
    D = D_layers[0]
    nb_layers = len(D_layers)-1
    nb_hidden_layers = nb_layers-1 #note the last "layer" is a summation layer for regression and does not increase the degree of the polynomial
    Degree_mdl = adegree**( nb_hidden_layers ) # only hidden layers have activation functions
    expected_nb_monomials = int(scipy.misc.comb(D+Degree_mdl,Degree_mdl))
    print('expected_nb_monomials = {}'.format(expected_nb_monomials))
    #
    data_generator = get_mdl(D_layers,act=act,biases=biases,mu=mu,std=std)
    np_filename = 'data_numpy_D_layers_{}_nb_layers{}_bias{}_mu{}_std{}_N_train_{}_N_test_{}_lb_{}_ub_{}_act_{}_msg_{}'.format(
        D_layers,len(D_layers),biases,mu,std,N_train,N_test,lb,ub,act.__name__,msg
    )
    #
    if D==1:
        X_train = np.linspace(lb,ub,N_train).reshape(N_train,D)
        X_test = np.linspace(lb,ub,N_train).reshape(N_train,D)
    elif D ==  2:
        Xm_train,Ym_train = generate_meshgrid(N_train,lb,ub)
        X_train,_ = make_mesh_grid_to_data_set(Xm_train,Ym_train)
        #
        Xm_test,Ym_test = generate_meshgrid(N_test,lb,ub)
        X_test,_ = make_mesh_grid_to_data_set(Xm_test,Ym_test)
    else:
        pass
    #
    Y_train = get_Y_from_new_net(data_generator=data_generator, X=X_train,dtype=dtype)
    #
    Y_test = get_Y_from_new_net(data_generator=data_generator, X=X_test,dtype=dtype)
    #
    if save_data:
        np.savez(path.format(np_filename), X_train=X_train,Y_train=Y_train, X_test=X_test,Y_test=Y_test)
        filename = 'data_gen_D_layers_{}_nb_layers{}_bias{}_mu{}_std{}_N_train_{}_N_test_{}_lb_{}_ub_{}_act_{}_msg_{}'.format(
            D_layers,len(D_layers),biases,mu,std,N_train,N_test,lb,ub,act.__name__,msg
        )
        torch.save( data_generator.state_dict(), path.format(filename) )
    if visualize:
        if D==1:
            pass
        elif D==2:
            Xp,Yp,Zp = make_meshgrid_data_from_training_data(X_data=X_test, Y_data=Y_test)
            ##
            fig = plt.figure()
            #ax = fig.gca(projection='3d')
            ax = Axes3D(fig)
            surf = ax.plot_surface(Xp,Yp,Zp, cmap=cm.coolwarm)
            plt.title('Test function')
            ##
            plt.show()

def get_mdl(D_layers,act,biases,mu=0.0,std=5.0):
    init_config_data = Maps( {'w_init':'w_init_normal','mu':mu,'std':std, 'bias_init':'b_fill','bias_value':0.1,'bias':biases ,'nb_layers':len(D_layers)} )
    w_inits_data, b_inits_data = get_initialization(init_config_data)
    data_generator = NN(D_layers=D_layers,act=act,w_inits=w_inits_data,b_inits=b_inits_data,biases=biases)
    return data_generator

def save_data_gen(path,D_layers,act,bias=True,mu=0.0,std=5.0):
    # data_generator = get_mdl(D_layers,act=act,bias=bias,mu=mu,std=std)
    # filename = 'data_gen_nb_layers{}_bias{}_mu{}_std{}'.format(str(len(D_layers)),str(bias),str(mu),str(std))
    # torch.save(data_generator.state_dict(),path.format(filename))
    pass

def load(path):
    # bias = True
    # mu, std = 0, 0
    # D_layers,act = [], lambda x: x**2
    # data_generator = get_mdl(D_layers,act=act,bias=bias,mu=mu,std=std)
    # data_generator.load_state_dict(torch.load(path))
    # return data_generator
    pass

def generate_meshgrid_h_add(N=60000,start_val=-1,end_val=1):
    (X,Y) = generate_meshgrid(N,start_val,end_val)
    #Z = sin(2*pi*X) + 4*(Y - 0.5).^2; %% h_add
    #Z = np.sin(2*np.pi*X) + 4*np.power(Y - 0.5, 2) # h_add
    #Z = np.sin(2*np.pi*X)
    #Z = np.sin(2*np.pi*X) + np.cos(1*np.pi*Y)
    Z = np.sin(1.8*np.pi*X) + np.cos(1.5*np.pi*Y)
    #pdb.set_trace()
    return X,Y,Z

def generate_meshgrid_h_gabor(N=60000,start_val=-1,end_val=1):
    (X,Y) = generate_meshgrid(N,start_val,end_val)
    Z = np.exp( -(X**2 + Y**2) )*np.cos(2*np.pi*(X+Y))
    return X,Y,Z

def visualize(X,Y,Z,title_name='Test function'):
    #Xp,Yp,Zp = make_meshgrid_data_from_training_data(X_data=X_test, Y_data=Y_test)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X,Y,Z, cmap=cm.coolwarm)
    plt.title(title_name)
    plt.show()

if __name__ == '__main__':
    #X,Y,Z = generate_meshgrid_h_add(N=60000,start_val=-1,end_val=1)
    #visualize(X,Y,Z)
    adegree=3
    act = get_relu_poly_act(degree=adegree,lb=-1,ub=1,N=100)
    act.adegree = adegree
    #act = quadratic
    H1 = 10
    D0,D1,D2 = 2,H1,1
    D_layers,act = [D0,D1,D2], act

    # H1,H2 = 15,15
    # D0,D1,D2,D3 = 2,H1,H2,1
    # D_layers,act = [D0,D1,D2,D3], act

    # H1,H2,H3 = 11,11,11
    # D0,D1,D2,D3,D4 = 2,H1,H2,H3,1
    # D_layers,act = [D0,D1,D2,D3,D4], act

    # H1,H2,H3,H4 = 2,2,2,2
    # D0,D1,D2,D3,D4,D5 = 1,H1,H2,H3,H4,1
    # D_layers,act = [D0,D1,D2,D3,D4,D5], act

    nb_layers = len(D_layers)-1
    #biases = [None] + [True] + (nb_layers-1)*[False] #bias only in first layer
    biases = [None] + (nb_layers)*[True] # biases in every layer
    msg = ''
    mu,std = 0.0,5.0
    N_train, N_test= 9,5041
    ##
    save_data = False
    #save_data_set_mdl_sgd(path='./data/{}', run_type='h_add', lb=-1,ub=1,N_train=35,N_test=5041,msg='',visualize=False)
    save_data_set(path='./data/{}',D_layers=D_layers,act=act,biases=biases,mu=mu,std=std, lb=-1,ub=1,N_train=N_train,N_test=N_test,msg=msg,visualize=True,save_data=save_data)
    #data_generator = load(path='./data/data_gen_nb_layers3_biasTrue_mu0.0_std5.0')
    print('End! \a')
