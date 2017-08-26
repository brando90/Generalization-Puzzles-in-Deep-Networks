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

def save_data_set(path, D_layers,act, bias=True,mu=0.0,std=5.0, lb=-1,ub=1,N_train=10,N_test=1000,msg=''):
    dtype = torch.FloatTensor
    #
    data_generator = get_mdl(D_layers,act=act,bias=bias,mu=mu,std=std)
    np_filename = 'data_numpy_D_layers_{}_nb_layers{}_bias{}_mu{}_std{}_N_train_{}_N_test_{}_lb_{}_ub_{}_act_{}_msg_{}'.format(
        D_layers,len(D_layers),bias,mu,std,N_train,N_test,lb,ub,act.__name__,msg
    )
    #
    X_train = np.linspace(lb,ub,N_train)
    X_train.shape = X_train.shape[0],1
    Y_train = get_Y_from_new_net(data_generator=data_generator, X=X_train,dtype=dtype)
    #
    X_test = np.linspace(lb,ub,N_test)
    X_test.shape = X_test.shape[0],1
    Y_test = get_Y_from_new_net(data_generator=data_generator, X=X_test,dtype=dtype)
    #
    np.savez(path.format(np_filename), X_train=X_train,Y_train=Y_train, X_test=X_test,Y_test=Y_test)
    filename = 'data_gen_D_layers_{}_nb_layers{}_bias{}_mu{}_std{}_N_train_{}_N_test_{}_lb_{}_ub_{}_act_{}_msg_{}'.format(
        D_layers,len(D_layers),bias,mu,std,N_train,N_test,lb,ub,act.__name__,msg
    )
    torch.save( data_generator.state_dict(), path.format(filename) )

def get_mdl(D_layers,act,bias=True,mu=0.0,std=5.0):
    init_config_data = Maps( {'w_init':'w_init_normal','mu':mu,'std':std, 'bias_init':'b_fill','bias_value':0.1,'bias':bias ,'nb_layers':len(D_layers)} )
    w_inits_data, b_inits_data = get_initialization(init_config_data)
    data_generator = NN(D_layers=D_layers,act=act,w_inits=w_inits_data,b_inits=b_inits_data,bias=bias)
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

if __name__ == '__main__':
    #act = get_relu_poly_act(degree=2,lb=-1,ub=1,N=100)
    act = quadratic
    # H1 = 2
    # D0,D1,D2 = 1,H1,1
    # D_layers,act = [D0,D1,D2], act

    # H1,H2 = 2,2
    # D0,D1,D2,D3 = 1,H1,H2,1
    # D_layers,act = [D0,D1,D2,D3], act

    H1,H2,H3 = 2,2,2
    D0,D1,D2,D3,D4 = 1,H1,H2,H3,1
    D_layers,act = [D0,D1,D2,D3,D4], act

    # H1,H2,H3,H4 = 2,2,2,2
    # D0,D1,D2,D3,D4,D5 = 1,H1,H2,H3,H4,1
    # D_layers,act = [D0,D1,D2,D3,D4,D5], act
    #
    save_data_set(path='./data/{}',D_layers=D_layers,act=act,bias=True,mu=0.0,std=2.0, lb=-1,ub=1,N_train=10,N_test=1000)
    #save_data_gen(path='./data/{}',D_layers=D_layers,act=act,bias=True,mu=0.0,std=5.0)
    #data_generator = load(path='./data/data_gen_nb_layers3_biasTrue_mu0.0_std5.0')
    print('End! \a')
