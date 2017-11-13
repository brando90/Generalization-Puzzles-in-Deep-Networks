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

import scipy
import scipy.io

def get_nb_monomials(nb_variables,degree):
    return int(scipy.misc.comb(nb_variables+degree,degree))

def count_params(mdl):
    '''
    count the number of parameters of a pytorch model
    '''
    tot = 0
    #params = []
    for m in mdl.parameters():
        #print('m: ',m)
        #params.append(m)
        tot += m.nelement() # returns Number of elements = nelement
    #print('\nparams: ',params)
    #pdb.set_trace()
    return tot # sum([m.nelement() for m in mdl.parameters()])

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
    #Y = data_generator.numpy_forward(X,dtype)
    Y = data_generator.forward(X)
    return Y.data.numpy()

def compare_first_layer(mdl_gen,mdl_sgd):
    W1_g = mdl_gen.linear_layers[1].weight
    W1 = mdl_sgd.linear_layers[1].weight
    print(W1)
    print(W1_g)
    pdb.set_trace()

####

def get_func_pointer_poly(c_mdl,Degree_data_set,D0):
    nb_monomials_data = get_nb_monomials(nb_variables=D0,degree=Degree_data_set)
    nb_terms = c_mdl.shape[0]
    if nb_monomials_data != nb_terms:
        raise ValueError(f'nb monomials and nb terms don\'t match: {nb_monomials_data}!={nb_terms}')
    def f_poly(x):
        poly_feat = PolynomialFeatures(degree=Degree_data_set)
        kern = poly_feat.fit_transform(x)
        y = np.dot(kern,c_mdl)
        return y.reshape(y.shape[0],1)
    return f_poly

def get_X_Y_data(f_2_imitate, D0,N,lb,ub):
    if D0 == 1:
        #pdb.set_trace()
        X = np.linspace(lb,ub,N).reshape(N,D0) # [N,D0]
        Y = f_2_imitate(X) #
    elif D0 == 2:
        ##
        X_cord,Y_cord = generate_meshgrid(N,lb,ub)
        Z_data = f_2_imitate(X_cord,Y_cord)
        ##
        X,Y = make_mesh_grid_to_data_set(X_cord,Y_cord,Z=Z_data)
    else:
        raise ValueError('Not implemented')
    return X.reshape(N,D0),Y.reshape(N,1)

def get_c_fit_function(D0,degree_mdl,X,Y):
    ## evaluate target_f on x_points
    if D0 == 1:
        ## copy that f with the target degree polynomial
        #poly_feat = PolynomialFeatures(degree=degree_mdl)
        #Kern = poly_feat.fit_transform(X)
        #c_target = np.dot(np.linalg.pinv( Kern ), Y)
        N,_ = X.shape
        print(X.shape)
        print(Y.shape)
        print(N)
        c_target = np.polyfit(X.reshape((N,)),Y.reshape((N,)),degree_mdl)[::-1]
    elif D0 == 2:
        ## LA models
        poly_feat = PolynomialFeatures(degree=degree_mdl)
        Kern = poly_feat.fit_transform(X)
        c_target = np.dot(np.linalg.pinv( Kern ), Y)
    else:
        # TODO
        raise ValueError(f'Not implemented D0={D0}')
    return c_target

def save_data_poly_fit_to_f_2_imitate(saving,path_to_save, f_2_imitate,Degree_data_set, D0,lb_train,ub_train,lb_test,ub_test, N_train,N_test, N_4_func_approx, noise_train=0,noise_test=0):
    print(f'D0 = {D0}, N_train = {N_train}, N_test = {N_test}')
    ##
    nb_monomials_data = get_nb_monomials(nb_variables=D0,degree=Degree_data_set)
    print(f'> Degree_data_set={Degree_data_set}, nb_monomials_data={nb_monomials_data}')
    ##
    X,Y = get_X_Y_data(f_2_imitate, D0,N_4_func_approx,lb_train,ub_train)
    c_target = get_c_fit_function(D0,Degree_data_set,X,Y)
    f_target = get_func_pointer_poly(c_target,Degree_data_set,D0)
    ##
    X_train,Y_train = get_X_Y_data(f_2_imitate, D0,N_train,lb_train,ub_train)
    X_test,Y_test = get_X_Y_data(f_2_imitate, D0,N_test,lb_test,ub_test)
    Y_train, Y_test = f_target(X_train)+noise_train, f_target(X_test)+noise_test
    Y_train, Y_test = Y_train.reshape(Y_train.shape[0],1), Y_test.reshape(Y_test.shape[0],1)
    #pdb.set_trace()
    ##
    if saving:
        experiment_data = dict(
            X_train=X_train,Y_train=Y_train, X_test=X_test,Y_test=Y_test,
            lb=lb,ub=ub
        )
        np.savez( path_to_save, **experiment_data)
        #np.savez( path_to_save, X_train=X_train,Y_train=Y_train, X_test=X_test,Y_test=Y_test)
    return X_train,Y_train, X_test,Y_test

def save_data_f_2_imitate(saving,path_to_save, f_2_imitate, D0, lb_train,ub_train,lb_test,ub_test, N_train,N_test, noise_train=0,noise_test=0):
    print(f'D0 = {D0}, N_train = {N_train}, N_test = {N_test}')
    ##
    f_target = f_2_imitate
    ##
    X_train,Y_train = get_X_Y_data(f_2_imitate, D0,N_train,lb_train,ub_train)
    X_test,Y_test = get_X_Y_data(f_2_imitate, D0,N_test,lb_test,ub_test)
    Y_train, Y_test = f_target(X_train)+noise_train, f_target(X_test)+noise_test
    Y_train, Y_test = Y_train.reshape(Y_train.shape[0],1), Y_test.reshape(Y_test.shape[0],1)
    #pdb.set_trace()
    ##
    if saving:
        experiment_data = dict(
            X_train=X_train,Y_train=Y_train, X_test=X_test,Y_test=Y_test,
            lb_train=lb_train,ub_train=ub_train,
            lb_test=lb_test,ub_test=ub_test
        )
        np.savez( path_to_save, **experiment_data)
        #np.savez( path_to_save, X_train=X_train,Y_train=Y_train, X_test=X_test,Y_test=Y_test)
    return X_train,Y_train, X_test,Y_test

####

def get_stand_param(D_layers,mu,std):
    ## set up dimensions and degree
    D = D_layers[0]
    D_out = D_layers[-1]
    nb_layers = len(D_layers)-1
    nb_hidden_layers = nb_layers-1 #note the last "layer" is a summation layer for regression and does not increase the degree of the polynomial
    Degree_mdl = adegree**( nb_hidden_layers ) # only hidden layers have activation functions
    ##
    poly_feat = PolynomialFeatures(degree=Degree_mdl)
    nb_monomials = int(scipy.misc.comb(D+Degree_mdl,Degree_mdl))
    ## get generator model
    c_generator = np.random.normal(loc=mu,scale=std,size=nb_monomials).reshape(D_out,nb_monomials)
    ##
    x = np.random.rand(3,D)
    X = poly_feat.fit_transform(x)
    #pdb.set_trace()
    if X.shape[1] != c_generator.shape[1]:
        raise ValueError( 'Dimension of PolynomialFeatures {} does not match that of c_generator {}'.format(X.shape[1],c_generator.shape[0]) )
    data_generator = torch.nn.Sequential(torch.nn.Linear(nb_monomials,D_out,bias=False))
    data_generator[0].weight.data.copy_( torch.FloatTensor(c_generator) )
    return data_generator

def save_data_set(path, type_mdl, D_layers,act, biases,mu=0.0,std=5.0, lb=-1,ub=1,N_train=10,N_test=1000,msg='',visualize=False,save_data=True):
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
    if type_mdl == 'WP':
        data_generator = get_mdl(D_layers,act=act,biases=biases,mu=mu,std=std)
        #data_generator.linear_layers[2].weight.data[0][0] = 0
        #data_generator.linear_layers[2].weight.data[0][1] = 0
        #print(data_generator.linear_layers[2].weight.data)
        #pdb.set_trace()
    elif type_mdl == 'SP':
        data_generator = get_stand_param(D_layers,mu,std)
        coefficients = data_generator[0].weight.data.numpy()
        print('coefficients = {}'.format(coefficients))
    else:
        raise ValueError( 'type_mdl {} doesn not exists'.format(type_mdl) )
    #
    nb_params = count_params(data_generator)
    print( 'nb_params = {}'.format(nb_params) )
    np_filename = 'data_numpy_type_mdl={}_D_layers_{}_nb_layers{}_bias{}_mu{}_std{}_N_train_{}_N_test_{}_lb_{}_ub_{}_act_{}_nb_params_{}_msg_{}'.format(
        type_mdl,D_layers,len(D_layers),biases,mu,std,N_train,N_test,lb,ub,act.__name__,nb_params,msg
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
        X_train = np.random.uniform(low=lb,high=ub,size=(N_train,D))
        X_test = np.random.uniform(low=lb,high=ub,size=(N_test,D))
    #
    if type_mdl == 'WP':
        K_X_train = X_train
        K_X_test = X_test
    elif type_mdl == 'SP':
        poly_feat = PolynomialFeatures(degree=Degree_mdl)
        K_X_train = poly_feat.fit_transform(X_train)
        K_X_test = poly_feat.fit_transform(X_test)
    #
    print('before y train')
    Y_train = get_Y_from_new_net(data_generator=data_generator, X=K_X_train,dtype=dtype)
    print('after y train')
    #
    print('before y test')
    Y_test = get_Y_from_new_net(data_generator=data_generator, X=K_X_test,dtype=dtype)
    print('after y test')
    #
    if save_data:
        np.savez(path.format(np_filename), X_train=X_train,Y_train=Y_train, X_test=X_test,Y_test=Y_test)
        filename = 'data_gen_type_mdl={}_D_layers_{}_nb_layers{}_bias{}_mu{}_std{}_N_train_{}_N_test_{}_lb_{}_ub_{}_act_{}_nb_params_{}_msg_{}'.format(
            type_mdl,D_layers,len(D_layers),biases,mu,std,N_train,N_test,lb,ub,act.__name__,nb_params,msg
        )
        torch.save( data_generator.state_dict(), path.format(filename) )
    #
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

##

def get_input_X():
    X_train, X_test = 2*np.random.rand(N_train,D0)-1, 2*np.random.rand(N_test,D0)-1

def generate_h_gabor_1d(X,noise=0):
    Z = np.exp( -(X**2) )*np.cos(4*np.pi*(X))
    return Z+noise

def generate_h_add_1d(X,noise=0):
    Z = np.sin(10*np.pi*X) + np.cos(8*np.pi*X)
    return Z+noise

def my_sin(X,period=10):
    return np.sin(period*np.pi*X)

def generate_h_add(X,noise=0):
    x,y = X[:,0], X[:,1]
    Z = np.sin(1.8*np.pi*x) + np.cos(1.5*np.pi*y)
    return Z+noise

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

##

def main_poly():
    saving=True
    #f_2_imitate = lambda x: np.sin(2*np.pi*x)
    freq_sin = 4
    f_2_imitate = lambda x: np.sin(2*np.pi*freq_sin*x).reshape(x.shape[0],1)
    Degree_data_set = 4
    D0 = 1
    ##
    eps_train = 0
    lb_train,ub_train = 0+eps_train,1-eps_train
    eps_test = 0.2
    lb_test,ub_test = 0+eps_test,1-eps_test
    ##
    N_train, N_test = 7,200
    ##
    #file_name=f'degree{Degree_data_set}_fit_2_sin_N_train_{N_train}_N_test_{N_test}'
    file_name=f'sin_freq_sin_{freq_sin}_N_train_{N_train}_N_test_{N_test}_lb_train,ub_train_{lb_train,ub_train}_lb_test,ub_test_{lb_test,ub_test}'
    path_to_save = f'./data/{file_name}'
    print(f'path_to_save = {path_to_save}')
    ##
    # X_train,Y_train, X_test,Y_test = save_data_poly_fit_to_f_2_imitate(
    #     saving,path_to_save, f_2_imitate,Degree_data_set, D0,lb,ub,N_train,N_test,N_4_func_approx,
    #     noise_train=0,noise_test=0
    # )
    X_train,Y_train, X_test,Y_test = save_data_f_2_imitate(
        saving,path_to_save, f_2_imitate, D0,
        lb_train,ub_train,lb_test,ub_test,
        N_train,N_test,
        noise_train=0,noise_test=0
    )
    print(f'X_train.shape={X_train.shape},Y_train={Y_train.shape}, X_test={X_test.shape},Y_test={Y_test.shape}')
    ##
    # plt.plot(X_train,Y_train)
    # plt.plot(X_test,Y_test)
    # x = np.linspace(0,1,50)
    # y = f_2_imitate(x)
    # plt.plot(x,y)
    # plt.show()

##

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
    ## activation params
    #adegree = 2
    #alb, aub = -100, 100
    #aN = 100
    #act = get_relu_poly_act(degree=adegree,lb=alb,ub=aub,N=aN) # ax**2+bx+c
    ##
    adegree = 2
    ax = np.concatenate( (np.linspace(-20,20,100), np.linspace(-10,10,1000)) )
    aX = np.concatenate( (ax,np.linspace(-2,2,100000)) )
    act, c_pinv_relu = get_relu_poly_act2(aX,degree=adegree) # ax**2+bx+c, #[1, x^1, ..., x^D]
    print('c_pinv_relu = ', c_pinv_relu)
    #act = relu
    #act = lambda x: x
    #act.__name__ = 'linear'
    act.adegree = adegree
    ##
    D0 = 2
    H1 = 1
    D0,D1,D2 = D0,H1,1
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
    biases = [None] + [True] + (nb_layers-1)*[False] #bias only in first layer
    #biases = [None] + (nb_layers)*[True] # biases in every layer
    #msg = '1st_2nd_units_are_zero'
    msg = ''
    mu,std = 0.0,5.0
    #N_train, N_test= 4,5041
    N_train, N_test= 4,25
    ##
    visualize=False
    save_data = True
    type_mdl = 'WP'
    #type_mdl = 'SP'
    #save_data_set_mdl_sgd(path='./data/{}', run_type='h_add', lb=-1,ub=1,N_train=35,N_test=5041,msg='',visualize=False)
    #save_data_set(path='./data/{}',type_mdl=type_mdl,D_layers=D_layers,act=act,biases=biases,mu=mu,std=std, lb=-1,ub=1,N_train=N_train,N_test=N_test,msg=msg,visualize=visualize,save_data=save_data)
    #data_generator = load(path='./data/data_gen_nb_layers3_biasTrue_mu0.0_std5.0')
    main_poly()
    print('End! \a')
