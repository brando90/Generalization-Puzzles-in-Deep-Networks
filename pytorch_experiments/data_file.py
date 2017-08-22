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
    X.shape = X.shape[0],1
    X = Variable(torch.FloatTensor(X).type(dtype), requires_grad=False)
    Y = data_generator.forward(X)
    #pdb.set_trace()
    return Y.data.numpy()

def compare_first_layer(mdl_gen,mdl_sgd):
    W1_g = mdl_gen.linear_layers[1].weight
    W1 = mdl_sgd.linear_layers[1].weight
    print(W1)
    print(W1_g)
    pdb.set_trace()
