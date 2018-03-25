import torch
from torch.autograd import Variable
import torch.nn.functional as F

import pdb

def w_init_zero(l,mu=0,std=1):
    l.weight.data.zero_()

def w_init_normal(l,mu=0,std=1):
    l.weight.data.normal_(mean=mu,std=std)

def b_fill(l,value=0.1):
    l.bias.data.fill_(value=value)

def get_initialization(init_config):
    #init_config = Maps( {'name':'xavier_normal','gain':1} )
    nb_layers = init_config.nb_layers
    if init_config.w_init == 'w_init_normal':
        w_inits = [None]+[lambda x: w_init_normal(x,mu=init_config.mu,std=init_config.std) for i in range(nb_layers) ]
    elif init_config.w_init == 'w_init_zero':
        w_inits = [None]+[lambda x: w_init_zero(x) for i in range(len(D_layers)) ]
    elif init_config.w_init == 'xavier_normal':
        w_inits = [None]+[lambda x: torch.nn.init.xavier_normal(x, gain=init_config.gain) for i in range(nb_layers) ]
    #
    if init_config.bias_init == 'b_fill':
        b_inits = [None]+[lambda x: b_fill(x,value=init_config.bias_value) for i in range(nb_layers) ]
    elif init_config.bias_init == 'empty':
        b_inits = []
    return w_inits, b_inits

##

def lifted_initializer(mdl, init_config):
    mdl[0].weight.data.normal_(mean=init_config.mu,std=init_config.std)
    if mdl[0].bias != None:
        mdl[0].bias.fill_(init_config.bias_value)
