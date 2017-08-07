import torch
from torch.autograd import Variable
import torch.nn.functional as F

def w_init_normal(l,mu=0,std=1):
    l.weight.data.normal_(mean=mu,std=std)

def b_fill(l,value=0.1):
    l.bias.data.fill_(value=value)
    
