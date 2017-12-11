import torch
from torch.autograd import Variable

import numpy as np

import pdb

def GD_step(batch_xs,batch_ys,mdl,eta=0.01):
    N,D0 = batch_xs.size()
    ##
    y_pred = mdl(batch_xs)
    loss = (1/N)*(y_pred - batch_ys).pow(2).sum()
    loss.backward()
    for W in mdl.parameters():
        delta = eta*W.grad.data
        W.data.copy_(W.data - delta)
    ##

def check(noise):
    N=4
    D0=3
    mdl = torch.nn.Sequential(torch.nn.Linear(D0,1,bias=False))
    mdl[0].weight.data.fill_(0)
    ##
    batch_xs = Variable( torch.arange(0,N*D0).view(N,D0), requires_grad=False)
    batch_ys = Variable( torch.arange(1,1+N*D0).view(N,D0), requires_grad=False)
    ##
    GD_step(batch_xs,batch_ys,mdl)
    mdl.zero_grad()
    #
    if noise:
        for W in mdl.parameters():
            #noise = torch.arange(0,D0).view(1,D0)
            W.data.copy_(W.data + W.data)
    #
    GD_step(batch_xs,batch_ys,mdl)
    ##
    for W in mdl.parameters():
        print(f'W.grad.data={W.grad.data}')

def check2(directly_on_data):
    x = Variable( torch.FloatTensor(1).fill_(3), requires_grad=True)
    #print(f'x={x}')
    y = 2*x
    if directly_on_data:
        y.data.copy_(y.data+y.data)
    else:
        y = y + y
    y.backward()
    ##
    print(f'x.grad={x.grad}')

def check3():
    x = Variable( torch.FloatTensor(1).fill_(3), requires_grad=True)
    y = 2*x
    y = y + y
    y.backward()
    print(f'x.grad={x.grad}')

##

if __name__ == '__main__':
    #check(noise=True)
    #check(noise=False)
    check2(True)
    check2(False)
    #check3()
