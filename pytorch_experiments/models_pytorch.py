import torch
from torch.autograd import Variable
import torch.nn.functional as F

import pdb

class NN(torch.nn.Module):
    # http://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html#sphx-glr-beginner-examples-nn-two-layer-net-module-py
    # http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-nn
    def __init__(self, D_layers,act,w_inits,b_inits,bias=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.

        D_layers = [D^(0),D^(1),...,D^(L)]
        w_inits = [None,W_f1,...,W_fL]
        b_inits = [None,b_f1,...,b_fL]
        bias = True
        """
        super(type(self), self).__init__()
        # if bias is false then we don't need any init for it (if we do have an init for it and bias=False throw an error)
        #if not bias and (b_inits != [] or b_inits != None):
        #    raise ValueError('bias is {} but b_inits is not empty nor None but isntead is {}'.format(bias,b_inits))
        # actiaction func
        self.act = act
        #create linear layers
        self.linear_layers = torch.nn.ModuleList([None])
        #self.linear_layers = torch.nn.ParameterList([None])
        for d in range(1,len(D_layers)):
            linear_layer = torch.nn.Linear(D_layers[d-1], D_layers[d],bias=bias)
            self.linear_layers.append(linear_layer)
        # initialize model
        for d in range(1,len(D_layers)):
            weight_init = w_inits[d]
            m = self.linear_layers[d]
            weight_init(m)
            if bias:
                bias_init = b_inits[d]
                bias_init(m)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        a = x
        for d in range(1,len(self.linear_layers)-1):
            W_d = self.linear_layers[d]
            z = W_d(a)
            a = self.act(z)
        d = len(self.linear_layers)-1
        y_pred = self.linear_layers[d](a)
        return y_pred

    def to_gpu(self,device_id=None):
        torch.nn.Module.cuda(device_id=device_id)

    def get_parameters(self):
        return list(self.parameters())

    def get_nb_params(self):
        return sum(p.numel() for p in model.parameters())
