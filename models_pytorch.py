import torch
from torch.autograd import Variable
from torch.

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in,H_l1,H_l2,D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class NN(torch.nn.Module):
    def __init__(self, D_layers,act,w_inits,b_inits):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.

        D_layers = [D^(0),D^(1),...,D^(L)]
        w_inits = [None,W_f1,...,W_fL]
        b_inits = [None,b_f1,...,b_fL]
        """
        super(TwoLayerNet, self).__init__()
        # actiaction func
        self.act = act
        #create linear layers
        self.linear_layers = [None]
        for d in range(1,len(D_layers)):
            linear_layer = torch.nn.Linear(D_layers[d-1], D_layers[d])
            self.linear_layers.append(linear_layer)
        # initialize model
        for d in range(1,len(D_layers)):
            weight_init, bias_init = w_inits[d], b_inits[d]
            m = self.linear_layers[d]
            weight_init(m)
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
        y_pred = self.linear_layers[d](a)
        return y_pred
