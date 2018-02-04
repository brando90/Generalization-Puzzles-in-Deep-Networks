import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize

import pdb

class OneLayerHBF(torch.nn.Module):
    def __init__(self,D_in,D_out, centers,std, train_centers,train_std, bias=False):
        """
        """
        super(OneLayerHBF, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out, bias=bias)
        self.std = Variable(centers,requires_grad=train_std)
        self.t = Variable(centers,requires_grad=train_centers)
        self.linear_layers = torch.nn.ModuleList([self.t,self.std])

    def forward(self, x):
        """
        """
        beta = (1.0/self.std)**2
        y_pred = torch.exp(-beta*euclidean_distances_pytorch(x=x,W=self.t)) # -beta*|| x - t ||^2
        return y_pred

def euclidean_distances_pytorch(x,W):
    '''
    x = input, M data points of size D^(l-1), MxD^(l-1)
    W = params/centers, D^(l-1) x D^(l) means that each center is of dimension D^(l-1) since
        they are being computed with respect to an element from the previous layer.
        Thus, they are D^(l) of them.
    return:
    Delta_tilde = (M x D^(l))
    '''
    WW = torch.sum(np.multiply(W,W), axis=0, keepdims=True) #(1 x D^(l))= sum( (D^(l-1) x D^(l)), 0 )
    XX = torch.sum(np.multiply(x,x), axis=1, keepdims=True) #(M x 1) = sum( (M x D^(l-1)), 1 )
    # || x - w ||^2 = (||x||^2 + ||w||^2) - 2<x,w>
    #Delta_tilde = 2.0*np.dot(x,W) - (WW + XX)
    xW = x.mm(W) #(M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l))
    Delta_tilde = (WW + XX) - 2.0*xW #(M x D^(l)) = (M x D^(l)) + ( (M x 1) + (1 x D^(l)) )
    return Delta_tilde

def get_rbf_coefficients(X,centers,Y,std):
    '''
    X = input, Dx1
    center = centers of RBF, Dx1
    std = standard dev of Gaussians, 1x1
    We want to solve ||Af(a) - y||^2 s.t. f(a) is smooth. Thus use RBF kernel
    with appropriate standard deviation.
    With that we solve:
    ||Kc - y||^2 where K is the kernel matrix K=exp(-beta|a-t|^2) where t are
    centers of the RBFs.
    To solve it do:
        c=(AK)^+y
    '''
    beta = np.power(1.0/std,2)
    Kern = get_kernel_matrix(X,centers.transpose(),beta)
    #(C,_,_,_) = np.linalg.lstsq( np.(A,Kern),Y)
    C = np.dot( np.linalg.pinv(Kern),Y)
    return C

## numpy code

def euclidean_distances_manual(x,W):
    '''
    x = input, M data points of size D^(l-1), MxD^(l-1)
    W = params/centers, D^(l-1) x D^(l) means that each center is of dimension D^(l-1) since
        they are being computed with respect to an element from the previous layer.
        Thus, they are D^(l) of them.
    return:
    Delta_tilde = (M x D^(l))
    '''
    WW = np.sum(np.multiply(W,W), axis=0, dtype=None, keepdims=True) #(1 x D^(l))= sum( (D^(l-1) x D^(l)), 0 )
    XX = np.sum(np.multiply(x,x), axis=1, dtype=None, keepdims=True) #(M x 1) = sum( (M x D^(l-1)), 1 )
    # || x - w ||^2 = (||x||^2 + ||w||^2) - 2<x,w>
    #Delta_tilde = 2.0*np.dot(x,W) - (WW + XX)
    xW = np.dot(x,W) #(M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l))
    Delta_tilde = (WW + XX) - 2.0*xW #(M x D^(l)) = (M x D^(l)) + ( (M x 1) + (1 x D^(l)) )
    return Delta_tilde
