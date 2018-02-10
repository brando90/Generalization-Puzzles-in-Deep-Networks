import numpy as np
import torch

from maps import NamedDict

import data_utils

from pdb import set_trace as st

def vectors_dims_dont_match(Y,Y_):
    '''
    Checks that vector Y and Y_ have the same dimensions. If they don't
    then there might be an error that could be caused due to wrong broadcasting.
    '''
    DY = tuple( Y.size() )
    DY_ = tuple( Y_.size() )
    if len(DY) != len(DY_):
        return True
    for i in range(len(DY)):
        if DY[i] != DY_[i]:
            return True
    return False

def is_NaN(value):
    '''
    Checks is value is problematic by checking if the value:
    is not finite, is infinite or is already NaN
    '''
    return not np.isfinite(value) or np.isinf(value) or np.isnan(value)

def index_batch(X,batch_indices,dtype):
    '''
    returns the batch indexed/sliced batch
    '''
    # if len(X.size()) == 1: # i.e. dimension (M,) just a vector
    #     batch_xs = X[batch_indices].type(dtype)
    # else:
    #     batch_xs = X[batch_indices,:].type(dtype)
    #batch_xs = torch.index_select(input=X,dim=0,index=batch_indices) #torch.index_select(input, dim, index, out=None)
    batch_xs = torch.index_select(X,0,batch_indices)
    return batch_xs

def get_batch2(X,Y,M,dtype):
    '''
    get batch for pytorch model
    '''
    # TODO fix and make it nicer, there is pytorch forum question
    dtype_x,dtype_y = (dtype[0],dtype[1]) if len(dtype) == 2 else (dtype,dtype)
    #X,Y = X.data.numpy(), Y.data.numpy()
    N = len(Y)
    valid_indices = np.array( range(N) )
    batch_indices = torch.LongTensor(np.random.choice(valid_indices,size=M,replace=False))
    batch_xs = index_batch(X,batch_indices,dtype_x)
    batch_ys = index_batch(Y,batch_indices,dtype_y)
    return batch_xs, batch_ys

def SGD_perturb(mdl, Xtr,Ytr,Xv,Yv,Xt,Yt, optimizer,loss, M,eta,nb_iter,A ,logging_freq ,dtype_x,dtype_y, perturbfreq,perturb_magnitude, reg,reg_lambda, stats_collector):
    '''
    '''
    classification_task = type(Ytr[0]) == np.int64
    ''' wrap data in torch '''
    Xtr,Xv,Xt = data_utils.data2FloatTensor(Xtr,Xv,Xt)
    Ytr,Yv,Yt = data_utils.data2LongTensor(Ytr,Yv,Yt) if classification_task else data_utils.data2FloatTensor(Ytr,Yv,Yt)
    ## wrap in pytorch Variables
    Xtr,Ytr,Xv,Yv,Xt,Yt = data_utils.data2torch_variable(Xtr,Ytr,Xv,Yv,Xt,Yt)
    ''' Start training '''
    N_train, _ = tuple( Xtr.size() )
    for i in range(0,nb_iter):
        optimizer.zero_grad()
        batch_xs, batch_ys = get_batch2(Xtr,Ytr,M,(dtype_x,dtype_y)) # [M, D], [M, 1]
        ''' FORWARD PASS '''
        y_pred = mdl(batch_xs)
        if vectors_dims_dont_match(batch_ys,y_pred) and not classification_task: ## Check vectors have same dimension
            raise ValueError('You vectors don\'t have matching dimensions. It will lead to errors: \n batch_ys={batch_ys.size()},y_pred={y_pred.size()}')
        batch_loss = loss(input=y_pred,target=batch_ys) + reg_lambda*reg
        batch_loss.backward() # Use autograd to compute the backward pass. Now w will have gradients
        """ Update parameters """
        optimizer.step()
        ''' Collect training stats '''
        if i % (nb_iter/10) == 0 or i == 0 and False:
            current_train_loss,train_acc = stats_collector.loss(mdl,Xtr,Ytr),stats_collector.acc(mdl,Xtr,Ytr)
            current_test_loss,test_acc = stats_collector.loss(mdl,Xt,Yt),stats_collector.acc(mdl,Xt,Yt)
            print('\n-------------')
            print(f'i={i}, current_train_loss={current_train_loss} \ni={i}, train_error = {train_acc}')
            print(f'i={i}, current_test_loss={current_test_loss}, \ni={i}, test_error = {test_acc}')
        ## stats logger
        if i % logging_freq == 0 or i == 0:
            stats_collector.collect_stats(i, mdl, Xtr,Ytr,Xv,Yv,Xt,Yt)
        ## DO OP
        if i%perturbfreq == 0 and perturb_magnitude != 0 and i != 0:
            for W in mdl.parameters():
                Din,Dout = W.data.size()
                std = perturb_magnitude
                noise = torch.normal(means=0.0*torch.ones(Din,Dout),std=std)
                W.data.copy_(W.data + noise)

def calc_loss(mdl,loss,X,Y):
    loss_val = loss(input=mdl(X),target=Y).data.numpy()
    if is_NaN(loss_val):
        raise ValueError(f'Nan Detected error happened at: loss_val={loss_val}, loss={loss}')
    return loss_val

def calc_accuracy(mdl,X,Y):
    # TODO: why can't we call .data.numpy() for train_acc as a whole?
    max_vals, max_indices = torch.max(mdl(X),1)
    train_acc = (max_indices == Y).sum().data.numpy()/max_indices.size()[0]
    if is_NaN(train_acc):
        loss = 'accuracy'
        raise ValueError(f'Nan Detected error happened at: i={i} loss_val={loss_val}, loss={loss}')
    return train_acc

def calc_error(mdl,X,Y):
    # TODO: why can't we call .data.numpy() for train_acc as a whole?
    max_vals, max_indices = torch.max(mdl(X),1)
    train_acc = 1 - (max_indices == Y).sum().data.numpy()/max_indices.size()[0]
    if is_NaN(train_acc):
        loss = 'accuracy'
        raise ValueError(f'Nan Detected error happened at: i={i} loss_val={loss_val}, loss={loss}')
    return train_acc

class StatsCollector:
    '''
    Class that has all the stats collected during training.
    '''
    def __init__(self, mdl,loss,accuracy, dynamic_stats=None):
        '''
            dynamic_stats = an array of tuples (STORER,UPDATER) where the storer
            is a data structure (like a list) that gets updated according to updater.
            For the moment updater receives storer and all the parameters from collect_stats
            (like the mdl, the data sets, the iteration number)
        '''
        ''' functions that encode reward/loss '''
        self.loss = loss
        self.acc = accuracy
        ''' loss & errors lists'''
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.train_errors, self.val_errors, self.test_errors = [], [], []
        ''' stats related to parameters'''
        nb_param_groups = len( list(mdl.parameters()) )
        self.grads = [ [] for i in range(nb_param_groups) ]
        self.w_norms = [ [] for i in range(nb_param_groups) ]
        ''' '''
        if dynamic_stats is not None:
            self.dynamic_stats_storer = {}
            self.dynamic_stats_updater = {}
            for name,(storer,updater) in dynamic_stats.items():
                self.dynamic_stats_storer[name] = storer
                self.dynamic_stats_updater[name] = updater
        else:
            # TODO empty dict or None?
            self.dynamic_stats_storer = None
            self.dynamic_stats_updater = None

    def collect_stats(self, i, mdl, Xtr,Ytr,Xv,Yv,Xt,Yt):
        ''' log train losses '''
        self.train_losses.append( float(self.loss(mdl,Xtr,Ytr)) )
        self.val_losses.append( float(self.loss(mdl,Xv,Yv)) )
        self.test_losses.append( float(self.loss(mdl,Xt,Yt)) )
        ''' log train errors '''
        self.train_errors.append( float(self.acc(mdl,Xtr,Ytr)) )
        self.val_errors.append( float(self.acc(mdl,Xv,Yv)) )
        self.test_errors.append( float(self.acc(mdl,Xt,Yt)) )
        ''' log parameter stats'''
        for index, W in enumerate(mdl.parameters()):
            self.w_norms[index].append( float(W.norm(2).data.numpy()) )
            self.grads[index].append( W.grad.data.norm(2) )
            if is_NaN(W.grad.data.norm(2)):
                raise ValueError(f'Nan Detected error happened at: i={i} loss_val={loss_val}, loss={loss}')
        ''' Update the  '''
        if self.dynamic_stats_storer is not None:
            for name in self.dynamic_stats_updater:
                storer = self.dynamic_stats_storer[name]
                updater = self.dynamic_stats_updater[name]
                updater(storer,i, mdl, Xtr,Ytr,Xv,Yv,Xt,Yt)

    def get_stats_dict(self):
        stats = NamedDict(
            train_losses=self.train_losses,val_losses=self.val_losses,test_losses=self.test_losses,
            train_errors=self.train_errors,val_errors=self.val_errors,test_errors=self.test_errors,
            grads=self.grads,
            w_norms=self.w_norms
        )
        if self.dynamic_stats_storer is not None:
            stats = NamedDict(stats,**self.dynamic_stats_storer)
        return stats
