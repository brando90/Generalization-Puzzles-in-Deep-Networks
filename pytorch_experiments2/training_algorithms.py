import numpy as np
import torch

import data_utils

from pdb import set_trace as st

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
           raise ValueError('You vectors don\'t have matching dimensions. It will lead to errors.')
        batch_loss = loss(input=y_pred,target=batch_ys) + reg_lambda*reg
        batch_loss.backward() # Use autograd to compute the backward pass. Now w will have gradients
        """ Update parameters """
        optimizer.step()
        ## train stats
        if i % (nb_iter/10) == 0 or i == 0 and False:
            current_train_loss = loss(input=mdl(Xtr),target=Ytr).data.numpy()
            train_acc = (max_indices == Ytr).sum().data.numpy()/max_indices.size()[0]
            current_test_loss = loss(input=mdl(Xt),target=Yt).data.numpy()
            max_vals, max_indices = torch.max(mdl(Xt),1)
            test_acc = (max_indices == Yt).sum().data.numpy()/max_indices.size()[0]
            print('-------------')
            print(f'i = {i}, current_train_loss = {current_train_loss}')
            print(f'i = {i}, train_acc = {train_acc}')
            print(f'i = {i}, current_test_loss = {current_test_loss}')
            print(f'i = {i}, test_acc = {test_acc}')
        ## stats logger
        if i % logging_freq == 0 or i == 0:

            # for index, W in enumerate(mdl.parameters()):
            #     w_norms[index].append( W.data.norm(2) )
        ## DO OP
        if i % perturbfreq == 0 and pert_magnitude != 0 and i != 0:
            for W in mdl.parameters():
                #pdb.set_trace()
                Din,Dout = W.data.size()
                std = pert_magnitude
                noise = torch.normal(means=0.0*torch.ones(Din,Dout),std=std)
                W.data.copy_(W.data + noise)
    return

def calc_loss(mdl,loss,X,Y):
    loss_val = loss(input=mdl(X),target=Y).data.numpy()
    if is_NaN(current_train_loss):
        raise ValueError(f'Nan Detected error happened at: i={i} loss_val={loss_val}, loss={loss}')
    return loss_val

def calc_accuracy(mdl,X,Y):
    # TODO: why can't we call .data.numpy() for train_acc as a whole?
    max_vals, max_indices = torch.max(mdl(X),1)
    train_acc = (max_indices == Y).sum().data.numpy()/max_indices.size()[0]
    if is_NaN(current_train_loss):
        loss = 'accuracy'
        raise ValueError(f'Nan Detected error happened at: i={i} loss_val={loss_val}, loss={loss}')
    return train_acc

class StatsCollector:
    def __init__(self, loss,accuracy):
        ''' functions that encode reward/loss '''
        self.loss = loss
        self.acc = accuracy
        ''' loss & errors lists'''
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.train_errors, self.val_errors, self.test_errors = [], [], []
        ''' stats related to parameters'''
        nb_param_groups = len( list(mdl.parameters()) )
        self.grads = [ [] for i in range(nb_param_groups) ]
        self.w_norms = [ [] for i in range(nb_module_params) ]

    def collect_stats(self, i, mdl, Xtr,Ytr,Xv,Yv,Xt,Yt):
        ''' log train losses '''
        self.train_losses.append( float(self.loss(mdl,Xtr,Ytr)) )
        self.val_losses.append( float(self.loss(mdl,Xv,Yv)) )
        self.test_losses.append( float(self.loss(md,Xt,Yt)) )
        ''' log train errors '''
        self.train_errors.append( float(self.acc(mdl,Xtr,Ytr)) )
        self.val_losses.append( float(self.acc(mdl,Xv,Yv)) )
        self.test_losses.append( float(self.add(mdl,Xt,Yt)) )
        ''' log parameter stats'''
        for index, W in enumerate(mdl.parameters()):
            delta = eta*W.grad.data
            grad_list[index].append( W.grad.data.norm(2) )
            if is_NaN(W.grad.data.norm(2)):
                raise ValueError(f'Nan Detected error happened at: i={i} loss_val={loss_val}, loss={loss}')

    def collect_stats(self,loss,accuracy, mdl, Xtr,Ytr,Xv,Yv,Xt,Yt):
        N_train,_ = tuple(Xtr.size())
        N_test,_ = tuple(Xt.size())
        ## log: TRAIN ERROR
        y_pred_train = mdl(Xtr)
        current_train_loss = loss(input=y_pred_train,target=Ytr).data.numpy()
        loss_list.append( float(current_train_loss) )
        ##
        y_pred_test = mdl(Xt)
        current_test_loss = loss(input=y_pred_test,target=Yt).data.numpy()
        test_loss_list.append( float(current_test_loss) )
        ## log: GEN DIFF/FUNC DIFF
        gen_diff = -1
        func_diff.append( float(gen_diff) )
        ## ERM + regularization
        #erm_reg = get_ERM_lambda(arg,mdl,reg_lambda,X=Xtr,Y=Ytr,l=2).data.numpy()
        reg = get_regularizer_term(arg, mdl,reg_lambda,X=Xtr,Y=Ytr,l=2)
        erm_reg = loss(input=y_pred_test,target=Yt) + reg_lambda*reg
        erm_lamdas.append( float(erm_reg.data.numpy()) )
        ##
        max_vals, max_indices = torch.max(mdl(Xtr),1)
        train_acc = (max_indices == Ytr).sum().data.numpy()/max_indices.size()[0]
        train_accs.append(train_acc)
        ##
        max_vals, max_indices = torch.max(mdl(Xt),1)
        test_acc = (max_indices == Yt).sum().data.numpy()/max_indices.size()[0]
        test_accs.append(test_acc)
        ##
        for index, W in enumerate(mdl.parameters()):
            delta = eta*W.grad.data
            grad_list[index].append( W.grad.data.norm(2) )
            if is_NaN(W.grad.data.norm(2)) or is_NaN(current_train_loss):
                print('\n----------------- ERROR HAPPENED \a')
                print('reg_lambda', reg_lambda)
                print('error happened at: i = {} current_train_loss: {}, grad_norm: {},\n ----------------- \a'.format(i,current_train_loss,W.grad.data.norm(2)))
                #sys.exit()
                #pdb.set_trace()
                raise ValueError('Nan Detected')
