import time
import numpy as np
import torch

from torch.autograd import Variable

from maps import NamedDict

import data_utils
import utils

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

def SGD_pert_then_train(mdl, Xtr,Ytr,Xv,Yv,Xt,Yt, optimizer,loss, M,nb_iter ,logging_freq ,dtype_x,dtype_y, perturbfreq,perturb_magnitude, iterations_switch_mode, reg,reg_lambda, stats_collector):
    '''
    '''
    classification_task = type(Ytr[0]) == np.int64
    ''' wrap data in torch '''
    Xtr,Xv,Xt = data_utils.data2FloatTensor(Xtr,Xv,Xt)
    Ytr,Yv,Yt = data_utils.data2LongTensor(Ytr,Yv,Yt) if classification_task else data_utils.data2FloatTensor(Ytr,Yv,Yt)
    ## wrap in pytorch Variables
    Xtr,Ytr,Xv,Yv,Xt,Yt = data_utils.data2torch_variable(Xtr,Ytr,Xv,Yv,Xt,Yt)
    ''' Start training '''
    pert_mode = True
    N_train, _ = tuple( Xtr.size() )
    for i in range(0,nb_iter):
        optimizer.zero_grad()
        batch_xs, batch_ys = get_batch2(Xtr,Ytr,M,(dtype_x,dtype_y)) # [M, D], [M, 1]
        ''' FORWARD PASS '''
        y_pred = mdl(batch_xs)
        if vectors_dims_dont_match(batch_ys,y_pred) and not classification_task: ## Check vectors have same dimension
            raise ValueError(f'You vectors don\'t have matching dimensions. It will lead to errors: \n batch_ys={batch_ys.size()},y_pred={y_pred.size()}')
        batch_loss = loss(input=y_pred,target=batch_ys) + reg_lambda*reg
        batch_loss.backward() # Use autograd to compute the backward pass. Now w will have gradients
        """ Update parameters """
        optimizer.step()
        ''' Collect training stats '''
        if i % (nb_iter/10) == 0 or i == 0 and False:
            current_train_loss,train_acc = stats_collector.loss(mdl,Xtr,Ytr),stats_collector.acc(mdl,Xtr,Ytr)
            current_test_loss,test_acc = stats_collector.loss(mdl,Xt,Yt),stats_collector.acc(mdl,Xt,Yt)
            print('\n-------------')
            print(f'i={i}, current_train_loss={current_train_loss}, i={i}, train_error = {train_acc}')
            print(f'i={i}, current_test_loss={current_test_loss}, i={i}, test_error = {test_acc}')
        ## stats logger
        if i % logging_freq == 0 or i == 0:
            stats_collector.collect_stats(i, mdl, Xtr,Ytr,Xv,Yv,Xt,Yt)
        ## DO OP
        if pert_mode:
            if i%perturbfreq == 0 and perturb_magnitude != 0 and i != 0:
                for W in mdl.parameters():
                    Din,Dout = W.data.size()
                    std = perturb_magnitude
                    noise = torch.normal(mean=0.0*torch.ones(Din,Dout),std=std*torch.ones(Din,Dout))
                    W.data.copy_(W.data + noise)
        ''' switch mode? '''
        pert_mode = (i < iterations_switch_mode)

######

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
        self.train_accs, self.val_accs, self.test_accs = [], [], []
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

    def collect_mdl_params_stats(self,mdl):
        ''' log parameter stats'''
        for index, W in enumerate(mdl.parameters()):
            self.w_norms[index].append( W.data.norm(2) )
            self.grads[index].append( W.grad.data.norm(2) )
            if utils.is_NaN(W.grad.data.norm(2)):
                raise ValueError(f'Nan Detected error happened at: i={i} loss_val={loss_val}, loss={loss}')

    def collect_stats(self, i, mdl, Xtr,Ytr,Xv,Yv,Xt,Yt):
        ''' log train losses '''
        self.train_losses.append( self.loss(mdl,Xtr,Ytr).item() )
        self.val_losses.append( self.loss(mdl,Xv,Yv).item() )
        self.test_losses.append( self.loss(mdl,Xt,Yt).item() )
        ''' log train errors '''
        self.train_errors.append( self.acc(mdl,Xtr,Ytr).item() )
        self.val_errors.append( self.acc(mdl,Xv,Yv).item() )
        self.test_errors.append( self.acc(mdl,Xt,Yt).item() )
        ''' log parameter stats'''
        for index, W in enumerate(mdl.parameters()):
            self.w_norms[index].append( W.data.norm(2) )
            self.grads[index].append( W.grad.data.norm(2) )
            if utils.is_NaN(W.grad.data.norm(2)):
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

    def append_losses_errors(self,train_loss, train_error, test_loss, test_error):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_errors.append(train_error)
        self.test_errors.append(test_error)
        self.train_accs.append(1.0-train_error)
        self.test_accs.append(1.0-test_error)

####

def train_cifar(args, nb_epochs, trainloader,testloader, net,optimizer,criterion,logging_freq=2000):
    # TODO: test loss
    for epoch in range(nb_epochs):  # loop over the dataset multiple times
        running_train_loss = 0.0
        #running_test_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            start_time = time.time()
            inputs, labels = data
            if args.enable_cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_train_loss += loss.item()
            seconds,minutes,hours = utils.report_times(start_time)
            st()
            if i % logging_freq == logging_freq-1:    # print every logging_freq mini-batches
                # note you dividing by logging_freq because you summed logging_freq mini-batches, so the average is dividing by logging_freq.
                print(f'monitoring during training: eptoch={epoch+1}, batch_index={i+1}, loss={running_train_loss/logging_freq}')
                running_train_loss = 0.0

def evalaute_mdl_data_set(loss,error,net,dataloader,enable_cuda):
    '''
    Evaluate the error of the model under some loss and error with a specific data set.
    '''
    running_loss,running_error = 0,0
    for i,data in enumerate(dataloader):
        inputs, labels = extract_data(enable_cuda,data,wrap_in_variable=True)
        outputs = net(inputs)
        running_loss += loss(outputs,labels).item()
        running_error += error(outputs,labels)
    return running_loss/(i+1),running_error/(i+1)

def extract_data(enable_cuda,data,wrap_in_variable=False):
    inputs, labels = data
    if enable_cuda:
        inputs, labels = inputs.cuda(), labels.cuda()
    if wrap_in_variable:
        inputs, labels = Variable(inputs), Variable(labels)
    return inputs, labels

def train_and_track_stats(args, nb_epochs, trainloader,testloader, net,optimizer,criterion,error_criterion ,stats_collector):
    enable_cuda = args.enable_cuda
    ##
    for epoch in range(nb_epochs):  # loop over the dataset multiple times
        running_train_loss,running_train_error = 0.0,0.0
        running_test_loss,running_test_error = 0.0,0.0
        for (i,(data_train,data_test)) in enumerate( zip(trainloader,testloader) ):
            ''' zero the parameter gradients '''
            optimizer.zero_grad()
            ''' train step = forward + backward + optimize '''
            inputs, labels = extract_data(enable_cuda,data_train,wrap_in_variable=True)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            running_train_error += error_criterion(outputs,labels)
            ''' test evaluation '''
            inputs, labels = extract_data(enable_cuda,data=data_test,wrap_in_variable=True)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()
            running_test_error += error_criterion(outputs,labels)
            ''' print error first iteration'''
            if i == 0: # print on the first iteration
                print(f'--\ni={i}, running_train_loss={running_train_loss}, running_train_error={running_train_error}, running_test_loss={running_test_loss},running_test_error={running_test_error}')
        ''' End of Epoch: collect stats'''
        train_loss_epoch, train_error_epoch = running_train_loss/(i+1), running_train_error/(i+1)
        test_loss_epoch, test_error_epoch = running_test_loss/(i+1), running_test_error/(i+1)
        stats_collector.collect_mdl_params_stats(net)
        stats_collector.append_losses_errors(train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch)
        print(f'epoch={epoch}, train_loss_epoch={train_loss_epoch}, train_error_epoch={train_error_epoch}, test_loss_epoch={test_loss_epoch},test_error_epoch={test_error_epoch}')
    return train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch
