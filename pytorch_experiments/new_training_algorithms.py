import time
import numpy as np
import torch

from torch.autograd import Variable

from maps import NamedDict

import data_utils
import utils

from math import inf

from pdb import set_trace as st

def evalaute_mdl_data_set(loss,error,net,dataloader,enable_cuda,iterations=inf):
    '''
    Evaluate the error of the model under some loss and error with a specific data set.
    '''
    running_loss,running_error = 0,0
    for i,data in enumerate(dataloader):
        if i >= iterations:
            break
        inputs, labels = extract_data(enable_cuda,data,wrap_in_variable=True)
        outputs = net(inputs)
        running_loss += loss(outputs,labels).data[0]
        running_error += error(outputs,labels)
    return running_loss/(i+1),running_error/(i+1)

def extract_data(enable_cuda,data,wrap_in_variable=False):
    inputs, labels = data
    if enable_cuda:
        inputs, labels = inputs.cuda(), labels.cuda() #TODO potential speed up?
    if wrap_in_variable:
        inputs, labels = Variable(inputs), Variable(labels)
    return inputs, labels

def train_and_track_stats(args, nb_epochs, trainloader,testloader, net,optimizer,criterion,error_criterion ,stats_collector,iterations=inf):
    enable_cuda = args.enable_cuda
    ''' Add stats before training '''
    train_loss_epoch, train_error_epoch = evalaute_mdl_data_set(criterion, error_criterion, net, trainloader, enable_cuda, iterations)
    test_loss_epoch, test_error_epoch = evalaute_mdl_data_set(criterion, error_criterion, net, testloader, enable_cuda, iterations)
    stats_collector.collect_mdl_params_stats(net)
    stats_collector.append_losses_errors_accs(train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch)
    print( f'[-1, -1], (train_loss: {train_loss_epoch}, train error: {train_error_epoch}) , (test loss: {test_loss_epoch}, test error: {test_error_epoch})')
    ##
    ''' Start training '''
    print('about to start training')
    for epoch in range(nb_epochs):  # loop over the dataset multiple times
        running_train_loss,running_train_error = 0.0, 0.0
        for i,data_train in enumerate(trainloader):
            ''' zero the parameter gradients '''
            optimizer.zero_grad()
            ''' train step = forward + backward + optimize '''
            inputs, labels = extract_data(enable_cuda,data_train,wrap_in_variable=True)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.data[0]
            running_train_error += error_criterion(outputs,labels)
            ''' print error first iteration'''
            #if i == 0 and epoch == 0: # print on the first iteration
            #    print(data_train[0].data)
        ''' End of Epoch: collect stats'''
        train_loss_epoch, train_error_epoch = running_train_loss/(i+1), running_train_error/(i+1)
        test_loss_epoch, test_error_epoch = evalaute_mdl_data_set(criterion,error_criterion,net,testloader,enable_cuda,iterations)
        stats_collector.collect_mdl_params_stats(net)
        stats_collector.append_losses_errors_accs(train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch)
        print(f'[{epoch}, {i+1}], (train_loss: {train_loss_epoch}, train error: {train_error_epoch}) , (test loss: {test_loss_epoch}, test error: {test_error_epoch})')
    return train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch
