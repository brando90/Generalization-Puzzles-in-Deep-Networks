import time
import numpy as np
import torch

from torch.autograd import Variable

import data_utils
import utils

from math import inf
import os
from maps import NamedDict

from pdb import set_trace as st

def dont_train(net):
    '''
    set training parameters to false.

    :param net:
    :return:
    '''
    for param in net.parameters():
        param.requires_grad = False
    return net

def initialize_to_zero(net):
    '''
    sets weights of net to zero.
    '''
    for param in net.parameters():
        #st()
        param.zero_()

def evalaute_mdl_data_set(loss,error,net,dataloader,device,iterations=inf):
    '''
    Evaluate the error of the model under some loss and error with a specific data set.
    '''
    running_loss,running_error = 0,0
    with torch.no_grad():
        #st()
        #for i,(samples) in enumerate(dataloader):
        for i,(inputs,targets) in enumerate(dataloader):
            if i >= iterations:
                break
            inputs,targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            running_loss += loss(outputs,targets).item()
            running_error += error(outputs,targets).item()
    return running_loss/(i+1),running_error/(i+1)

class Trainer:

    def __init__(self,trainloader,testloader, optimizer,criterion,error_criterion, stats_collector, device, expt_path='',net_file_name='',save_every_epoch=False):
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.error_criterion = error_criterion
        self.stats_collector = stats_collector
        self.device = device
        ''' save all models during training '''
        self.save_every_epoch = save_every_epoch
        self.expt_path = expt_path
        self.net_file_name = net_file_name
        if self.expt_path != '' and self.net_file_name != '' and self.save_every_epoch:
            utils.make_and_check_dir(expt_path)

    def train_and_track_stats(self,net, nb_epochs,iterations=inf,target_train_loss=inf,precision=0.10**-7):
        '''
        train net with nb_epochs and 1 epoch only # iterations = iterations
        '''
        ''' Add stats before training '''
        train_loss_epoch, train_error_epoch = evalaute_mdl_data_set(self.criterion, self.error_criterion, net, self.trainloader, self.device, iterations)
        test_loss_epoch, test_error_epoch = evalaute_mdl_data_set(self.criterion, self.error_criterion, net, self.testloader, self.device, iterations)
        self.stats_collector.collect_mdl_params_stats(net)
        self.stats_collector.append_losses_errors_accs(train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch)
        print(f'[-1, -1], (train_loss: {train_loss_epoch}, train error: {train_error_epoch}) , (test loss: {test_loss_epoch}, test error: {test_error_epoch})')
        ''' perhaps save net @ epoch '''
        self.perhaps_save(net,epoch=0)
        ''' Start training '''
        print('about to start training')
        for epoch in range(nb_epochs):  # loop over the dataset multiple times
            net.train()
            running_train_loss,running_train_error = 0.0, 0.0
            for i,(inputs,targets) in enumerate(self.trainloader):
                ''' zero the parameter gradients '''
                self.optimizer.zero_grad()
                ''' train step = forward + backward + optimize '''
                inputs,targets = inputs.to(self.device),targets.to(self.device)
                outputs = net(inputs)
                loss = self.criterion(outputs,targets)
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item()
                running_train_error += self.error_criterion(outputs,targets)
                ''' print error first iteration'''
                #if i == 0 and epoch == 0: # print on the first iteration
                #    print(data_train[0].data)
            ''' End of Epoch: collect stats'''
            train_loss_epoch, train_error_epoch = running_train_loss/(i+1), running_train_error/(i+1)
            net.eval()
            test_loss_epoch, test_error_epoch = evalaute_mdl_data_set(self.criterion,self.error_criterion,net,self.testloader,self.device,iterations)
            self.stats_collector.collect_mdl_params_stats(net)
            self.stats_collector.append_losses_errors_accs(train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch)
            print(f'[{epoch}, {i+1}], (train_loss: {train_loss_epoch}, train error: {train_error_epoch}) , (test loss: {test_loss_epoch}, test error: {test_error_epoch})')
            ''' perhaps save net @ epoch '''
            self.perhaps_save(net,epoch=epoch)
            ''' check target loss '''
            if abs(train_loss_epoch - target_train_loss) < precision:
                return train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch
        return train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch

    def perhaps_save(self,net,epoch):
        ''' save net model '''
        if self.save_every_epoch:
            epoch_net_file_name = f'{self.net_file_name}_epoch_{epoch}'
            net_path_to_filename = os.path.join(self.expt_path,epoch_net_file_name)
            torch.save(net, net_path_to_filename)