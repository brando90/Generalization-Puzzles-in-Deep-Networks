import torch

from math import inf

import copy

import numpy as np

from stats_collector import StatsCollector

from new_training_algorithms import dont_train
from new_training_algorithms import get_function_evaluation_from_name

from pdb import set_trace as st

class LandscapeInspector:

    def __init__(self,net_original,net_pert, nb_epochs,iterations, trainloader,testloader, optimizer,criterion,error_criterion, device, lambdas,save_all_learning_curves=False,save_all_perts=False,evalaute_mdl_data_set='evalaute_mdl_on_full_data_set'):
        '''
        :return:
        '''
        self.device = device
        ''' data sets '''
        self.trainloader = trainloader
        self.testloader = testloader
        ''' optimzier & criterions '''
        self.nb_epochs = nb_epochs
        self.iterations = iterations
        self.optimizer = optimizer
        self.criterion = criterion
        self.error_criterion = error_criterion
        ''' three musketeers '''
        #self.net_original = dont_train(net_original)
        self.net_original = net_original
        self.net_place_holder = copy.deepcopy(net_pert)
        self.net_pert = net_pert
        ''' lanscape stats '''
        self.lambdas = lambdas
        self.eps = []
        self.pert_norms = []
        self.save_all_perts = save_all_perts
        if self.save_all_perts:
            #self.perts = [] # TODO: saving lots of models
            print('Not implemented')
        ''' save all learning stats for each perturbation '''
        self.save_all_learning_curves = save_all_learning_curves
        if self.save_all_learning_curves:
            # map [lambda/pert] -> stats collector
            self.stats_collector = StatsCollector(self.net_pert,trials=len(lambdas),epochs=self.nb_epochs+1)
        ''' '''
        evalaute_mdl_data_set = get_function_evaluation_from_name(evalaute_mdl_data_set)
        if evalaute_mdl_data_set is None:
            raise ValueError(f'Data set function evaluator evalaute_mdl_data_set={evalaute_mdl_data_set} is not defined.')
        else:
            self.evalaute_mdl_data_set = evalaute_mdl_data_set

    def do_sharpness_experiment(self):
        '''
        Go through the lambda's

        :return:
        '''
        for lambda_index,lambda_eps in enumerate(self.lambdas):
            ''' get sharpness for current lambda '''
            loss_lambda, eps_lambda = self._record_sharpness(lambda_index)
            self.eps.append(eps_lambda)
            ''' record perturbation size '''
            pert_norm = self.get_pert_norm()
            self.pert_norms.append(pert_norm)

    def _record_sharpness(self,lambda_index):
        lambda_e = float(self.lambdas[lambda_index])
        ''' Add stats before training '''
        train_loss_epoch, train_error_epoch = self.evalaute_mdl_data_set(self.criterion,self.error_criterion,self.net_pert,self.trainloader,self.device,self.iterations)
        self.save_current_stats(0,lambda_index, train_loss_epoch,train_error_epoch)
        print(f'[{-1}, {-1}], (lambda={self.lambdas[lambda_index]}),(train_loss: {train_loss_epoch}, train error: {train_error_epoch}, adverserial objective: {None}, pert_W_norm(l=2): {self.get_pert_norm(l=2)})')
        st()
        ''' Start training '''
        for epoch in range(self.nb_epochs):  # loop over the dataset multiple times
            self.net_pert.train()
            self.net_place_holder.train()
            ''' do Epoch '''
            running_train_loss,running_train_error = 0.0, 0.0
            for i,(inputs,targets) in enumerate(self.trainloader):
                inputs,targets = inputs.to(self.device),targets.to(self.device)
                ''' zero the parameter gradients '''
                self.optimizer.zero_grad()
                ''' net_place_holder = net_original + net_pert , net(W+p) '''
                self.net_place_holder = self.combine_nets(net_train=self.net_pert,net_no_train=self.net_original,net_place_holder=self.net_place_holder)
                #self.net_place_holder = self.net_pert
                #self.net_place_holder = self.net_original
                ''' sum_i Loss(net(W+p),l_i)'''
                #outputs = self.net_pert(inputs)
                outputs = self.net_place_holder(inputs)
                loss = self.criterion(outputs,targets)
                #print(f'self.net_place_holder(inputs): {outputs}')
                #print( f'self.net_pert(inputs): {self.net_pert(inputs)}' )
                #st()
                loss_val = loss.item()
                #loss = self.get_pert_norm(l=2) + lambda_e*loss
                #loss = self.get_pert_norm(l=2) - lambda_e * loss
                #loss = self.get_pert_norm(l=2)
                #loss = - lambda_e*loss
                #loss = lambda_e*loss
                ''' train/update pert '''
                loss.backward()
                self.optimizer.step()
                ''' stats '''
                running_train_loss += loss.item()
                running_train_error += self.error_criterion(outputs,targets)
                ''' print error first iteration'''
                #if i == 0 and epoch == 0: # print on the first iteration
                #    print(data_train[0].data)
            ''' End of Epoch: collect stats'''
            self.net_pert.eval()
            train_loss_epoch, train_error_epoch = running_train_loss/(i+1), running_train_error/(i+1)
            self.save_current_stats(epoch,lambda_index, train_loss_epoch,train_error_epoch)
            print(f'[{epoch}, {i+1}], (lambda={self.lambdas[lambda_index]}),(train_loss: {loss_val}, train error: {train_error_epoch}, adverserial objective: {running_train_loss}, pert_W_norm(l=2): {self.get_pert_norm(l=2)})')
        return train_loss_epoch, train_error_epoch

    def get_pert_norm(self,l=2):
        '''
        get l norm of the perturabation
        '''
        w_norms = 0
        for index, W in enumerate(self.net_pert.parameters()):
            w_norms += W.norm(l)
        return w_norms

    def save_current_stats(self,epoch,lambda_index, train_loss_epoch,train_error_epoch):
        '''
        saves the current stats of the error of the adverserial perturbation on the weights.
        '''
        if self.save_all_learning_curves:
            errors_losses=(train_loss_epoch,train_error_epoch,-1,-1) # last two are test errors, they aren't aplicable right now
            self.stats_collector.append_all_losses_errors_accs(lambda_index,epoch,errors_losses)

    @staticmethod
    def combine_nets(net_train, net_no_train, net_place_holder):
        '''
            Combine nets in a way train net is trainable
        '''
        params_train = net_train.named_parameters()
        dict_params_place_holder = dict(net_place_holder.named_parameters())
        dict_params_no_train = dict(net_no_train.named_parameters())
        for name, param_train in params_train:
            if name in dict_params_place_holder:
                layer_name, param_name = name.split('.')
                param_no_train = dict_params_no_train[name]
                ## get place holder layer
                layer_place_holder = getattr(net_place_holder, layer_name)
                #print(f'layer_place_holder = {getattr(layer_place_holder,param_name)}')
                #print(f'param_no_train = {param_no_train}')
                #print(f'getattr(layer_place_holder,param_name) = {id(getattr(layer_place_holder,param_name))}')
                delattr(layer_place_holder, param_name)
                #print(f'param_no_train = {param_no_train}')
                #print(f'param_no_train = {id(param_no_train)}')
                #st()
                ## get new param
                W_new = param_train + param_no_train  # notice addition is just chosen for the sake of an example
                ## store param in placehoder net
                setattr(layer_place_holder, param_name, W_new)
        return net_place_holder

    def add_PertNet_OriginalNet(self,w1=1,w2=1):
        '''
            Convex interpolation of two nets w1*W_l + w2*W_l.
        '''
        ''' '''
        params_pert = self.net_pert.named_parameters()
        dict_params_place_holder = dict( self.net_place_holder.named_parameters() )
        dict_params_original = dict(self.net_original.named_parameters())
        for name,param_pert in params_pert:
            if name in dict_params_place_holder:
                print(name)
                param_original = dict_params_original[name]
                delattr(self.net_place_holder, name)
                W_new = w1 * param_pert + w2 * param_original
                print(f'W_new.requires_grad = {W_new.requires_grad}')
                print(f'id(W_new)={id(W_new)}\n')
                setattr(self.net_place_holder, name, W_new)
                print(f'id(self.net_place_holder.conv0.weight)={id(self.net_place_holder.conv0.weight)}')
                print(f'self.net_place_holder.conv0.weight.requires_grad = {self.net_place_holder.conv0.weight.requires_grad}\n')
                #st()
        return self.net_place_holder

if __name__ == '__main__':
    print('main')