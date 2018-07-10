import torch
from torch.autograd import Variable
import copy

import sys

import numpy as np
from math import inf

from new_training_algorithms import evalaute_running_mdl_data_set
import nn_models as nn_mdls

from pdb import set_trace as st

def get_errors_for_all_perturbations(net,perturbation_magnitudes,device,nb_perturbation_trials,stats_collector,criterion,error_criterion,trainloader,testloader):
    '''
        Evaluate the errors of perturbed models.
    '''
    std_dict_params = get_std_of_net(net)
    for index_perturbation_trial in range(nb_perturbation_trials):
        train_loss, train_error, test_loss, test_error = perturb_model(net,perturbation_magnitudes,std_dict_params,device,stats_collector,criterion,error_criterion,trainloader,testloader)
    return train_loss, train_error, test_loss, test_error #note this is just some random trial

def perturb_model(net,perturbation_magnitudes,std_dict_params,relative,device,stats_collector,criterion,error_criterion,trainloader,testloader):
    perturbations = []
    ''' pertub model '''
    params = net.named_parameters()
    index = 0
    for name, W in params:
        ''' perturb according to the current perturbation, skip the perturbation if the magnitude of pert  is zero '''
        ## get perturbation
        if perturbation_magnitudes[index] != 0:
            reference_magnitude = std_dict_params[name] if relative else 1
            std = perturbation_magnitudes[index]*reference_magnitude
            perturbation = torch.normal(means=0.0*torch.ones(W.size()),std=std)
        else:
            perturbation = torch.zeros(W.size())
        ''' perturb model '''
        perturbation = perturbation.to(device)
        W.data.copy_(W.data + perturbation)
        ##
        perturbations.append(perturbation.norm(2))
        i+=1
    ''' evalaute model '''
    train_loss, train_error = evalaute_approx_mdl_data_set(criterion,error_criterion,net,trainloader,device)
    test_loss, test_error = evalaute_approx_mdl_data_set(criterion,error_criterion,net,testloader,device)
    ''' record result '''
    stats_collector.append_losses_errors_accs(train_loss, train_error, test_loss, test_error)
    stats_collector.collect_mdl_params_stats(net)
    stats_collector.add_perturbation_norms_from_perturbations(perturbations)
    ''' undo the perturbation '''
    for index, W in enumerate(net.parameters()):
        #with torch.no_grad():
        W.data.copy_(W.data - perturbation[index])
    return train_loss, train_error, test_loss, test_error #note this is just some random trial

def get_std_of_net(net):
    params = net.named_parameters()
    std_dict_params = dict(params)
    for name, param in params:
        dict_params[name] = param.std()
    return std_dict_params
####

def get_landscapes_stats_between_nets(net1,net2, interpolations, device,stats_collector,criterion,error_criterion,trainloader,testloader,iterations):
    '''
        Records the errors for the path by convexly averaging two nets. The goal
        is to be able to estimate the size of the wall between the two different minimums.
    '''
    ''' '''
    interpolated_net = copy.deepcopy(net1)
    diff_W1_W2 = weight_diff_btw_nets(net1,net2)
    for i,alpha in enumerate(interpolations):
        print(f'i={i}, alpha={alpha}')
        ''' interpolate nets with current alpha '''
        interpolated_net = convex_interpolate_nets(interpolated_net,net1,net2,alpha)
        ''' evalaute model '''
        train_loss, train_error = evalaute_running_mdl_data_set(criterion,error_criterion,interpolated_net,trainloader,device,iterations)
        test_loss, test_error = evalaute_running_mdl_data_set(criterion,error_criterion,interpolated_net,testloader,device,iterations)
        ''' record result '''
        stats_collector.append_losses_errors_accs(train_loss, train_error, test_loss, test_error)
        stats_collector.collect_mdl_params_stats(interpolated_net)
        ''' record distance '''
        r = alpha*diff_W1_W2
        stats_collector.rs.append(r)
    return train_loss, train_error, test_loss, test_error, interpolations #note this is just some random trial

def convex_interpolate_nets(interpolated_net,net1,net2,alpha):
    '''
        Convex interpolation of two nets alpha*W_l + (1-alpha)*W_l.
    '''
    params1 = net1.named_parameters()
    params2 = net2.named_parameters()
    params_interpolated = interpolated_net.named_parameters()
    dict_params2 = dict(params2)
    dict_params_interpolated = dict(params_interpolated)
    for name1, param1 in params1:
        if name1 in dict_params_interpolated:
            dict_params_interpolated[name1].data.copy_(alpha*param1.data + (1-alpha)*dict_params2[name1].data)
    interpolated_net.load_state_dict(dict_params_interpolated)
    return interpolated_net

def weight_diff_btw_nets(net1,net2):
    '''
        Computes the difference between the net1 & net2 in the weight space.
    '''
    params1 = net1.named_parameters()
    params2 = net2.named_parameters()
    dict_params2 = dict(params2)
    total_norm_squared = 0
    for name1, param1 in params1:
        if name1 in dict_params2:
            W1 = param1.data
            W2 = dict_params2[name1].data
            total_norm_squared += (W1-W2).norm(2)**2
    return total_norm_squared**0.5

##

def get_all_radius_errors_loss_list(nb_dirs, net,r_large,rs,device,stats_collector,criterion,error_criterion,trainloader,testloader,iterations):
    '''
    '''
    for dir_index in range(nb_dirs):
        get_radius_errors_loss_list(dir_index, net,r_large,rs,device,stats_collector,criterion,error_criterion,trainloader,testloader,iterations)

def get_radius_errors_loss_list(dir_index, net,r_large,rs,device,stats_collector,criterion,error_criterion,trainloader,testloader,iterations):
    '''
        Computes I = [..., I(W+r*dx),...]. A sequence of errors/losses
        from a starting minimum W to the final minum net r*dx.
        The goal of this is for the easy of computation of the epsilon radius of
        a network which is defined as follows:
            r(dx,eps,W) = sup{r \in R : |I(W) - I(W+r*dx)|<=eps}
        W_all = r*dx
        dx = isotropic unit vector from the net
    '''
    ''' record reference errors/losses '''
    stats_collector.record_errors_loss_reference_net(criterion,error_criterion,net,trainloader,testloader,device)
    ''' get isotropic direction '''
    nb_params = nn_mdls.count_nb_params(net)
    #mvg_sampler = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(nb_params), torch.eye(nb_params))
    v = torch.normal(torch.zeros(nb_params),torch.ones(nb_params)).to(device)
    dx = v/v.norm(2)
    ''' fill up I list '''
    net_r = copy.deepcopy(net)
    for epoch,r in enumerate(rs):
        ''' compute I(W+r*dx) = I(W+W_all)'''
        net_r = translate_net_by_rdx(net,net_r,r,dx)
        Er_train_loss, Er_train_error = evalaute_running_mdl_data_set(criterion,error_criterion,net_r,trainloader,device)
        Er_test_loss, Er_test_error = evalaute_running_mdl_data_set(criterion,error_criterion,net_r,testloader,device)
        ''' record result '''
        stats_collector.append_losses_errors_accs(Er_train_loss, Er_train_error, Er_test_loss, Er_test_error)
        errors_losses = [Er_train_loss,Er_train_error,Er_test_loss,Er_test_error]
        stats_collector.append_all_losses_errors_accs(dir_index,epoch,errors_losses)
    return Er_train_loss, Er_train_error, Er_test_loss, Er_test_error, net_r

def produce_new_translated_net(net_start,r,dx):
    '''
        translate reference net net by r*dx and store it in net_r
    '''
    net_translated = copy.deepcopy(net_start)
    params_translated = net_translated.named_parameters()
    dict_params_transalted = dict(params_translated)
    W_all = r*dx
    ''' '''
    params = net_start.named_parameters()
    i_start, i_end = 0, 0
    for name, W in params:
        ''' get relevant parameters from random translation '''
        i_end = i_start+W.numel()
        #W_relevant = W_all[i_start:i_end] #index is exclusive
        W_relevant = W_all[i_start:i_end].view(W.size())
        ''' translate original net by r*dx[relevant] = W_all[relevant]'''
        if name in dict_params_transalted:
            dict_params_transalted[name].data.copy_(W.data+W_relevant)
        ''' change index to the next relevant params from the random translation '''
        i_start = i_end # index is exclusive so we are already at the right place
    net_translated.load_state_dict(dict_params_transalted)
    return net_translated

##

def get_all_radius_errors_loss_list_interpolate(nb_dirs, net,r_large,interpolations,device,stats_collector,criterion,error_criterion,trainloader,testloader,iterations):
    '''
    '''
    for dir_index in range(nb_dirs):
        get_radius_errors_loss_list_via_interpolation(dir_index, net,r_large,interpolations,device,stats_collector,criterion,error_criterion,trainloader,testloader,iterations)

def get_radius_errors_loss_list_via_interpolation(dir_index, net,r_large,interpolations,device,stats_collector,criterion,error_criterion,trainloader,testloader,iterations):
    '''
        Computes I = [..., I(W+r*dx),...]. A sequence of errors/losses
        from a starting minimum W to the final minum net r*dx.
        The goal of this is for the easy of computation of the epsilon radius of
        a network which is defined as follows:
            r(dx,eps,W) = sup{r \in R : |I(W) - I(W+r*dx)|<=eps}
        W_all = r*dx
        dx = isotropic unit vector from the net
    '''
    ''' record reference errors/losses '''
    stats_collector.record_errors_loss_reference_net(criterion,error_criterion,net,trainloader,testloader,device)
    ''' get isotropic direction '''
    nb_params = nn_mdls.count_nb_params(net)
    v = torch.normal(torch.zeros(nb_params),torch.ones(nb_params)).to(device)
    dx = v/v.norm(2)
    ''' fill up I list '''
    net_r = copy.deepcopy(net)
    net_end = produce_new_translated_net(net,r_large,dx)
    #print(f'||net1 - net2|| = {weight_diff_btw_nets(net_end,net)}')
    for epoch,alpha in enumerate(interpolations):
        ''' compute I(W+r*dx) = I(W+W_all)'''
        net_r = convex_interpolate_nets(net_r,net1=net_end,net2=net,alpha=alpha) # alpha*net_end+(1-alpha)*net
        Er_train_loss, Er_train_error = evalaute_running_mdl_data_set(criterion,error_criterion,net_r,trainloader,device,iterations)
        Er_test_loss, Er_test_error = evalaute_running_mdl_data_set(criterion,error_criterion,net_r,testloader,device,iterations)
        ''' record result '''
        stats_collector.append_losses_errors_accs(Er_train_loss, Er_train_error, Er_test_loss, Er_test_error)
        errors_losses = [Er_train_loss,Er_train_error,Er_test_loss,Er_test_error]
        stats_collector.append_all_losses_errors_accs(dir_index,epoch,errors_losses)
        ''' record current r '''
        r = alpha*r_large
        stats_collector.rs.append(r)
    #stats_collector.random_dirs.append(dx.cpu().numpy())
    return Er_train_loss, Er_train_error, Er_test_loss, Er_test_error, net_r

##

class RandLandscapeInspector:

    def __init__(self,epsilon,net,r_initial, device,criterion,error_criterion,trainloader,testloader,iterations):
        '''
        '''
        self.epsilon = epsilon
        self.net = net
        self.r_initial = r_initial
        ''' '''
        self.device = device
        #self.stats_collector = stats_collector
        ''' '''
        self.criterion = criterion
        self.error_criterion = error_criterion
        self.trainloader = trainloader
        self.testloader = testloader
        self.iterations = iterations
        ''' '''
        self.flatness_radii = []

    def get_faltness_radii_for_isotropic_directions(self,nb_dirs,precision=0.0001):
        '''

        note: we could map dict(direction->radius) if we wanted, perhaps for debugging
        '''
        for dir_index in range(nb_dirs):
            #r = self.get_random_faltness_radius_repeated_doubling(precision=precision)
            r = self.get_random_faltness_radius_upper_lower_bs(precision=precision)
            print(f'r = {r}')
            self.flatness_radii.append(r)

    def get_random_faltness_radius_repeated_doubling(self,precision=0.0001):
        '''
            The goal of this is for the easy of computation of the epsilon radius of
            a network which is defined as follows:
                r(dx,eps,W) = sup{r \in R : |I(W) - I(W+r*dx)|<=eps}
            W_all = r*dx
            dx = isotropic unit vector from the net
        '''
        ''' record reference errors/losses '''
        #stats_collector.record_errors_loss_reference_net(criterion,error_criterion,net,trainloader,testloader,device)
        ''' get isotropic direction '''
        nb_params = nn_mdls.count_nb_params(self.net)
        v = torch.normal(torch.zeros(nb_params),torch.ones(nb_params)).to(self.device)
        dx = v/v.norm(2)
        ''' fill up I list '''
        r = self.r_initial
        Loss_minima,Error_minima = evalaute_running_mdl_data_set(self.criterion,self.error_criterion,self.net,self.trainloader,self.device,self.iterations)
        while True:
            net_rdx = produce_new_translated_net(self.net,r,dx)
            Loss_rdx,Error_rdx = evalaute_running_mdl_data_set(self.criterion,self.error_criterion,net_rdx,self.trainloader,self.device,self.iterations)
            diff = Error_rdx - Error_minima
            print(f'\n--\nr = {r}')
            print(f'Error_minima={Error_minima}')
            print(f'Error_rdx={Error_rdx}')
            print(f'diff = {diff}')
            print(f'epsilon={self.epsilon}')
            print(f'abs(abs(diff)-eps)={abs(abs(diff)-self.epsilon)}')
            print(f'precision={precision}')
            print(f'abs(abs(diff)-eps) < precision={abs(abs(diff)-self.epsilon) < precision}')
            print(f'approx_equals(diff,self.epsilon,precision=precision)={approx_equals(diff,self.epsilon,precision=precision)}')
            ''' check if we reached epsilon jump '''
            if approx_equals(diff,self.epsilon,precision=precision): ## 10^-4.5 for half machine precision
                ''' compute I(W+r*dx) = I(W+W_all)'''
                st()
                return r
            elif diff > self.epsilon: # I(w+rdx) - I(W) > eps, r is too large
                r/=2
            else: # I(w+rdx) - I(W) < eps, r is too small
                r*=1.5

    def get_random_faltness_radius_upper_lower_bs(self,precision=0.0001):
        '''
            The goal of this is for the easy of computation of the epsilon radius of
            a network which is defined as follows:
                r(dx,eps,W) = sup{r \in R : |I(W) - I(W+r*dx)|<=eps}
            W_all = r*dx
            dx = isotropic unit vector from the net
        '''
        ''' record reference errors/losses '''
        #stats_collector.record_errors_loss_reference_net(criterion,error_criterion,net,trainloader,testloader,device)
        ''' get isotropic direction '''
        nb_params = nn_mdls.count_nb_params(self.net)
        v = torch.normal(torch.zeros(nb_params),torch.ones(nb_params)).to(self.device)
        dx = v/v.norm(2)
        ''' set up '''
        lb,ub = 0,2*self.r_initial
        _,error_minima = evalaute_running_mdl_data_set(self.criterion,self.error_criterion,self.net,self.trainloader,self.device,self.iterations)
        y_target = error_minima + self.epsilon # I(W) + eps = I(W+0*dx) + eps
        def f(r):
            net_rdx = produce_new_translated_net(self.net, r, dx)
            Loss_rdx, Error_rdx = evalaute_running_mdl_data_set(self.criterion, self.error_criterion, net_rdx, self.trainloader,self.device, self.iterations)
            return Error_rdx - y_target
        ''' start bisect method ''' # https://en.wikipedia.org/wiki/Bisection_method
        f_lb = error_minima - y_target
        #f_ub = f(ub)
        while True:
            r = (lb + ub)/2
            f_r = f(r) # note this value could be negative
            # print('--')
            # print(f'lb={lb}')
            # print(f'ub={ub}')
            # print(f'r = {r}')
            # print(f'f(r) = f({r}) = {f_r}')
            # print(f'f(lb) = f({lb}) = {f_lb}')
            # print(f'abs(f_r - 0) < precision = {abs(f_r - 0) < precision}')
            # print(f'abs(ub - lb) < precision = {abs(ub - lb) < precision}')
            ''' check if we reached epsilon jump '''
            if abs(f_r - 0) < precision or abs(ub - lb) < precision: ## 10^-4.5 for half machine precision
                ''' compute I(W+r*dx) = I(W+W_all)'''
                return r
            elif np.sign(f_r) == np.sign(f_lb):
                lb = r
                f_lb = f_r
            else:
                ub = r
                #f_ub = f_r

####

def print_evaluation_of_nets(net_nl,net_rlnl,criterion,error_criterion,trainloader,testloader,device,iterations=inf,l=2):
    W_nl = get_norm(net_nl, l=l)
    W_rlnl = get_norm(net_rlnl, l=l)
    print('')
    print(f'W_nl = {W_nl}')
    print(f'W_rlnl = {W_rlnl}')
    ''' train errors '''
    loss_nl_train, error_nl_train = evalaute_running_mdl_data_set(criterion, error_criterion, net_nl, trainloader, device,iterations)
    loss_rlnl_train, error_rlnl_train = evalaute_running_mdl_data_set(criterion, error_criterion, net_rlnl, trainloader, device,iterations)
    print(f'loss_nl_train, error_nl_train = {loss_nl_train, error_nl_train}')
    print(f'loss_rlnl_train, error_rlnl_train = {loss_rlnl_train, error_rlnl_train}')
    ''' test errors '''
    loss_nl_test, error_nl_test = evalaute_running_mdl_data_set(criterion, error_criterion, net_nl, testloader, device,iterations)
    loss_rlnl_test, error_rlnl_test = evalaute_running_mdl_data_set(criterion, error_criterion, net_rlnl, testloader, device,iterations)
    print(f'loss_nl_test, error_nl_test = {loss_nl_test, error_nl_test}')
    print(f'loss_rlnl_test, error_rlnl_test = {loss_rlnl_test, error_rlnl_test}')

def get_norm(net,l=2):
    w_norms = 0
    for index, W in enumerate(net.parameters()):
        w_norms += W.norm(l)
    return w_norms

def divide_params_by(W,net):
    '''
        W: make sure W is non-trainable if you wish to divide by a constant.
    '''
    params = net.named_parameters()
    dict_params = dict(params)
    for name, param in dict_params.items():
        if name in dict_params:
            new_param = param/W
            dict_params[name] = new_param
    net.load_state_dict(dict_params)
    return net

def divide_params_by_taking_bias_into_account(W,net):
    '''
        Divides by making sure the later layers are divided by the right constant

        W: make sure W is non-trainable if you wish to divide by a constant.
    '''
    params = net.named_parameters()
    dict_params = dict(params)
    ''' '''
    l=0
    current_layer_name = ''
    for name, param in dict_params.items():
        prefix,suffix = name.split('.')
        if prefix != current_layer_name: # detect if we changed layer
            current_layer_name = prefix
            l+=1
        #print(f'l={l}, name={name}')
        if name in dict_params:
            if suffix == 'weight':
                new_param = param / W
            else: # else bias
                new_param = param / (W**l)
            dict_params[name] = new_param
    net.load_state_dict(dict_params)
    return net

def l2_norm_all_params(net):
    '''
        print l2 norm of all params
    '''
    params = net.named_parameters()
    dict_params = dict(params)
    for name, param in dict_params.items():
        if name in dict_params:
            prefix, suffix = name.split('.')
            print(f'name = {name}')
            print(f'|params.norm(2) = ||params|| = {param.norm(2)}')

# class WeightNormInspector:
#
#     def __init__(self,net_nl,net_rlnl,device,criterion,error_criterion,trainloader,testloader,iterations):
#         self.net_nl = net_nl
#         self.net_rlnl = net_rlnl
#         ''' '''
#         self.device = device
#         #self.stats_collector = stats_collector
#         ''' '''
#         self.criterion = criterion
#         self.error_criterion = error_criterion
#         self.trainloader = trainloader
#         self.testloader = testloader
#         self.iterations = iterations
#
#     def get_pert_norm(self,l=2):
#         w_norms = 0
#         for index, W in enumerate(self.net_pert.parameters()):
#             w_norms += W.norm(l)
#         return w_norms
#
#     def compare(self):


