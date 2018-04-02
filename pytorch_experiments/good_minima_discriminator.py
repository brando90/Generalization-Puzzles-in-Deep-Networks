import torch
import copy

import numpy as np

from new_training_algorithms import evalaute_mdl_data_set
import nn_models as nn_mdls

from pdb import set_trace as st

def get_errors_for_all_perturbations(net,perturbation_magnitudes,relative,std_dict_params,enable_cuda,nb_perturbation_trials,stats_collector,criterion,error_criterion,trainloader,testloader):
    '''
        Evaluate the errors of perturbed models.
    '''
    std_dict_params = get_std_of_net(net)
    for index_perturbation_trial in range(nb_perturbation_trials):
        train_loss, train_error, test_loss, test_error = perturb_model(net,perturbation_magnitudes,std_dict_params,enable_cuda,stats_collector,criterion,error_criterion,trainloader,testloader)
    return train_loss, train_error, test_loss, test_error #note this is just some random trial

def perturb_model(net,perturbation_magnitudes,std_dict_params,relative,enable_cuda,stats_collector,criterion,error_criterion,trainloader,testloader):
    perturbations = []
    ''' pertub model '''
    params = net.named_parameters()
    index = 0
    for name, W in params:
        if perturbation_magnitudes[index] != 0:
            reference_magnitude = std_dict_params[name] if relative else 1
            std = perturbation_magnitudes[index]*reference_magnitude
            perturbation = torch.normal(means=0.0*torch.ones(W.size()),std=std)
        else:
            perturbation = torch.zeros(W.size())
        perturbation = perturbation.cuda() if enable_cuda else perturbation
        W.data.copy_(W.data + perturbation)
        ##
        perturbations.append(perturbation.norm(2))
        i+=1
    ''' evalaute model '''
    train_loss, train_error = evalaute_mdl_data_set(criterion,error_criterion,net,trainloader,enable_cuda)
    test_loss, test_error = evalaute_mdl_data_set(criterion,error_criterion,net,testloader,enable_cuda)
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

def get_landscapes_stats_between_nets(net1,net2, interpolations, enable_cuda,stats_collector,criterion,error_criterion,trainloader,testloader):
    '''
        Records the errors for the path by convexly averaging two nets. The goal
        is to be able to estimate the size of the wall between the two different minimums.
    '''
    ''' '''
    interpolated_net = copy.deepcopy(net1)
    for i,alpha in enumerate(interpolations):
        ''' interpolate nets with current alpha '''
        interpolated_net = convex_interpolate_nets(interpolated_net,net1,net2,alpha)
        ''' evalaute model '''
        train_loss, train_error = evalaute_mdl_data_set(criterion,error_criterion,interpolated_net,trainloader,enable_cuda)
        test_loss, test_error = evalaute_mdl_data_set(criterion,error_criterion,interpolated_net,testloader,enable_cuda)
        ''' record result '''
        stats_collector.append_losses_errors_accs(train_loss, train_error, test_loss, test_error)
        stats_collector.collect_mdl_params_stats(interpolated_net)
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

##

def get_radius_errors_loss_list(net,r_large,rs,enable_cuda,stats_collector,criterion,error_criterion,trainloader,testloader):
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
    stats_collector.record_errors_loss_reference_net(criterion,error_criterion,net,trainloader,testloader,enable_cuda)
    ''' get isotropic direction '''
    nb_params = nn_mdls.count_nb_params(net)
    #mvg_sampler = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(nb_params), torch.eye(nb_params))
    #v = mvg_sampler.sample().cuda()
    v = torch.normal(torch.zeros(nb_params),torch.eye(nb_params)).cuda() if enable_cuda else torch.normal(torch.zeros(nb_params),torch.ones(nb_params))
    dx = v/v.norm(2)
    ''' fill up I list '''
    net_r = copy.deepcopy(net)
    for r in rs:
        ''' compute I(W+r*dx) = I(W+W_all)'''
        net_r = translate_net_by_rdx(net,net_r,r,dx)
        Er_train_loss, Er_train_error = evalaute_mdl_data_set(criterion,error_criterion,net_r,trainloader,enable_cuda)
        Er_test_loss, Er_test_error = evalaute_mdl_data_set(criterion,error_criterion,net_r,testloader,enable_cuda)
        ''' record result '''
        stats_collector.append_losses_errors_accs(Er_train_loss, Er_train_error, Er_test_loss, Er_test_error)
    return Er_train_loss, Er_train_error, Er_test_loss, Er_test_error, net_r

def translate_net_by_rdx(net,net_r,r,dx):
    '''
        translate reference net net by r*dx and store it in net_r
    '''
    params_r = net_r.named_parameters()
    dict_params_r = dict(params_r)
    W_all = r*dx
    ''' '''
    i_start, i_end = 0, 0
    for name, W in enumerate(net_r.parameters()):
        ''' get relevant parameters from random translation '''
        i_end = i_start+W.numel()
        #W_relevant = W_all[i_start:i_end] #index is exclusive
        W_relevant = W_all[i_start:i_end].view(W.size())
        ''' translate original net by r*dx[relevant] = W_all[relevant]'''
        if name in dict_params_r:
            dict_params_r[name].data.copy_(W+W_relevant)
        ''' change index to the next relevant params from the random translation '''
        i_start = i_end # index is exclusive so we are already at the right place
    net_r.load_state_dict(dict_params_r)
    return net_r
