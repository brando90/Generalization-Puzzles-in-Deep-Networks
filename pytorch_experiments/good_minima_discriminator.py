import torch
import copy

from new_training_algorithms import evalaute_mdl_data_set

from pdb import set_trace as st

def get_errors_for_all_perturbations(net,perturbation_magnitudes,use_w_norm2,enable_cuda,nb_perturbation_trials,stats_collector,criterion,error_criterion,trainloader,testloader):
    '''
        Evaluate the errors of perturbed models.
    '''
    for index_perturbation_trial in range(nb_perturbation_trials):
        train_loss, train_error, test_loss, test_error = perturb_model(net,perturbation_magnitudes,use_w_norm2,enable_cuda,stats_collector,criterion,error_criterion,trainloader,testloader)
    return train_loss, train_error, test_loss, test_error #note this is just some random trial

def perturb_model(net,perturbation_magnitudes,use_w_norm2,enable_cuda,stats_collector,criterion,error_criterion,trainloader,testloader):
    perturbations = []
    ''' pertub model '''
    for index, W in enumerate(net.parameters()):
        reference_magnitude = 1 if use_w_norm2 else W.norm(2)
        if perturbation_magnitudes[index] != 0:
            std = perturbation_magnitudes[index]*reference_magnitude
            perturbation = torch.normal(means=0.0*torch.ones(W.size()),std=std)
        else:
            perturbation = torch.zeros(W.size())
        perturbation = perturbation.cuda() if enable_cuda else perturbation
        perturbations.append(perturbation)
        #with torch.no_grad():
        W.data.copy_(W.data + perturbation)
    ''' evalaute model '''
    train_loss, train_error = evalaute_mdl_data_set(criterion,error_criterion,net,trainloader,enable_cuda)
    test_loss, test_error = evalaute_mdl_data_set(criterion,error_criterion,net,testloader,enable_cuda)
    ''' record result '''
    stats_collector.append_losses_errors_accs(train_loss, train_error, test_loss, test_error)
    stats_collector.collect_mdl_params_stats(net)
    ''' undo the perturbation '''
    for index, W in enumerate(net.parameters()):
        #with torch.no_grad():
        W.data.copy_(W.data - perturbation[index])
    return train_loss, train_error, test_loss, test_error #note this is just some random trial

####

def get_landscapes_stats_between_nets(net1,net2, nb_interpolations, enable_cuda,stats_collector,criterion,error_criterion,trainloader,testloader):
    '''
        Records the errors for the path by convexly averaging two nets. The goal
        is to be able to estimate the size of the wall between the two different minimums.
    '''
    interpolations = np.linspace(0,1,nb_interpolations)
    ''' '''
    interpolated_net = copy.deepcopy(net1)
    for i,alpha in enumerate(interpolations):
        ''' interpolate nets with current alpha '''
        interpolated_net = convex_interpolate_nets(net1,net2,alpha)
        ''' evalaute model '''
        net = interpolated_net
        train_loss, train_error = evalaute_mdl_data_set(criterion,error_criterion,net,trainloader,enable_cuda)
        test_loss, test_error = evalaute_mdl_data_set(criterion,error_criterion,net,testloader,enable_cuda)
        ''' record result '''
        stats_collector.append_losses_errors_accs(train_loss, train_error, test_loss, test_error)
        stats_collector.collect_mdl_params_stats(interpolated_net)
    return train_loss, train_error, test_loss, test_error #note this is just some random trial

def convex_interpolate_nets(interpolated_net,net1,net2,alpha):
    '''
        Convex interpolation of two nets alpha*W_l + (1-alpha)*W_l.
    '''
    params1 = model1.named_parameters()
    params2 = model2.named_parameters()
    params_interpolated = interpolated_net.named_parameters()
    dict_params_interpolated = dict(params_interpolated)
    for name1, param1 in params1:
        if name1 in dict_params_interpolated:
            dict_params_interpolated[name1].data.copy_(alpha*param1.data + (1-alpha)*dict_params2[name1].data)
    interpolated_net.load_state_dict(dict_params_interpolated)
    return interpolated_net
