import torch

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
        std = perturbation_magnitudes[index]*reference_magnitude
        perturbation = torch.normal(means=0.0*torch.ones(W.size()),std=std)
        perturbation = perturbation.cuda() if enable_cuda else perturbation
        perturbations.append(perturbation)
        #with torch.no_grad():
        W.data.copy_(W.data + perturbation)
    ''' evalaute model '''
    train_loss, train_error = evalaute_mdl_data_set(criterion,error_criterion,net,trainloader,enable_cuda)
    test_loss, test_error = evalaute_mdl_data_set(criterion,error_criterion,net,testloader,enable_cuda)
    ''' record result '''
    stats_collector.append_losses_errors_accs(train_loss, train_error, test_loss, test_error)
    ''' undo the perturbation '''
    for index, W in enumerate(net.parameters()):
        #with torch.no_grad():
        W.data.copy_(W.data - perturbation[index])
    return train_loss, train_error, test_loss, test_error #note this is just some random trial
