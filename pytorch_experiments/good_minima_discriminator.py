import torch


def add_perturbation(perturbation_magnitudes,use_w_norm2=False):
    '''
        perturbs the whole model base on a array perturbations.
    '''
    for index, W in enumerate(mdl.parameters()):
        perturbation_magnitude = perturbation_magnitudes[i]
        reference_magnitude = 1
        if use_w_norm2:
            reference_magnitude = W.norm(2)
        std = perturbation_magnitude*reference_magnitude
        perturbation = torch.normal(means=0.0*torch.ones(Din,Dout),std=std)
        W.data.copy_(W.data + perturbation)
    return perturbed_net

def remove_perturbation(perturbation_magnitudes,use_w_norm2=False):
    '''
        remove perturbion on whole model base on a array perturbations.
    '''
    perturbation_magnitudes = [-1*delta for delta perturbation_magnitudes]
    unperturbed_net = add_perturbation(perturbation_magnitudes,use_w_norm2=False)
    return unperturbed_net
