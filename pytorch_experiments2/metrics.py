import torch

from pdb import set_trace as st

def is_NaN(value):
    '''
    Checks is value is problematic by checking if the value:
    is not finite, is infinite or is already NaN
    '''
    return not np.isfinite(value) or np.isinf(value) or np.isnan(value)

def calc_loss(mdl,loss,X,Y):
    loss_val = loss(input=mdl(X),target=Y)
    return loss_val

def calc_accuracy(mdl,X,Y):
    # TODO: why can't we call .data.numpy() for train_acc as a whole?
    max_vals, max_indices = torch.max(mdl(X),1)
    train_acc = (max_indices == Y).sum().data[0]/max_indices.size()[0]
    return train_acc

def calc_error(mdl,X,Y):
    max_vals, max_indices = torch.max(mdl(X),1)
    train_acc = 1 - (max_indices == Y).sum().data[0]/max_indices.size()[0]
    return train_acc

def error_criterion(outputs,labels):
    max_vals, max_indices = torch.max(outputs,1)
    train_acc = 1 - (max_indices == labels).sum().data[0]/max_indices.size()[0]
    return train_acc

def error_criterion2(outputs,labels):
    max_vals, max_indices = torch.max(outputs,1)
    train_acc = (max_indices != labels).sum().data[0]/max_indices.size()[0]
    return train_acc
