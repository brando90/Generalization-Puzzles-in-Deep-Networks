import torch

from utils import is_NaN

from pdb import set_trace as st

def calc_loss(mdl,loss,X,Y):
    loss_val = loss(input=mdl(X),target=Y)
    return loss_val

def calc_accuracy(mdl,X,Y):
    # TODO: why can't we call .data.numpy() for train_acc as a whole?
    max_vals, max_indices = torch.max(mdl(X),1)
    train_acc = (max_indices == Y).sum().item()/max_indices.size()[0]
    return train_acc

def calc_error(mdl,X,Y):
    max_vals, max_indices = torch.max(mdl(X),1)
    train_acc = 1 - (max_indices == Y).sum().item()/max_indices.size()[0]
    return train_acc.item()

# def error_criterion_subtract_1_method(outputs,labels):
#     max_vals, max_indices = torch.max(outputs,1)
#     train_acc = 1 - (max_indices == labels).sum().data[0]/max_indices.size()[0]
#     return train_acc

def error_criterion(outputs,labels):
    max_vals, max_indices = torch.max(outputs,1)
    error = (max_indices != labels).float().sum()/max_indices.size()[0]
    return error
