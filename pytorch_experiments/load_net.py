import torch
import math

import os

import metrics
import utils
import data_classification as data_class
import nn_models as nn_mdls
from new_training_algorithms import evalaute_mdl_data_set

print('start')

''' Paths to net '''
results_root = './test_runs_flatness'
#path = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_0.0_exptlabel_re_train_RLBoixNet_noBN_polestar_150/net_28_March_18')
path = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_1.0_exptlabel_BoixNet_om_350_stand_rand_labels/net_28_March_2')
path = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_1.0_exptlabel_BoixNet_om_350_stand_rand_labels/net_28_March_111')
path = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_1.0_exptlabel_BoixNet_om_350_stand_rand_labels/net_28_March_22')
#path = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_1.0_exptlabel_BoixNet_om_350_stand_rand_labels/net_28_March_167')
#path = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_1.0_exptlabel_BoixNet_om_350_stand_rand_labels/net_28_March_0')
enable_cuda=True
''' data '''
data_path = './data'
trainset,trainloader, testset,testloader, classes = data_class.get_cifer_data_processors(data_path,256,256,0,0,standardize=True)
''' Criterion '''
error_criterion = metrics.error_criterion
criterion = torch.nn.CrossEntropyLoss()
iterations = math.inf
''' Nets'''
net = utils.restore_entire_mdl(path).cuda()
#net2 = utils.restore_entire_mdl(path).cuda()
#net3 = utils.restore_entire_mdl(path).cuda()
''' stats about the nets '''
train_loss_epoch, train_error_epoch = evalaute_mdl_data_set(criterion,error_criterion,net,trainloader,enable_cuda,iterations=iterations)
test_loss_epoch, test_error_epoch = evalaute_mdl_data_set(criterion,error_criterion,net,testloader,enable_cuda,iterations=iterations)
nb_params = nn_mdls.count_nb_params(net)
''' print net stats '''
print(f'train_loss_epoch, train_error_epoch  = {train_loss_epoch}, {train_error_epoch}')
print(f'test_loss_epoch, test_error_epoch  = {test_loss_epoch}, {test_error_epoch}')
print(f'nb_params {nb_params}')
''' END '''
print('end no issue \a')
