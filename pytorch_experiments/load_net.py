import torch
import math

import os

import metrics
import utils
import data_classification as data_class
import nn_models as nn_mdls
from new_training_algorithms import get_function_evaluation_from_name

from pdb import set_trace as st

print('\nstart')

''' Paths to net '''
results_root = './test_runs_flatness5_ProperOriginalExpt'
#path = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_0.0_exptlabel_re_train_RLBoixNet_noBN_polestar_150/net_28_March_18')
path = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_1.0_exptlabel_BoixNet_om_350_stand_rand_labels/net_28_March_2')
path = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_1.0_exptlabel_BoixNet_om_350_stand_rand_labels/net_28_March_111')
path = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_1.0_exptlabel_BoixNet_om_350_stand_rand_labels/net_28_March_22')
#path = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_1.0_exptlabel_BoixNet_om_350_stand_rand_labels/net_28_March_167')
#path = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_1.0_exptlabel_BoixNet_om_350_stand_rand_labels/net_28_March_0')
path = os.path.join(results_root,'flatness_6_April_label_corrupt_prob_1.0_exptlabel_train_RL1/net_6_April_sj_0_staid_0_seed_28758823811649733')
path = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_1.0_exptlabel_BoixNet_BN_polestar_350_stand_rand_labels/net_28_March_238')
path = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_1.0_exptlabel_BoixNet_polestar_350_stand_rand_labels/net_28_March_215')
path = os.path.join(results_root,'flatness_6_April_label_corrupt_prob_0.0_exptlabel_train_RLNL2/net_6_April_sj_0_staid_0_seed_39485133104469717')
path = os.path.join(results_root,'flatness_6_April_label_corrupt_prob_0.0_exptlabel_train_RLNL2/net_6_April_sj_0_staid_0_seed_45465090904297403')
path = os.path.join(results_root,'flatness_6_April_label_corrupt_prob_0.0_exptlabel_train_NL1_300/net_6_April_sj_0_staid_0_seed_65723867866542355')
path = os.path.join(results_root,'flatness_9_April_label_corrupt_prob_0.0_exptlabel_DEBUG/net_9_April_sj_0_staid_0_seed_20543310490530753')
path = os.path.join(results_root,'flatness_9_April_label_corrupt_prob_0.0_exptlabel_DEBUG/net_9_April_sj_0_staid_0_seed_20543310490530753')
path = os.path.join(results_root,'flatness_July_label_corrupt_prob_0.0_exptlabel_debug_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means__stds__batch_size_train_1024_lr_0.02_momentum_0.95_epochs_200/net_18_July_sj_0_staid_0_seed_25144932459028958_polestar-old')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f'path = {path}')
''' '''
data_set = 'mnist'
data_eval_type = 'evalaute_mdl_on_full_data_set'
evalaute_mdl_data_set = get_function_evaluation_from_name(data_eval_type)
''' data '''
data_path = './data'
trainset, testset, classes = data_class.get_data_processors(data_path, 0.0,dataset_type=data_set,standardize=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True,
                                          num_workers=10)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False,
                                         num_workers=10)
''' Criterion '''
error_criterion = metrics.error_criterion
criterion = torch.nn.CrossEntropyLoss()
iterations = math.inf
''' Nets'''
net = utils.restore_entire_mdl(path).cuda()
#net2 = utils.restore_entire_mdl(path).cuda()
#net3 = utils.restore_entire_mdl(path).cuda()
''' stats about the nets '''
train_loss_epoch, train_error_epoch = evalaute_mdl_data_set(criterion, error_criterion, net,trainloader,device)
test_loss_epoch, test_error_epoch = evalaute_mdl_data_set(criterion, error_criterion, net,testloader,device)
nb_params = nn_mdls.count_nb_params(net)
''' print net stats '''
print(f'train_loss_epoch, train_error_epoch  = {train_loss_epoch}, {train_error_epoch}')
print(f'test_loss_epoch, test_error_epoch  = {test_loss_epoch}, {test_error_epoch}')
print(f'nb_params {nb_params}')
''' '''
st()
''' END '''
print('end no issue \a')
