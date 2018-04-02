import torch

import os

import utils
import data_classification as data_class
import nn_models as nn_mdls

results_root = './test_runs_flatness'
path = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_0.0_exptlabel_re_train_RLBoixNet_noBN_polestar_150/net_28_March_18')
enable_cuda=True

data_path = './data'
trainset,trainloader, testset,testloader, classes = data_class.get_cifer_data_processors(data_path,256,256,0,0,standardize=True)

net = utils.restore_entire_mdl(path).cuda()
net2 = utils.restore_entire_mdl(path).cuda()
#net3 = utils.restore_entire_mdl(path).cuda()

nb_params = nn_mdls.count_nb_params(net)
print(f'nb_params {nb_params}')
v = torch.normal(torch.zeros(nb_params),torch.eye(nb_params)).cuda()

print('end no issue')
