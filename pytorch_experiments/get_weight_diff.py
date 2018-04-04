import torch

from good_minima_discriminator import weight_diff_btw_nets
import os

results_root = './test_runs_flatness'
enable_cuda=True
''' load nets '''
results_root = './test_runs_flatness'
# NL
path_nl = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_0.0_exptlabel_BoixNet_polestar_300_stand_natural_labels/net_28_March_206')
net_nl = torch.load(path_nl)
# RLNL
path_rlnl = os.path.join(results_root,'flatness_28_March_label_corrupt_prob_0.0_exptlabel_re_train_RLBoixNet_noBN_polestar_150/net_28_March_18')
net_rlnl = torch.load(path_rlnl)
''' compute weight diff '''
total_norm = weight_diff_btw_nets(net_nl,net_rlnl)
print(f'||W1 - W2|| = {total_norm}')
