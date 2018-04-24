import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import os
import pickle

import utils
import data_classification as data_class
from new_training_algorithms import extract_data
from data_classification import get_standardized_transform
from data_classification import IndxCifar10

from pdb import set_trace as st

def get_second_largest(scores,max_score):
    ## delete max
    ## now get max
    return

def save_index_according_to_criterion(path,dataloader_standardize,dataloader_pixels,net):
    '''
        Creates data set to measure sharpness
    '''
    ''' produce list of scores score_list = [(i,score)] '''
    enable_cuda = True
    score_list = []
    for i_old,(inputs,labels,indices) in enumerate(dataloader_standardize):
        inputs,labels = extract_data(enable_cuda,(inputs,labels),wrap_in_variable=True)
        scores = net(inputs)
        st()
        max_scores, max_indices = torch.max(scores) # float
        second_largest_scores, new_label = get_second_largest(scores,max_scores)
        score_list.append( (i_old,new_label,max_score) )
    ''' sort(scores list), based on scores '''
    sorting_criterion = lambda tup: tup[1]
    ## note sorted: i_new -> ( i_old, l^(i_old), s_^(i_old) )
    sorted_scores = sorted(score_list, key=sorting_criterion) # smallest to largest
    ''' old 2 new mapping'''
    ''' '''
    X_new = np.zero((50000,3,32,32))
    Y_new = np.zero((50000))

def main():
    ''' get data loaders '''
    transform = get_standardized_transform()
    dataset_standardize = IndxCifar10(transform=transform)
    dataset_pixels = IndxCifar10(transform=lambda x: x)
    dataloader_pixels = DataLoader(dataset_pixels,batch_size=4,shuffle=False,num_workers=1)
    dataloader_standardize = DataLoader(dataset_standardize,batch_size=4,shuffle=False,num_workers=1)
    ''' load net for the criterion (we are perturbing to give the sharpest) '''
    results_root = './test_runs_flatness'
    # path_2_net = 'TODO' # NL
    # path_2_net = os.path.join(results_root,'flatness_22_April_label_corrupt_prob_1.0_exptlabel_GB_15_13_10_154229_BN_RL/net_22_April_sj_10583197_staid_7_seed_37801283806432755') #RLNL
    path_2_net = os.path.join(results_root,'flatness_22_April_label_corrupt_prob_1.0_exptlabel_Net_13_12_10_123434_RL/net_22_April_sj_10577986_staid_2_seed_39037026647362915')
    net = torch.load(path_2_net)
    ''' create new data set '''
    path_train = 'TODO'
    save_index_according_to_criterion(path_train,dataloader_standardize,dataloader_pixels,net)
    # path_test = 'TODO'
    # save_index_according_to_criterion(path_test,testdataloader, criterion=net)

if __name__ == '__main__':
    main()