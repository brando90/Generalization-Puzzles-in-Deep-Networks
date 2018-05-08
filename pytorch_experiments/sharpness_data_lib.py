#!/usr/bin/env python
#SBATCH --mem=20000
#SBATCH --time=1-10:30
#SBATCH --mail-type=END
#SBATCH --mail-user=brando90@mit.edu
#SBATCH --gres=gpu:1

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import os
import pickle

import math
import numpy as np

import utils
import data_classification as data_class
from data_classification import get_standardized_transform
from data_classification import IndxCifar10

from pdb import set_trace as st

import argparse

parser = argparse.ArgumentParser(description='Sharpness data Creator Submission script')
''' Flags '''
parser.add_argument("-net_name", "--net_name", type=str, default='NL',
                    help="which net to use")
#parser.add_argument("-data_label", "--data_label", type=str, default='NL_nolabel',
#                    help="the label to use to identify experiment in the folder name")

# python sharpness_data_lib.py -net NL
# python sharpness_data_lib.py -net RLNL
''' process args '''
args = parser.parse_args()

##### Debug

def check_images_are_same_index():
    '''
    Checks if the indices in .data_train and how batcher indexes match.
    :return:
    '''
    device = torch.device("cuda" if use_cuda else "cpu")
    transform = get_standardized_transform()
    dataset_standardize = IndxCifar10(transform=transform)
    dataloader_standardize = DataLoader(dataset_standardize,batch_size=2**10,shuffle=False,num_workers=10)
    cifar10 = dataloader_standardize.dataset.cifar10.train_data
    img1 = transform(cifar10[0])
    for i, (inputs, labels, indices) in enumerate(dataloader_standardize):
        inputs, labels = inputs.to(device), labels.to(device)
        img1_batch = inputs[0]
        break
    ''' compare them '''
    print( np.sum(img1.numpy() == img1_batch.data.numpy()) == img1.numel() )

###### Code

def get_second_largest(scores,max_indices):
    '''

    :param scores: M x 10 cuda vector
    :param max_score:
    :param max_index:
    :return:
    '''
    ## delete max by replacing it by -infinity
    #scores[:,max_indices] = -math.inf
    for k,i in enumerate(max_indices):
        scores[k,i] = 0
    ## now get new max = 2nd largest
    second_largest_scores, max_indices = torch.max(scores,1)
    return second_largest_scores, max_indices

def get_old_2_new_mapping(sorted_scores):
    '''

    :param sorted_scores: i_new -> ( i_old, l^(i_old), s_^(i_old) ),
    the indices of sorted_scores are the i_new but they map to i_old
    :return: old_2_new

    loop through sorted_scores
    '''
    N = len(sorted_scores)
    old_2_new = np.zeros(N,dtype=int)
    for i_new in range(N):
        i_old = sorted_scores[i_new][0]
        old_2_new[i_old] = i_new
    return old_2_new

def save_index_according_to_criterion(path_2_save,dataloader_standardize,net, device):
    '''
        Creates data set to measure sharpness
    '''
    ''' produce list of scores score_list = [(i,new_labels,score)] '''
    print('produce score list')
    net.eval()
    score_list = []
    for i,(inputs,labels,indices) in enumerate(dataloader_standardize):
        inputs,labels = inputs.to(device),labels.to(device)
        scores = net(inputs) # M x 10 Float cuda vector
        # get max scores (to later sort original idicies) and get new labels for the data set
        max_scores, max_indices = torch.max(scores,1) # M, M long,float (note max_indices are the labels predicted by net)
        second_largest_scores, new_label = get_second_largest(scores,max_indices) # M, M float,long
        # create [(i_old,new_label,max_score_old_label)]
        new_elements = [ (int(indices[i]),int(new_label[i]),float(max_scores[i])) for i in range(len(scores)) ]
        score_list += new_elements
    ''' sort(scores list) = sort([ (i_old,new_label,max_score) ]), based on scores '''
    print('sort score list')
    sorting_criterion = lambda tup: tup[2]
    sorted_scores = sorted(score_list, key=sorting_criterion) # smallest to largest: get_old_2_new_mapping(sorted_scores):
    ''' old 2 new mapping'''
    print('produce old_2_new mapping')
    old_2_new = get_old_2_new_mapping(sorted_scores)
    ''' produce new dataset '''
    print('about to produce dataset')
    cifar10 = dataloader_standardize.dataset.cifar10.train_data
    N = cifar10.shape[0]
    X_new = np.zeros((N,32,32,3))
    Y_new = np.zeros(N).astype('int')
    for i_old in range(N):
        data = cifar10[i_old]
        ## get location of data
        i_new = old_2_new[i_old]
        X_new[i_new,:,:,:] = data
        ## assign 2nd best label
        new_label = sorted_scores[i_new][1]
        Y_new[i_new] = new_label
    ''' store data '''
    np.savez(path_2_save,X_train=X_new,Y_train=Y_new)

def main():
    print('\nmain')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'device={device}')
    ''' get data loaders '''
    transform = get_standardized_transform()
    dataset_standardize = IndxCifar10(transform=transform)
    #dataset_pixels = IndxCifar10(transform=transforms.ToTensor())
    #dataloader_pixels = DataLoader(dataset_pixels,batch_size=2**10,shuffle=False,num_workers=10)
    dataloader_standardize = DataLoader(dataset_standardize,batch_size=2**10,shuffle=False,num_workers=10)
    ''' load NL '''
    net_name = args.net_name
    results_root = './test_runs_flatness2'
    if net_name == 'NL':
        expt_path = 'flatness_27_April_label_corrupt_prob_0.0_exptlabel_GB_24_24_10_2C1FC_momentum_NL_polestar/'
        full_net_name = 'net_27_April_sj_343_staid_1_seed_56134200848018679'
        path_to_restore = os.path.join(results_root,expt_path,full_net_name)
        print(path_to_restore)
        net = torch.load(path_to_restore)
    else: #RLNL
        expt_path = 'flatness_27_April_label_corrupt_prob_0.0_exptlabel_GB_24_24_10_2C1FC_momentum_RLNL_polestar/'
        full_net_name = 'net_27_April_sj_345_staid_1_seed_57700439347820897'
        path_to_restore = os.path.join(results_root,expt_path,full_net_name)
        net = torch.load(path_to_restore)
    ''' create new data set '''
    folder_path = f'./data/sharpness_data_{net_name}'
    filename = f'sdata_{net_name}_{full_net_name}'
    utils.make_and_check_dir(folder_path)
    path_2_save = os.path.join(folder_path, filename)
    save_index_according_to_criterion(path_2_save,dataloader_standardize,net, device)

if __name__ == '__main__':
    main()
    #check_images_are_same_index()
    ''' print done '''
    print('Done!\a')