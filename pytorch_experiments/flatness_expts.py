#!/usr/bin/env python
#SBATCH --mem=7000
#SBATCH --time=0-05:00
#SBATCH --mail-type=END
#SBATCH --mail-user=brando90@mit.edu
'''
    #SBATCH --gres=gpu:1
'''

"""
training an image classifier so that it overfits
----------------------------

"""
import time
from datetime import date
import calendar

import os
import sys
import subprocess

current_directory = os.getcwd() #The method getcwd() returns current working directory of a process.
sys.path.append(current_directory)

import numpy as np

import torch

from torch.autograd import Variable
import torch.optim as optim

import data_classification as data_class

import nn_models as nn_mdls
import new_training_algorithms as tr_alg
import save_to_matlab_format as save2matlab
from stats_collector import StatsCollector
import metrics
import utils
import plot_utils

from pdb import set_trace as st

import argparse

from maps import NamedDict

import matplotlib as mpl
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('-cuda','--enable-cuda',action='store_true',
                    help='Enable cuda/gpu')
parser.add_argument("-seed", "--seed", type=int, default=None,
                    help="The number of games to simulate")
parser.add_argument("-epochs", "--epochs", type=int, default=None,
                    help="The number of games to simulate")
args = parser.parse_args()
if not torch.cuda.is_available() and args.enable_cuda:
    print('Cuda is enabled but the current system does not have cuda')
    sys.exit()

satid = 0
sj = 0
if 'SLURM_ARRAY_TASK_ID' in os.environ and 'SLURM_JOBID' in os.environ:
    satid = int(os.environ['SLURM_ARRAY_TASK_ID'])
    sj = int(os.environ['SLURM_JOBID'])

def main(plot=False):
    ''' reproducibility setup/params'''
    #num_workers = 2 # how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    githash = subprocess.check_output(["git", "describe", "--always"]).strip()
    seed = args.seed
    if seed is None: # if seed is None it has not been set, so get a random seed, else use the seed that was set
        seed = ord(os.urandom(1))
    print(f'seed: {seed}')
    ## SET SEED/determinism
    num_workers = 0
    torch.manual_seed(seed)
    if args.enable_cuda:
        torch.backends.cudnn.deterministic=True
    ''' date parameters setup'''
    today_obj = date.today() # contains datetime.date(year, month, day); accessible via .day etc
    day = today_obj.day
    month = calendar.month_name[today_obj.month]
    start_time = time.time()
    ''' filenames '''
    label_corrupt_prob = 0
    results_root = './test_runs_flatness'
    expt_path = f'flatness_{day}_{month}_label_corrupt_prob_{label_corrupt_prob}_sj_{sj}'
    matlab_file_name = f'flatness_{day}_{month}_seed_{seed}_staid_{satid}'
    ''' experiment params '''
    nb_epochs = 4 if args.epochs is None else args.epochs
    batch_size = 256
    #batch_size_train,batch_size_test = batch_size,batch_size
    batch_size_train = batch_size
    batch_size_test = 256
    data_path = './data'
    ''' get data set '''
    standardize = True # x - mu / std , [-1,+1]
    trainset,trainloader, testset,testloader, classes = data_class.get_cifer_data_processors(data_path,batch_size_train,batch_size_test,num_workers,label_corrupt_prob,standardize=standardize)
    ''' get NN '''
    mdl = 'debug'
    #mdl = 'cifar_10_tutorial_net'
    #mdl = 'BoixNet'
    #mdl = 'LiaoNet'
    ##
    print(f'model = {mdl}')
    if mdl == 'cifar_10_tutorial_net':
        do_bn = False
        net = nn_mdls.Net()
    elif mdl == 'debug':
        do_bn=False
        nb_conv_layers=3
        ## conv params
        Fs = [32]*nb_conv_layers
        Ks = [5]*nb_conv_layers
        ## fc params
        FC = len(classes)
        C,H,W = 3,32,32
        net = nn_mdls.LiaoNet(C,H,W,Fs,Ks,FC,do_bn)
    elif mdl == 'BoixNet':
        do_bn=False
        ## conv params
        nb_filters1,nb_filters2 = 32, 32
        kernel_size1,kernel_size2 = 5,5
        ## fc params
        nb_units_fc1,nb_units_fc2,nb_units_fc3 = 512,256,len(classes)
        C,H,W = 3,32,32
        net = nn_mdls.BoixNet(C,H,W,nb_filters1,nb_filters2, kernel_size1,kernel_size2, nb_units_fc1,nb_units_fc2,nb_units_fc3,do_bn)
    elif mdl == 'LiaoNet':
        do_bn=False
        nb_conv_layers=3
        ## conv params
        Fs = [32]*nb_conv_layers
        Ks = [5]*nb_conv_layers
        ## fc params
        FC = len(classes)
        C,H,W = 3,32,32
        net = nn_mdls.LiaoNet(C,H,W,Fs,Ks,FC,do_bn)
    # elif mdl == 'MMNISTNet':
    #     net = MMNISTNet()
    if args.enable_cuda:
        net.cuda()
    nb_params = nn_mdls.count_nb_params(net)
    ''' Cross Entropy + Optmizer'''
    lr = 0.01
    momentum = 0.0
    ## Errors
    error_criterion = metrics.error_criterion
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.MultiMarginLoss()
    #criterion = torch.nn.MSELoss(size_average=True)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    ''' stats collector '''
    stats_collector = StatsCollector(net)
    ''' Train the Network '''
    print(f'----\nSTART training: label_corrupt_prob={label_corrupt_prob},nb_epochs={nb_epochs},batch_size={batch_size},lr={lr},mdl={mdl},batch-norm={do_bn},nb_params={nb_params}')
    overparametrized = len(trainset)<nb_params # N < W ?
    print(f'Model over parametrized? N, W = {len(trainset)} vs {nb_params}')
    print(f'Model over parametrized? N < W = {overparametrized}')
    # We simply have to loop over our data iterator, and feed the inputs to the network and optimize.
    #tr_alg.train_cifar(args, nb_epochs, trainloader,testloader, net,optimizer,criterion)
    train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch = tr_alg.train_and_track_stats(args, nb_epochs, trainloader,testloader, net,optimizer,criterion,error_criterion, stats_collector)
    seconds,minutes,hours = utils.report_times(start_time)
    print(f'Finished Training, hours={hours}')
    print(f'seed = {seed}, githash = {githash}')
    ''' Test the Network on the test data '''
    print(f'train_loss_epoch={train_loss_epoch} \ntrain_error_epoch={train_error_epoch} \ntest_loss_epoch={test_loss_epoch} \ntest_error_epoch={test_error_epoch}')
    ''' save results from experiment '''
    other_stats = {'nb_epochs':nb_epochs,'batch_size':batch_size,'mdl':mdl,'lr':lr,'momentum':momentum, 'seed':seed,'githash':githash,
        'seconds':seconds,'minutes':minutes,'hours':hours}
    save2matlab.save2matlab_flatness_expt(results_root,expt_path,matlab_file_name, stats_collector,other_stats=other_stats)
    ''' save net model '''
    path = os.path.join(results_root,expt_path,f'net_{day}_{month}_{seed}')
    utils.save_entire_mdl(path,net)
    restored_net = utils.restore_entire_mdl(path)
    loss_restored,error_restored = tr_alg.evalaute_mdl_data_set(criterion,error_criterion,restored_net,testloader,args.enable_cuda)
    print(f'\nloss_restored={loss_restored},error_restored={error_restored}\a')
    ''' plot '''
    if plot:
        #TODO
        plot_utils.plot_loss_and_accuracies(stats_collector)
        plt.show()

if __name__ == '__main__':
    main(plot=True)
    print('\a')
