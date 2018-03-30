#!/usr/bin/env python
#SBATCH --mem=7000
#SBATCH --time=0-08:00
#SBATCH --mail-type=END
#SBATCH --mail-user=brando90@mit.edu
#SBATCH --array=1-5
#SBATCH --gres=gpu:1

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
from good_minima_discriminator import get_errors_for_all_perturbations, perturb_model

from pdb import set_trace as st

import argparse

from maps import NamedDict

import matplotlib as mpl
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Example')
''' setup params '''
parser.add_argument('-cuda','--enable-cuda',action='store_true',
                    help='Enable cuda/gpu')
parser.add_argument("-seed", "--seed", type=int, default=None,
                    help="The number of games to simulate")
parser.add_argument("-exptlabel", "--exptlabel", type=str, default='nolabel',
                    help="experiment label")
''' NN related arguments '''
parser.add_argument("-epochs", "--epochs", type=int, default=None,
                    help="The number of games to simulate")
parser.add_argument("-mdl", "--mdl", type=str, default='debug',
                    help="experiment label") # options: debug, cifar_10_tutorial_net, BoixNet, LiaoNet
parser.add_argument('-use_bn','--use_bn',action='store_true',
                    help='turns on BN')
parser.add_argument('-standardize_data','--standardize_data',action='store_true',
                    help='uses x-u/s, standardize data')
parser.add_argument("-label_corrupt_prob", "--label_corrupt_prob", type=float, default=0.0,
                    help="The probability of a label getting corrupted")
''' training argument '''
parser.add_argument("-train_alg", "--train_alg", type=str, default='SGD',
                    help="Training algorithm to use")
parser.add_argument("-noise_level", "--noise_level", type=float, default=0.0001,
                    help="Noise level for perturbation")
parser.add_argument("-not_pert_w_norm2", "--not_pert_w_norm2", type=float, action='store_false',
                    help="Noise level for perturbation")
''' process args '''
args = parser.parse_args()
if not torch.cuda.is_available() and args.enable_cuda:
    print('Cuda is enabled but the current system does not have cuda')
    sys.exit()
sj, satid = 0, 0
if 'SLURM_ARRAY_TASK_ID' in os.environ and 'SLURM_JOBID' in os.environ:
    satid = int(os.environ['SLURM_ARRAY_TASK_ID'])
    sj = int(os.environ['SLURM_JOBID'])

def main(plot=False):
    other_stats = dict({'sj':sj,'satid':satid})
    ''' reproducibility setup/params'''
    #num_workers = 2 # how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    githash = subprocess.check_output(["git", "describe", "--always"]).strip()
    seed = args.seed
    if seed is None: # if seed is None it has not been set, so get a random seed, else use the seed that was set
        seed = int.from_bytes(os.urandom(8), byteorder="big")
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
    results_root = './test_runs_flatness'
    expt_folder = f'flatness_{day}_{month}_label_corrupt_prob_{args.label_corrupt_prob}_exptlabel_{args.exptlabel}'
    ## filenames
    matlab_file_name = f'flatness_{day}_{month}_sj_{sj}_staid_{satid}_seed_{seed}'
    net_file_name = f'net_{day}_{month}_sj_{sj}_staid_{satid}_seed_{seed}'
    ## experiment path
    expt_path = os.path.join(results_root,expt_folder)
    utils.make_and_check_dir(expt_path)
    ''' experiment params '''
    nb_epochs = 4 if args.epochs is None else args.epochs
    batch_size = 256
    #batch_size_train,batch_size_test = batch_size,batch_size
    batch_size_train = batch_size
    batch_size_test = 256
    data_path = './data'
    ''' get data set '''
    standardize = args.standardize_data # x - mu / std , [-1,+1]
    trainset,trainloader, testset,testloader, classes = data_class.get_cifer_data_processors(data_path,batch_size_train,batch_size_test,num_workers,args.label_corrupt_prob,standardize=standardize)
    ''' get NN '''
    mdl = args.mdl
    do_bn = args.use_bn
    other_stats = dict({'mdl':mdl,'do_bn':do_bn},**other_stats)
    ##
    print(f'model = {mdl}')
    if mdl == 'cifar_10_tutorial_net':
        net = nn_mdls.Net()
    elif mdl == 'debug':
        nb_conv_layers=1
        ## conv params
        Fs = [3]*nb_conv_layers
        Ks = [2]*nb_conv_layers
        ## fc params
        FC = len(classes)
        C,H,W = 3,32,32
        net = nn_mdls.LiaoNet(C,H,W,Fs,Ks,FC,do_bn)
    elif mdl == 'MirandaNet':
        ## conv params
        nb_filters1,nb_filters2 = 32, 32
        kernel_size1,kernel_size2 = 5,5
        ## fc params
        nb_units_fc1,nb_units_fc2,nb_units_fc3 = 512,256,len(classes)
        C,H,W = 3,32,32
        net = nn_mdls.MirandaNet(C,H,W,nb_filters1,nb_filters2, kernel_size1,kernel_size2, nb_units_fc1,nb_units_fc2,nb_units_fc3,do_bn)
    elif mdl == 'LiaoNet':
        nb_conv_layers=5
        ## conv params
        Fs = [32]*nb_conv_layers
        Ks = [5]*nb_conv_layers
        ## fc params
        FC = len(classes)
        C,H,W = 3,32,32
        net = nn_mdls.LiaoNet(C,H,W,Fs,Ks,FC,do_bn)
    else:
        ''' RESTORED PRE-TRAINED NET '''
        # example name of file, os.path.join(results_root,expt_path,f'net_{day}_{month}_{seed}')
        # args.net_path = 'flatness_27_March_label_corrupt_prob_0_exptlabel_BoixNet_stand_600/net_27_Match_64'
        path_to_mdl = args.mdl
        path = os.path.join(results_root,path_to_mdl)
        net = utils.restore_entire_mdl(path)
    if args.enable_cuda:
        #set_default_tensor_type
        net.cuda()
    else:
        net.cpu()
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
    other_stats = dict({'nb_epochs':nb_epochs,'batch_size':batch_size,'mdl':mdl,'lr':lr,'momentum':momentum, 'seed':seed,'githash':githash},**other_stats)
    ''' Train the Network '''
    print(f'----\nSTART training: label_corrupt_prob={args.label_corrupt_prob},nb_epochs={nb_epochs},batch_size={batch_size},lr={lr},mdl={mdl},batch-norm={do_bn},nb_params={nb_params}')
    overparametrized = len(trainset)<nb_params # N < W ?
    print(f'Model over parametrized? N, W = {len(trainset)} vs {nb_params}')
    print(f'Model over parametrized? N < W = {overparametrized}')
    if args.train_alg == 'SGD':
        # We simply have to loop over our data iterator, and feed the inputs to the network and optimize.
        train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch = tr_alg.train_and_track_stats(args, nb_epochs, trainloader,testloader, net,optimizer,criterion,error_criterion, stats_collector)
        ''' Test the Network on the test data '''
        print(f'train_loss_epoch={train_loss_epoch} \ntrain_error_epoch={train_error_epoch} \ntest_loss_epoch={test_loss_epoch} \ntest_error_epoch={test_error_epoch}')
    elif args.train_alg == 'pert':
        ''' batch sizes '''
        batch_size_train, batch_size_test = 50*10**3, 10*10**3
        ''' number of repetitions '''
        nb_perturbation_trials = nb_epochs
        ''' noise level '''
        nb_layers = len(list(net.parameters()))
        noise_level = args.noise_level
        perturbation_magnitudes = nb_layers*[noise_level]
        print(f'noise_level={noise_level}')
        ''' locate where to save it '''
        folder_name_noise = f'noise_{perturbation_magnitudes[0]}'
        expt_path = os.path.join(expt_path,folder_name_noise)
        utils.make_and_check_dir(expt_path)
        matlab_file_name = f'noise_{perturbation_magnitudes}_{matlab_file_name}'
        ## TODO collect by perburbing current model X number of times with current perturbation_magnitudes
        use_w_norm2 = args.not_pert_w_norm2
        train_loss,train_error,test_loss,test_error = get_errors_for_all_perturbations(net,perturbation_magnitudes,use_w_norm2,args.enable_cuda,nb_perturbation_trials,stats_collector,criterion,error_criterion,trainloader,testloader)
        print(f'noise_level={noise_level},train_loss,train_error,test_loss,test_error={train_loss},{train_error},{test_loss},{test_error}')
    seconds,minutes,hours = utils.report_times(start_time)
    other_stats = dict({'seconds':seconds,'minutes':minutes,'hours':hours}, **other_stats)
    print(f'nb_epochs = {nb_epochs}')
    print(f'Finished Training, hours={hours}')
    print(f'seed = {seed}, githash = {githash}')
    ''' save results from experiment '''
    matlab_path_to_filename = os.path.join(expt_path,matlab_file_name)
    save2matlab.save2matlab_flatness_expt(matlab_path_to_filename, stats_collector,other_stats=other_stats)
    ''' save net model '''
    net_path_to_filename = os.path.join(expt_path,net_file_name)
    utils.save_entire_mdl(net_path_to_filename,net)
    # restored_net = utils.restore_entire_mdl(path)
    # loss_restored,error_restored = tr_alg.evalaute_mdl_data_set(criterion,error_criterion,restored_net,testloader,args.enable_cuda)
    #print(f'\nloss_restored={loss_restored},error_restored={error_restored}\a')
    ''' plot '''
    if plot:
        #TODO
        plot_utils.plot_loss_and_accuracies(stats_collector)
        plt.show()

if __name__ == '__main__':
    main(plot=True)
    print('\a')
