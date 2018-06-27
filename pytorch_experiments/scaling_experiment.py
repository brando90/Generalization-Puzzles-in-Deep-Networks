#!/usr/bin/env python
#SBATCH --mem=30000
#SBATCH --time=1-22:30
#SBATCH --mail-type=END
#SBATCH --mail-user=brando90@mit.edu
#SBATCH --array=1-1
#SBATCH --gres=gpu:1

import sys
import os

current_directory = os.getcwd() #The method getcwd() returns current working directory of a process.
sys.path.append(current_directory)

import torch
import numpy as np
from math import inf

import scipy

import scipy.io as sio

import data_classification as data_class
from new_training_algorithms import evalaute_mdl_on_full_data_set
import metrics
from good_minima_discriminator import divide_params_by
#from good_minima_discriminator import divide_params_by_taking_bias_into_account

from maps import NamedDict

import matplotlib as mpl
import matplotlib.pyplot as plt

from pdb import set_trace as st

def get_corruption_label( path_to_experiment ):
    '''
    extract corrpution label from path to experiments
    e.g. flatness_June_label_corrupt_prob_0.0_exptlabel_NL_only_1st_layer_BIAS_True_batch_size_train_1024_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0
    '''
    corrupt_prob = path_to_experiment.split('corrupt_prob_')[1].split('_exptlabel_NL')
    return float(corrupt_prob)

class Normalizer:

    def __init__(self,data_path,normalization_scheme,num_workers=10,label_corrupt_prob=0.0,batch_size_train=1024,batch_size_test=1024,standardize=True,iterations=inf):
        '''
        :param standardize: x - mu / std , [-1,+1]
        :return:
        '''
        self.normalize = normalization_scheme
        ''' '''
        self.error = metrics.error_criterion
        self.loss = torch.nn.CrossEntropyLoss()
        self.iterations = iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainset, self.trainloader, self.testset, self.testloader, self.classes_data = data_class.get_cifer_data_processors(data_path,batch_size_train,batch_size_test,num_workers,label_corrupt_prob,standardize=standardize)
        ''' dta we are collecting '''
        ## normalized
        self.train_all_losses_normalized = []
        self.train_all_errors_normalized = []
        self.test_all_losses_normalized = []
        self.gen_all_errors_normalized = []
        ## unnormalized
        self.train_all_losses_unnormalized = []
        self.train_all_errors_unnormalized = []
        self.test_all_losses_unnormalized = []
        self.gen_all_errors_unnormalized = []
        ##
        self.epoch_all_numbers = []
        self.corruption_all_probs = []

    def extract_all_results_vs_test_errors(self,path_all_expts,list_names,target_loss):
        '''
        extracts all the results for each experiment and updates an internal data structure of the results.

        :param path_all_expts: main path to all experiments
        :param list_names: the list of each experiment
        :param target_loss: target loss to halt at
        :return:
        '''
        for name in list_names:
            path_to_folder_expts = os.path.join(path_all_expts,name)
            print(f'path_to_folder_expts={path_to_folder_expts}')
            results = self.extract_results_with_target_loss(path_to_folder_expts, target_loss)
            ''' extend results ''' #
            self.collect_all(results) # adds all errors to internal lists
            ##
            self.epoch_all_numbers.extend(results.epoch_numbers)
            self.corruption_all_probs.extend(results.corruption_probs)
        return self.return_results()

    def extract_results_with_target_loss(self,path_to_folder_expts,target_loss):
        '''
        extracts specific results of the current experiment, given a specific train loss.

        :param path_to_folder_expts:
        :param target_loss:
        :return:
        '''
        ##
        train_losses_norm, train_errors_norm = [], []
        test_losses_norm, test_errors_norm = [], []
        ##
        train_losses_unnorm, train_errors_unnorm = [], []
        test_losses_unnorm, test_errors_unnorm = [], []
        ##
        epoch_numbers = []
        corruption_probs = []
        ''' go through results and get the ones with specific target loss '''
        matlab_filenames = [filename for filename in os.listdir(path_to_folder_expts) if '.mat' in filename]
        for matlab_filename in matlab_filenames: # essentially looping through all the nets that were trained
            matlab_path = os.path.join(path_to_folder_expts,matlab_filename)
            mat_contents = sio.loadmat(matlab_path)
            ''' '''
            epoch,seed_id,actual_train_loss = self.match_train_error(target_loss, mat_contents)
            #epoch, seed_id, actual_train_loss = self.final_train_error(mat_contents)
            if seed_id != -1: # if matched train error actually matched something
                normalized_results, unnormalized_results = self.get_results_from_normalized_net(epoch-1,seed_id, path_to_folder_expts) # not ethe -1 is cuz files where labeled with 0 as the first epoch and after that it ends at 299 which is the last one but train errors had 0th mean the virgin net
                train_loss_norm, train_error_norm, test_loss_norm, test_error_norm = normalized_results
                train_loss_un, train_error_un, test_loss_un, test_error_un = unnormalized_results
                ''' '''
                corruption_prob = self.get_corruption_prob(path_to_folder_expts)
                ''' append results '''
                train_losses_norm.append(train_loss_norm), train_errors_norm.append(train_error_norm)
                test_losses_norm.append(test_loss_norm), test_errors_norm.append(test_error_norm)
                ##
                train_losses_unnorm.append(train_loss_un), train_errors_unnorm.append(train_error_un)
                test_losses_unnorm.append(test_loss_un), test_errors_unnorm.append(test_error_un)
                ##
                epoch_numbers.append(epoch)
                ##
                corruption_probs.append(corruption_prob)
        ''' organize/collect results'''
        results = NamedDict(train_losses_norm=train_losses_norm,train_errors_norm=train_errors_norm,
                            test_losses_norm=test_losses_norm, test_errors_norm=test_errors_norm,
                            train_losses_unnorm=train_losses_unnorm, train_errors_unnorm=train_errors_unnorm,
                            test_losses_unnorm=test_losses_unnorm, test_errors_unnorm=test_errors_unnorm,
                            epoch_numbers=epoch_numbers,corruption_probs=corruption_probs)
        return results

    def match_train_error(self,target_loss, mat_contents):
        '''
        gets the closest loss to the target loss.

        :param target_loss:
        :param mat_contents:
        :return:
        '''
        train_losses = mat_contents['train_losses'][0]
        train_errors = mat_contents['train_errors'][0]
        ''' look for the closest trian loss to the target '''
        differences_from_target_loss = []
        list_epochs,list_seeds,list_losses = [], [], []
        for epoch in range(len(train_losses)):
            train_loss = train_losses[epoch]
            differences_from_target = abs(train_loss - target_loss)
            if differences_from_target < 0.0001:
                train_error = train_errors[epoch]
                if train_error == 0.0:
                    seed = mat_contents['seed'][0][0]
                    ''' append relevant results '''
                    list_losses.append(train_loss)
                    differences_from_target_loss.append(differences_from_target)
                    list_epochs.append(epoch)
                    list_seeds.append(seed)
        if len(differences_from_target_loss) == 0: # if we collected no results
            epoch, seed_id, actual_train_loss = -1,-1,train_loss
        else: # extract the loss with the smallest difference
            index_smallest = np.argmin(differences_from_target_loss)
            epoch, seed_id, actual_train_loss = list_epochs[index_smallest],list_seeds[index_smallest],list_losses[index_smallest]
        return epoch, seed_id, actual_train_loss

    def final_train_error(self,mat_contents):
        '''
        gets the final train error

        :param mat_contents:
        :return:
        '''
        train_losses = mat_contents['train_losses'][0]
        train_errors = mat_contents['train_errors'][0]
        ''' look for the closest trian loss to the target '''
        differences_from_target_loss = []
        list_epochs,list_seeds,list_losses = [], [], []
        ''' '''
        epoch = len(train_errors)-1
        train_error = train_errors[epoch]
        train_loss = train_losses[epoch]
        if train_error == 0.0:
            differences_from_target = 0
            seed = mat_contents['seed'][0][0]
            ''' append relevant results '''
            list_losses.append(train_loss)
            differences_from_target_loss.append(differences_from_target)
            list_epochs.append(epoch)
            list_seeds.append(seed)
        if len(differences_from_target_loss) == 0: # if we collected no results
            epoch, seed_id, actual_train_loss = -1,-1,train_loss
        else: # extract the loss with the smallest difference
            index_smallest = np.argmin(differences_from_target_loss)
            epoch, seed_id, actual_train_loss = list_epochs[index_smallest],list_seeds[index_smallest],list_losses[index_smallest]
        return epoch, seed_id, actual_train_loss

    def get_results_from_normalized_net(self,epoch,seed_id, path_to_folder_expts):
        ''' '''
        ''' get net '''
        nets_folders = [filename for filename in os.listdir(path_to_folder_expts) if 'nets_folder' in filename]
        net_folder = [filename for filename in nets_folders if f'seed_{seed_id}' in filename][0] # note seed are unique very h.p.
        net_path = os.path.join(path_to_folder_expts,net_folder)
        if len([net_name for net_name in os.listdir(net_path) if f'epoch_{epoch}' in net_name]) == 0:
            st()
        net_name = [net_name for net_name in os.listdir(net_path) if f'epoch_{epoch}' in net_name][0]
        net_path = os.path.join(net_path, net_name)
        net = torch.load(net_path)
        ''' get unormalized test error '''
        train_loss_un, train_error_un = evalaute_mdl_on_full_data_set(self.loss, self.error, net, self.trainloader, self.device)
        test_loss_un, test_error_un = evalaute_mdl_on_full_data_set(self.loss, self.error, net, self.testloader, self.device)
        ''' normalize net '''
        net = self.normalize(net)
        ''' get normalized train errors '''
        train_loss_norm, train_error_norm = evalaute_mdl_on_full_data_set(self.loss, self.error, net, self.trainloader, self.device)
        test_loss_norm, test_error_norm = evalaute_mdl_on_full_data_set(self.loss, self.error, net, self.testloader, self.device)
        return (train_loss_norm, train_error_norm,test_loss_norm, test_error_norm),(train_loss_un,train_error_un,test_loss_un,test_error_un),

    def normalize_net(self,net):
        return self.normalization_scheme(net)

    def collect_all(self,results):
        ## normalized
        self.train_all_losses_normalized.extend(results.train_losses_norm)
        self.train_all_errors_normalized.extend(results.train_errors_norm)
        self.test_all_losses_normalized.extend(results.test_losses_norm)
        self.gen_all_errors_normalized.extend(results.test_errors_norm)
        ## unnormalized
        self.train_all_losses_unnormalized.extend(results.train_losses_unnorm)
        self.train_all_errors_unnormalized.extend(results.train_errors_unnorm)
        self.test_all_losses_unnormalized.extend(results.test_losses_unnorm)
        self.gen_all_errors_unnormalized.extend(results.test_errors_unnorm)

    def get_corruption_prob(self,name):
        '''
        extracts the corruption probability from the name of the path. Note that the information we care about
        is about the corruption probability in the *initialization* (since its trained on NL). Thus, we need
        to extract the corruption information from the RLNL part.
        :param name:
        :return:
        '''
        if 'RLNL' in name:
            # string it expects:
            # '/cbcl/cbcl01/brando90/home_simulation_research/overparametrized_experiments/pytorch_experiments/test_runs_flatness4/flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.0001_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0'
            corruption_prob = float( name.split('RLNL_')[1].split('_only_1st')[0] )
        else:
            corruption_prob = 0.0
        return corruption_prob

    def return_results(self):
        attributes = [attribute for attribute in dir(self) if not attribute.startswith('__') and not callable(getattr(self, attribute))]
        results = {attribute:getattr(self,attribute) for attribute in attributes if 'normalized' in attribute}
        results = dict({'epoch_all_numbers':self.epoch_all_numbers,'corruption_all_probs':self.corruption_all_probs}, **results)
        return NamedDict(results)

    def return_attributes(self):
        attributes = [attribute for attribute in dir(self) if not attribute.startswith('__') and not callable(getattr(self, attribute))]
        results = {attribute:getattr(self,attribute) for attribute in attributes}
        return NamedDict(results)

####

def divide_params(net,norm_func):
    '''
    net: network
    norm_func: function that returns the norm of W depending to the specified scheme.

    normalizes the network per layer.
    '''
    conv0_w = None
    ##
    params = net.named_parameters()
    dict_params = dict(params)
    for name, param in dict_params.items():
        if name in dict_params:
            if name == 'conv0.weight':
                conv0_w = param
            elif name == 'conv0.bias':
                conv0_b = param
                ##
                w_norm = norm_func(conv0_w)
                b_norm = norm_func(conv0_b)
                if norm_func.__name__ == 'frobenius_normalization':
                    W_norm = (w_norm**2+b_norm**2)**0.5
                else:
                    W_norm = w_norm + b_norm
                ## normalize W
                new_param = conv0_w / W_norm
                dict_params['conv0.weight'] = new_param
                ## normalize b
                new_param = conv0_b / W_norm
                dict_params[name] = new_param
            else:
                W_norm = norm_func(param)
                new_param = param/W_norm
                dict_params[name] = new_param
    net.load_state_dict(dict_params)
    return net

def frobenius_normalization(W):
    '''
    return net/||net||_2
    '''
    return W.norm(2)

def l1_normalization(W):
    '''
    return net/||net||_1
    '''
    return W.norm(1)

def spectral_normalization(W):
    '''
    return net/||net||_spectral
    '''
    eigenvalues = torch.eig(W.mm(W))
    spectral_norm = torch.max(eigenvalues)**0.5
    return spectral_norm

def main():
    # TODO: IMPORTANT: Don't forget to include biases in the [W, b]
    print('start main')
    #path_all_expts = '/cbcl/cbcl01/brando90/home_simulation_research/overparametrized_experiments/pytorch_experiments/test_runs_flatness4'
    path_all_expts = '/cbcl/cbcl01/brando90/home_simulation_research/overparametrized_experiments/pytorch_experiments/test_runs_flatness5_ProperOriginalExpt'
    ''' expt_paths '''
    list_names = []
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_NL_only_1st_layer_BIAS_True_batch_size_train_1024_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.0001_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.001_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.01_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.1_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.2_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.5_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.75_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_1.0_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    ''' normalization scheme '''
    norm = 'frobenius'
    normalization_scheme = lambda net: divide_params(net,frobenius_normalization)
    #norm = 'l1'
    #normalization_scheme = lambda net: divide_params(net, l1_normalization)
    #normalization_scheme = lambda net: divide_params(net, spectral_normalization)
    ''' get results'''
    data_path = './data'
    target_loss = 0.0044
    normalizer = Normalizer(data_path,normalization_scheme)
    results = normalizer.extract_all_results_vs_test_errors(path_all_expts,list_names,target_loss)
    ''' '''
    path = os.path.join(path_all_expts, f'loss_vs_gen_errors_norm_{norm}')
    #path = os.path.join(path_all_expts,f'loss_vs_gen_errors_norm_{norm}_final')
    scipy.io.savemat(path, results)
    ''' plot '''
    #plt.scatter(train_all_losses,gen_all_errors)
    #plt.show()
    ''' cry '''
    print('\a')

if __name__ == '__main__':
    main()
    print('end')
