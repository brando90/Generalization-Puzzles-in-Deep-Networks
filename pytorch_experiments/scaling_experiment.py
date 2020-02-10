#!/usr/bin/env python
#SBATCH --mem=30000
#SBATCH --time=1-22:30
#SBATCH --mail-type=END
#SBATCH --mail-user=brando90@mit.edu
#SBATCH --gres=gpu:1

import sys
import os
import time

current_directory = os.getcwd() #The method getcwd() returns current working directory of a process.
sys.path.append(current_directory)

import torch
import numpy as np
from math import inf

import scipy

import scipy.io as sio

import data_classification as data_class
from new_training_algorithms import evalaute_mdl_on_full_data_set
from new_training_algorithms import collect_hist
import metrics
import utils
import list_experiments as lists
from good_minima_discriminator import divide_params_by
#from good_minima_discriminator import divide_params_by_taking_bias_into_account

from maps import NamedDict

import matplotlib as mpl
import matplotlib.pyplot as plt

import nn_models as nn_mdls
import math
from new_training_algorithms import get_function_evaluation_from_name

from pdb import set_trace as st

def get_corruption_label( path_to_experiment ):
    '''
    extract corrpution label from path to experiments
    e.g. flatness_June_label_corrupt_prob_0.0_exptlabel_NL_only_1st_layer_BIAS_True_batch_size_train_1024_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0
    '''
    corrupt_prob = path_to_experiment.split('corrupt_prob_')[1].split('_exptlabel_NL')
    return float(corrupt_prob)

class Normalizer:

    def __init__(self,list_names,data_path,normalization_scheme,p,division_constant,data_set,num_workers=10,batch_size_train=1024,batch_size_test=1024,standardize=True,iterations=inf,type_standardize='default'):
        '''
        :param standardize: x - mu / std , [-1,+1]
        :return:

        IMPORTANT: all results must have the string "all" to be returned correctly as a new result
        '''
        self.list_names = list_names # the list of each experiment
        ''' '''
        self.p = p
        self.division_constant = division_constant
        ''' '''
        self.normalization_scheme = normalization_scheme
        ''' '''
        self.data_set = data_set
        self.error = metrics.error_criterion
        self.loss = torch.nn.CrossEntropyLoss()
        self.iterations = iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ''' data loaders '''
        self.loaders = {} # {corruption:(trainloader,testloader)}
        for i,name in enumerate(self.list_names):
            corruption_prob = self.get_corruption_prob(name)
            if corruption_prob not in self.loaders:
                trainset, testset, classes_data = data_class.get_data_processors(data_path,corruption_prob,data_set,standardize=standardize,type_standardize=type_standardize)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=False, num_workers=num_workers)
                testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=True, num_workers=num_workers)
                ''' '''
                self.loaders[corruption_prob] = (trainloader,testloader)
                self.classes_data = classes_data
        ''' data we are collecting '''
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
        self.std_inits_all = []
        ##
        self.w_norms_all = [ ]
        ''' '''
        self.hist_all_train_norm = []
        self.hist_all_test_norm = []
        self.hist_all_train_un = []
        self.hist_all_test_un = []

    def extract_all_results_vs_test_errors(self,path_all_expts,target_loss):
        '''
        extracts all the results for each experiment and updates an internal data structure of the results.

        :param path_all_expts: main path to all experiments
        :param target_loss: target loss to halt at
        :return:
        '''
        for name in self.list_names:
            path_to_folder_expts = os.path.join(path_all_expts,name)
            print()
            print(f'path_to_folder_expts={path_to_folder_expts}')
            #results = self.extract_results_with_target_loss(path_to_folder_expts, target_loss)
            results = self.extract_results_final_model(path_to_folder_expts)
            ''' extend results ''' #
            self.collect_all(results) # adds all errors to internal lists
        return self.return_results()

    def extract_results_with_target_loss(self,path_to_folder_expts,target_loss):
        '''
        extracts specific results of the current experiment, given a specific train loss.

        :param path_to_folder_expts:
        :param target_loss:
        :return:
        '''
        ####
        train_losses_norm, train_errors_norm = [], []
        test_losses_norm, test_errors_norm = [], []
        #
        train_losses_unnorm, train_errors_unnorm = [], []
        test_losses_unnorm, test_errors_unnorm = [], []
        ####
        train_losses_norm_rand, train_errors_norm_rand = [], []
        test_losses_norm_rand, test_errors_norm_rand = [], []
        #
        train_losses_unnorm_rand, train_errors_unnorm_rand = [], []
        test_losses_unnorm_rand, test_errors_unnorm_rand = [], []
        ##
        epoch_numbers = []
        corruption_probs = []
        ''' go through results and get the ones with specific target loss '''
        matlab_filenames = [filename for filename in os.listdir(path_to_folder_expts) if '.mat' in filename]
        for matlab_filename in matlab_filenames: # essentially looping through all the nets that were trained
            matlab_path = os.path.join(path_to_folder_expts,matlab_filename)
            mat_contents = sio.loadmat(matlab_path)
            ''' '''
            #epoch,seed_id,actual_train_loss = self.match_zero_train_error(mat_contents)
            #epoch, seed_id, actual_train_loss = self.match_train_loss(target_loss, mat_contents)
            epoch, seed_id, actual_train_loss = self.final_train_error(mat_contents)
            if seed_id != -1: # if matched train error actually matched something
                #normalized_results, unnormalized_results = self.get_results_from_normalized_net(epoch-1,seed_id, path_to_folder_expts) # not ethe -1 is cuz files where labeled with 0 as the first epoch and after that it ends at 299 which is the last one but train errors had 0th mean the virgin net
                normalized_results, unnormalized_results, normalized_results_rand, unnormalized_results_rand = self.get_results_from_normalized_net(epoch-1,seed_id, path_to_folder_expts)
                ## extract natural labels results
                train_loss_norm, train_error_norm, test_loss_norm, test_error_norm = normalized_results
                train_loss_un, train_error_un, test_loss_un, test_error_un = unnormalized_results
                ## extract random labels results
                train_loss_norm_rand, train_error_norm_rand, test_loss_norm_rand, test_error_norm_rand = normalized_results_rand
                train_loss_un_rand, train_error_un_rand, test_loss_un_rand, test_error_un_rand = unnormalized_results_rand
                ''' '''
                corruption_prob = self.get_corruption_prob(path_to_folder_expts)
                ''' append results '''
                #### natural label
                train_losses_norm.append(train_loss_norm), train_errors_norm.append(train_error_norm)
                test_losses_norm.append(test_loss_norm), test_errors_norm.append(test_error_norm)
                #
                train_losses_unnorm.append(train_loss_un), train_errors_unnorm.append(train_error_un)
                test_losses_unnorm.append(test_loss_un), test_errors_unnorm.append(test_error_un)
                #### random label
                train_losses_norm_rand.append(train_loss_norm_rand), train_errors_norm_rand.append(train_error_norm_rand)
                test_losses_norm_rand.append(test_loss_norm_rand), test_errors_norm_rand.append(test_error_norm_rand)
                #
                train_losses_unnorm_rand.append(train_loss_un_rand), train_errors_unnorm_rand.append(train_error_un_rand)
                test_losses_unnorm_rand.append(test_loss_un_rand), test_errors_unnorm_rand.append(test_error_un_rand)
                ##
                epoch_numbers.append(epoch)
                ##
                corruption_probs.append(corruption_prob)
        ''' organize/collect results'''
        results = NamedDict(train_losses_norm=train_losses_norm,train_errors_norm=train_errors_norm,
                            test_losses_norm=test_losses_norm, test_errors_norm=test_errors_norm,
                            train_losses_unnorm=train_losses_unnorm, train_errors_unnorm=train_errors_unnorm,
                            test_losses_unnorm=test_losses_unnorm, test_errors_unnorm=test_errors_unnorm,
                            train_losses_norm_rand=train_losses_norm_rand, train_errors_norm_rand=train_errors_norm_rand,
                            test_losses_norm_rand=test_losses_norm_rand, test_errors_norm_rand=test_errors_norm_rand,
                            train_losses_unnorm_rand=train_losses_unnorm_rand, train_errors_unnorm_rand=train_errors_unnorm_rand,
                            test_losses_unnorm_rand=test_losses_unnorm_rand, test_errors_unnorm_rand=test_errors_unnorm_rand,
                            epoch_numbers=epoch_numbers,corruption_probs=corruption_probs)
        return results

    def match_train_loss(self,target_loss, mat_contents):
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

    def match_zero_train_error(self, mat_contents):
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
            train_error = train_errors[epoch]
            if train_error == 0.0:
                seed = mat_contents['seed'][0][0]
                ''' append relevant results '''
                list_losses.append(train_loss)
                differences_from_target_loss.append(0)
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
        ## random labels
        train_loss_un_rand, train_error_un_rand = evalaute_mdl_on_full_data_set(self.loss, self.error, net, self.trainloader_rand, self.device)
        test_loss_un_rand, test_error_un_rand = evalaute_mdl_on_full_data_set(self.loss, self.error, net, self.testloader_rand, self.device)
        ''' normalize net '''
        net = self.normalize(net)
        ''' get normalized train errors '''
        ## natural labels
        train_loss_norm, train_error_norm = evalaute_mdl_on_full_data_set(self.loss, self.error, net, self.trainloader, self.device)
        test_loss_norm, test_error_norm = evalaute_mdl_on_full_data_set(self.loss, self.error, net, self.testloader, self.device)
        ## random labels
        train_loss_norm_rand, train_error_norm_rand = evalaute_mdl_on_full_data_set(self.loss, self.error, net, self.trainloader_rand, self.device)
        test_loss_norm_rand, test_error_norm_rand = evalaute_mdl_on_full_data_set(self.loss, self.error, net, self.testloader_rand, self.device)
        ''' pack results '''
        normalized_results = (train_loss_norm, train_error_norm, test_loss_norm, test_error_norm)
        unnormalized_results = (train_loss_un, train_error_un, test_loss_un, test_error_un)
        ##
        normalized_results_rand = (train_loss_norm_rand, train_error_norm_rand, test_loss_norm_rand, test_error_norm_rand)
        unnormalized_results_rand = (train_loss_un_rand, train_error_un_rand, test_loss_un_rand, test_error_un_rand)
        ''' return '''
        return normalized_results, unnormalized_results, normalized_results_rand, unnormalized_results_rand

    def extract_results_final_model(self, path_to_folder_expts):
        '''
            extracts specific results of the current experiment, given a specific train loss.

            :param path_to_folder_expts:
            :param target_loss:
            :return:
        '''
        ## normalized
        train_losses_norm, train_errors_norm = [], []
        test_losses_norm, test_errors_norm = [], []
        ## unnormalized
        train_losses_unnorm, train_errors_unnorm = [], []
        test_losses_unnorm, test_errors_unnorm = [], []
        ## other stats
        epoch_numbers = []
        corruption_probs = []
        stds_inits = []
        '''  get un/normalized net results for all experiments '''
        print(f'os.listdir(path_to_folder_expts) = {os.listdir(path_to_folder_expts)}')
        net_filenames = [filename for filename in os.listdir(path_to_folder_expts) if 'net_' in filename]
        matlab_filenames = [filename for filename in os.listdir(path_to_folder_expts) if '.mat' in filename]
        nb_zero_train_error = 0
        for j,net_filename in enumerate(net_filenames):  # looping through all the nets that were trained
            print('------- part of the loop -------')
            print(f'>jth NET = {j}')
            print(f'>path_to_folder_expts = {path_to_folder_expts}')
            print(f'>net_filename = {net_filename}')
            ''' get matlab file '''
            seed = net_filename.split('seed_')[1].split('_')[0]
            matlab_filename = [filename for filename in matlab_filenames if seed in filename][0]
            matlab_path = os.path.join(path_to_folder_expts, matlab_filename)
            mat_contents = sio.loadmat(matlab_path)
            ''' get results of normalized net if train_error == 0 '''
            train_errors = mat_contents['train_errors'][0]
            corruption_prob = self.get_corruption_prob(path_to_folder_expts)
            print(f'>train_errors final epoch = {train_errors[-1]} ')
            print(f'---> corruption_prob={corruption_prob}')
            if train_errors[-1] == 0:
                nb_zero_train_error += 1
                std = mat_contents['stds'][0][0]
                corruption_prob = self.get_corruption_prob(path_to_folder_expts)
                epoch = len(train_errors)
                ''' get results from normalized net'''
                results = self.get_results_of_net(net_filename,path_to_folder_expts,corruption_prob)
                #results = self.get_results_of_net_divided_by_product_norm(net_filename, path_to_folder_expts, corruption_prob)
                ## extract results
                normalized_results, unnormalized_results = results
                train_loss_norm, train_error_norm, test_loss_norm, test_error_norm = normalized_results
                train_loss_un, train_error_un, test_loss_un, test_error_un = unnormalized_results
                print(f'>normalized_results = {normalized_results}')
                print(f'>unnormalized_results = {unnormalized_results}')
                ''' catch error if trian performance dont match'''
                if train_error_norm != 0 or train_error_un != 0:
                    print()
                    print(f'---> ERROR: train_error_norm != 0 or train_error_un != 0 values are train_error_norm={train_error_norm},train_error_un={train_error_un} they should be zero.')
                    print(f'path_to_folder_expts = {path_to_folder_expts}\nnet_filename = {net_filename}')
                    print(f'seed = {seed}\nmatlab_filename = {matlab_filename}')
                    st()
                ''' append results '''
                ## normalized
                train_losses_norm.append(train_loss_norm), train_errors_norm.append(train_error_norm)
                test_losses_norm.append(test_loss_norm), test_errors_norm.append(test_error_norm)
                ## unnormalized
                train_losses_unnorm.append(train_loss_un), train_errors_unnorm.append(train_error_un)
                test_losses_unnorm.append(test_loss_un), test_errors_unnorm.append(test_error_un)
                ''' append stats'''
                epoch_numbers.append(epoch)
                corruption_probs.append(corruption_prob)
                stds_inits.append(std)
        print(f'-------------> # of nets trained = {len(net_filenames)}')
        print(f'-------------> nb_zero_train_error = {nb_zero_train_error}')
        print(f'-------------> frac zero train error = {nb_zero_train_error}/{len(net_filenames)} = {nb_zero_train_error/len(net_filenames)}')
        ''' organize/collect results'''
        ## IMPORTANT: adding things to this list is not enough to return it to matlab, also edit collect_all
        ## the field name needs the string all as part of its name for it to be saved
        results = NamedDict(train_losses_norm=train_losses_norm, train_errors_norm=train_errors_norm,
                            test_losses_norm=test_losses_norm, test_errors_norm=test_errors_norm,
                            train_losses_unnorm=train_losses_unnorm, train_errors_unnorm=train_errors_unnorm,
                            test_losses_unnorm=test_losses_unnorm, test_errors_unnorm=test_errors_unnorm,
                            epoch_numbers=epoch_numbers, corruption_probs=corruption_probs,stds_inits=stds_inits)
        return results

    def get_results_of_net(self,net_filename,path_to_folder_expts,corruption_prob):
        ''' '''
        ''' '''
        trainloader = self.loaders[corruption_prob][0]
        testloader = self.loaders[corruption_prob][1]
        ''' '''
        net_path = os.path.join(path_to_folder_expts,net_filename)
        net = torch.load(net_path)
        ''' get unormalized test error '''
        train_loss_un, train_error_un = evalaute_mdl_on_full_data_set(self.loss, self.error, net, trainloader, self.device)
        test_loss_un, test_error_un = evalaute_mdl_on_full_data_set(self.loss, self.error, net, testloader, self.device)
        ''' normalize net '''
        net = self.normalize(net)
        ''' get normalized train errors '''
        train_loss_norm, train_error_norm = evalaute_mdl_on_full_data_set(self.loss, self.error, net, trainloader, self.device)
        test_loss_norm, test_error_norm = evalaute_mdl_on_full_data_set(self.loss, self.error, net, testloader, self.device)
        ''' pack results '''
        normalized_results = (train_loss_norm, train_error_norm, test_loss_norm, test_error_norm)
        unnormalized_results = (train_loss_un, train_error_un, test_loss_un, test_error_un)
        ''' return '''
        return normalized_results, unnormalized_results

    def get_results_of_net_divided_by_product_norm(self,net_filename,path_to_folder_expts,corruption_prob):
        ''' '''
        trainloader = self.loaders[corruption_prob][0]
        testloader = self.loaders[corruption_prob][1]
        ''' '''
        net_path = os.path.join(path_to_folder_expts,net_filename)
        net = torch.load(net_path)
        ''' get unormalized test error '''
        train_loss_un, train_error_un = evalaute_mdl_on_full_data_set(self.loss, self.error, net, trainloader, self.device)
        test_loss_un, test_error_un = evalaute_mdl_on_full_data_set(self.loss, self.error, net, testloader, self.device)
        ''' get product norm of net '''
        product_norm = self.normalization_scheme(net)
        ''' get normalized train errors '''
        train_loss_norm, train_error_norm = train_loss_un/product_norm ,train_error_un/product_norm
        test_loss_norm, test_error_norm = test_loss_un/product_norm, test_error_un/product_norm
        ''' pack results '''
        normalized_results = (train_loss_norm, train_error_norm, test_loss_norm, test_error_norm)
        unnormalized_results = (train_loss_un, train_error_un, test_loss_un, test_error_un)
        ''' return '''
        return normalized_results, unnormalized_results

    def normalize(self,net):
        '''

            Note: for each time this function is called, it appends the stats once. If it goes through each list and
            append each time it means it extends the list each time it's called. If this function is called every time
            we normalize a net, then it means that we are adding stats for every time a specific corruption of a net is
            present. So each index corresponds to some corruption level on the corruption array that we are collecting.
        '''
        if len(self.w_norms_all) == 0:
            nb_param_groups = len(list(net.parameters()))
            self.w_norms_all = [[] for i in range(nb_param_groups)]
        for index, W in enumerate(net.parameters()):
            self.w_norms_all[index].append( W.norm(self.p).item() )
        return self.normalization_scheme(net)

    def collect_all(self,results):
        ''' '''
        ''' Natural Labels '''
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
        ''' other stats '''
        self.epoch_all_numbers.extend(results.epoch_numbers)
        self.corruption_all_probs.extend(results.corruption_probs)
        self.std_inits_all.extend(results.stds_inits)
        ''' '''
        if 'hist_train_norm' in results:
            print('SAVED HISTOGRAMS')
            ## TODO fix, add for loop over all mdls
            self.hist_all_train_norm.extend(results.hist_train_norm)
            self.hist_all_test_norm.extend(results.hist_test_norm)
            self.hist_all_train_un.extend(results.hist_train_un)
            self.hist_all_test_un.extend(results.hist_test_un)

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
            corruption_prob = float(name.split('corrupt_prob_')[1].split('_exptlabel')[0])
        return corruption_prob

    def return_results(self):
        ''' '''
        ''' get list of attributes'''
        attributes = [attribute for attribute in dir(self) if not attribute.startswith('__') and not callable(getattr(self, attribute))]
        ''' get results '''
        ## all results must have the string "all"
        results = {attribute:getattr(self,attribute) for attribute in attributes if 'all' in attribute}
        return NamedDict(results)

    def return_attributes(self):
        attributes = [attribute for attribute in dir(self) if not attribute.startswith('__') and not callable(getattr(self, attribute))]
        results = {attribute:getattr(self,attribute) for attribute in attributes}
        return NamedDict(results)

    def get_hist_from_single_net(self,path_all_expts):
        '''
        '''
        for current_expt_name in self.list_names:
            path_to_folder_expts = os.path.join(path_all_expts,current_expt_name)
            print(f'path_all_expts = {path_all_expts}')
            print(f'name = {current_expt_name}')
            print(f'path_to_folder_expts={path_to_folder_expts}')
            results = self.extract_hist(path_to_folder_expts)
            ''' extend results ''' #
            if results != -1:
                self.collect_all(results) # adds all errors to internal lists
        return self.return_results()

    def extract_hist(self, path_to_folder_expts):
        '''
            extracts a single hist sample

            :param path_to_folder_expts:
        '''
        ## normalized
        train_losses_norm, train_errors_norm = [], []
        test_losses_norm, test_errors_norm = [], []
        hist_train_norm = []
        hist_test_norm = []
        ## unnormalized
        train_losses_unnorm, train_errors_unnorm = [], []
        test_losses_unnorm, test_errors_unnorm = [], []
        hist_train_un = []
        hist_test_un = []
        ## other stats
        epoch_numbers = []
        corruption_probs = []
        stds_inits = []
        '''  get un/normalized net results for all experiments '''
        print(f'os.listdir(path_to_folder_expts) = {os.listdir(path_to_folder_expts)}')
        net_filenames = [filename for filename in os.listdir(path_to_folder_expts) if 'net_' in filename]
        matlab_filenames = [filename for filename in os.listdir(path_to_folder_expts) if '.mat' in filename]
        nb_zero_train_error = 0
        for j,net_filename in enumerate(net_filenames): # looping through all the nets that were trained
            print('------- part of the loop -------')
            print(f'>jth NET = {j}')
            print(f'>path_to_folder_expts = {path_to_folder_expts}')
            print(f'>net_filename = {net_filename}')
            ''' get matlab file '''
            seed = net_filename.split('seed_')[1].split('_')[0]
            matlab_filename = [filename for filename in matlab_filenames if seed in filename][0]
            matlab_path = os.path.join(path_to_folder_expts, matlab_filename)
            mat_contents = sio.loadmat(matlab_path)
            ''' get results of normalized net if train_error == 0 '''
            train_errors = mat_contents['train_errors'][0]
            corruption_prob = self.get_corruption_prob(path_to_folder_expts)
            print(f'>train_errors final epoch = {train_errors[-1]} ')
            print(f'---> corruption_prob={corruption_prob}')
            if train_errors[-1] == 0:
                nb_zero_train_error += 1
                std = mat_contents['stds'][0][0]
                corruption_prob = self.get_corruption_prob(path_to_folder_expts)
                epoch = len(train_errors)
                ''' get results from normalized net'''
                hist_norm, hist_un = self.get_hist_last_layer_activations(net_filename,path_to_folder_expts,corruption_prob)
                results = self.get_results_of_net(net_filename,path_to_folder_expts,corruption_prob)
                ## extract results
                normalized_results, unnormalized_results = results
                train_loss_norm, train_error_norm, test_loss_norm, test_error_norm = normalized_results
                train_loss_un, train_error_un, test_loss_un, test_error_un = unnormalized_results
                print(f'>normalized_results = {normalized_results}')
                print(f'>unnormalized_results = {unnormalized_results}')
                ## extract histograms
                current_hist_train_norm, current_hist_test_norm = hist_norm
                current_hist_train_un, current_hist_test_un = hist_un
                ''' catch error if trian performance dont match'''
                if train_error_norm != 0 or train_error_un != 0:
                    print()
                    print(f'---> ERROR: train_error_norm != 0 or train_error_un != 0 values are train_error_norm={train_error_norm},train_error_un={train_error_un} they should be zero.')
                    print(f'path_to_folder_expts = {path_to_folder_expts}\nnet_filename = {net_filename}')
                    print(f'seed = {seed}\nmatlab_filename = {matlab_filename}')
                    st()
                ''' append results '''
                ## normalized
                train_losses_norm.append(train_loss_norm), train_errors_norm.append(train_error_norm)
                test_losses_norm.append(test_loss_norm), test_errors_norm.append(test_error_norm)
                hist_train_norm.append(current_hist_train_norm)
                hist_test_norm.append(current_hist_test_norm)
                ## unnormalized
                train_losses_unnorm.append(train_loss_un), train_errors_unnorm.append(train_error_un)
                test_losses_unnorm.append(test_loss_un), test_errors_unnorm.append(test_error_un)
                hist_train_un.append(current_hist_train_un)
                hist_test_un.append(current_hist_test_un)
                ''' append stats '''
                epoch_numbers.append(epoch)
                corruption_probs.append(corruption_prob)
                stds_inits.append(std)
        ''' '''
        print(f'-------------> # of nets trained = {len(net_filenames)}')
        print(f'-------------> nb_zero_train_error = {nb_zero_train_error}')
        print(f'-------------> frac zero train error = {nb_zero_train_error}/{len(net_filenames)} = {nb_zero_train_error/len(net_filenames)}')
        if nb_zero_train_error != 0:
            ''' organize/collect results'''
            ## IMPORTANT: adding things to this list is not enough to return it to matlab, also edit collect_all
            results = NamedDict(train_losses_norm=train_losses_norm, train_errors_norm=train_errors_norm,
                                test_losses_norm=test_losses_norm, test_errors_norm=test_errors_norm,
                                train_losses_unnorm=train_losses_unnorm, train_errors_unnorm=train_errors_unnorm,
                                test_losses_unnorm=test_losses_unnorm, test_errors_unnorm=test_errors_unnorm,
                                epoch_numbers=epoch_numbers, corruption_probs=corruption_probs,stds_inits=stds_inits,
                                hist_train_norm=hist_train_norm, hist_test_norm=hist_test_norm,
                                hist_train_un=hist_train_un, hist_test_un=hist_test_un)
            return results
        else:
            return -1

    def get_hist_last_layer_activations(self,net_filename,path_to_folder_expts,corruption_prob):
        ''' '''
        ''' '''
        trainloader = self.loaders[corruption_prob][0]
        testloader = self.loaders[corruption_prob][1]
        ''' '''
        net_path = os.path.join(path_to_folder_expts,net_filename)
        net = torch.load(net_path)
        ''' get unormalized test error '''
        hist_train_un = collect_hist(net, trainloader, self.device)
        hist_test_un = collect_hist(net, testloader, self.device)
        ''' normalize net '''
        net = self.normalize(net)
        ''' get normalized train errors '''
        hist_train_norm = collect_hist(net, trainloader, self.device)
        hist_test_norm = collect_hist(net, testloader, self.device)
        ''' pack results '''
        hist_norm = (hist_train_norm,hist_test_norm)
        hist_un = (hist_train_un,hist_test_un)
        ''' return '''
        return hist_norm, hist_un

####

def divide_params(net,norm_func):
    '''
    net: network
    norm_func: function that returns the norm of W depending to the specified scheme.

    normalizes the network per layer.
    '''
    p = norm_func.p
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
                ''' get W_norm = []'''
                w_norm = norm_func(conv0_w)
                b_norm = norm_func(conv0_b)
                ## W_norm = pth_root( ||w||^p + ||b||^p ), we need to compute first the sum of squares first basically, thats why we can't just add the norms
                W_norm = (w_norm ** p + b_norm ** p) ** (1.0/p)
                ''' normalize W and bias '''
                ## normalize W
                new_param = conv0_w / W_norm
                dict_params['conv0.weight'] = new_param
                ## normalize b
                new_param = conv0_b / W_norm
                dict_params['conv0.bias'] = new_param
            else:
                W_norm = norm_func(param)
                #print(f'W_norm = {W_norm}')
                new_param = param/W_norm
                dict_params[name] = new_param
    net.load_state_dict(dict_params)
    return net

def get_product_norm(net, norm_func):
    '''
    net: network
    norm_func: function that returns the norm of W depending to the specified scheme.

    normalizes the network per layer.
    '''
    p = norm_func.p
    conv0_w = None
    ##
    prod_W_norm = 1.0
    ##
    params = net.named_parameters()
    dict_params = dict(params)
    for name, param in dict_params.items():
        if name in dict_params:
            if name == 'conv0.weight':
                conv0_w = param
            elif name == 'conv0.bias':
                conv0_b = param
                w_norm = norm_func(conv0_w)
                b_norm = norm_func(conv0_b)
                ## W_norm = pth_root( ||w||^p + ||b||^p ), we need to compute first the sum of squares first basically, thats why we can't just add the norms
                W_norm = (w_norm ** p + b_norm ** p) ** (1.0 / p)
                ''' product norm collection '''
                prod_W_norm = prod_W_norm * W_norm
            else:
                W_norm = norm_func(param)
                ''' product norm collection '''
                prod_W_norm = prod_W_norm * W_norm
    return prod_W_norm.item()

def lp_normalizer(W,p,division_constant=1):
    '''
        return W.norm(p)
    '''
    W = W/division_constant
    return W.norm(p)

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
    RL_str = ''
    # TODO: IMPORTANT: Don't forget to include biases in the [W, b]
    print('start main')
    #path_all_expts = '/cbcl/cbcl01/brando90/home_simulation_research/overparametrized_experiments/pytorch_experiments/test_runs_flatness4'
    path_all_expts = '/cbcl/cbcl01/brando90/home_simulation_research/overparametrized_experiments/pytorch_experiments/test_runs_flatness5_ProperOriginalExpt'
    ''' expt_paths '''
    #list_names, RL_str, data_set_type = lists.DEBUG_KNOWN_NET()
    #list_names, RL_str, data_set_type = lists.experiment_RLNL_RL()
    #list_names, RL_str, data_set_type = lists.experiment_BigInits_MNIST_34u_2c_1fc()
    #list_names, RL_str, data_set_type = lists.experiment_BigInits_MNIST_34u_2c_1fc_hyperparams2()
    #list_names, RL_str, data_set_type = lists.experiment_BigInits_MNIST_different_HP()
    #list_names, RL_str, data_set_type = lists.experiment_cifar100_big_inits()
    #list_names, RL_str, data_set_type = lists.experiment_BigInits_MNIST_different_HP_HISTOGRAM()
    #list_names, RL_str, data_set_type = lists.experiment_Lambdas()
    #list_names, RL_str, data_set_type = lists.experiment_BigInits_MNIST_different_HP_product_norm_div()
    list_names, RL_str, data_set_type = lists.experiment_BigInits_MNIST_different_HP_product_norm_div()
    ###
    #list_names, RL_str, data_set_type = lists.experiment_BigInits_MNIST_34u_2c_1fc()
    list_names, RL_str, data_set_type = lists.expt_big_inits_cifar10()
    print(f'RL_str = {RL_str}')
    ''' normalization scheme '''
    p = 2
    division_constant = 1
    norm = f'l{p}_division_constant{division_constant}'
    weight_normalizer = lambda W: lp_normalizer(W,p,division_constant=division_constant)
    weight_normalizer.p = p
    normalization_scheme = lambda net: divide_params(net,weight_normalizer)
    #product_norm = lambda net: get_product_norm(net,weight_normalizer)
    #normalization_scheme = product_norm
    #norm = 'spectral'
    #normalization_scheme = lambda net: divide_params(net, spectral_normalization)
    print(f'norm = {norm}')
    ''' get results'''
    type_standardize = 'default'
    data_path = './data'
    target_loss = 0.0044
    normalizer = Normalizer(list_names, data_path,normalization_scheme, p,division_constant, data_set_type, type_standardize=type_standardize)
    results = normalizer.extract_all_results_vs_test_errors(path_all_expts, target_loss)
    #results = normalizer.get_hist_from_single_net(path_all_expts)
    ''' Saving '''
    print()
    path = os.path.join(path_all_expts, f'{RL_str}loss_vs_gen_errors_norm_{norm}_data_set_{data_set_type}')
    print(f'path = {path}')
    #path = os.path.join(path_all_expts, f'RL_corruption_1.0_loss_vs_gen_errors_norm_{norm}')
    #path = os.path.join(path_all_expts,f'loss_vs_gen_errors_norm_{norm}_final')
    scipy.io.savemat(path, results)
    ''' plot '''
    #plt.scatter(train_all_losses,gen_all_errors)
    #plt.show()
    ''' cry '''
    print('\a')

if __name__ == '__main__':
    time_taken = time.time()
    main()
    seconds_setup, minutes_setup, hours_setup = utils.report_times(time_taken)
    print('end')
    print('\a')
