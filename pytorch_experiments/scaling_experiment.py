import torch
import os
from math import inf

import scipy.io as sio

import data_classification as data_class
from new_training_algorithms import evalaute_mdl_data_set
import metrics
from good_minima_discriminator import divide_params_by
#from good_minima_discriminator import divide_params_by_taking_bias_into_account

from maps import NamedDict

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
        self.normalization_scheme = normalization_scheme
        ''' '''
        self.error = metrics.error_criterion
        self.loss = torch.nn.CrossEntropyLoss()
        self.iterations = iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainset, self.trainloader, self.testset, self.testloader, self.classes_data = data_class.get_cifer_data_processors(data_path,batch_size_train,batch_size_test,num_workers,label_corrupt_prob,standardize=standardize)

    def extract_all_results_vs_test_errors(self,path_all_expts,list_names,target_loss):
        ## main 2
        train_all_losses = []
        gen_all_errors = []
        ##
        epoch_all_numbers = []
        corruption_all_probs = []
        ##
        for name in list_names:
            path_to_folder_expts = os.path.join(path_all_expts,name)
            results = self.extract_results_with_target_loss(path_to_folder_expts, target_loss)
            ''' extend results ''' #
            train_all_losses.extend(results.train_losses)
            gen_all_errors.extend(results.gen_errors)
        return train_all_losses,gen_all_errors

    def extract_results_with_target_loss(self,path_to_folder_expts,target_loss):
        train_losses = []
        gen_errors = []
        ## TODO
        #epoch_numbers = []
        #orruption_probs = []
        ''' '''
        matlab_filenames = [filename for filename in os.listdir(path_to_folder_expts) if '.mat' in filename]
        for matlab_filename in matlab_filenames: # essentially looping through all the nets that were trained
            matlab_path = os.path.join(path_to_folder_expts,matlab_filename)
            mat_contents = sio.loadmat(matlab_path)
            ''' '''
            epoch,seed_id = self.match_train_error(target_loss, mat_contents)
            test_error,train_loss = self.get_results_from_normalized_net(epoch,seed_id, path_to_folder_expts)
            ''' '''
            train_losses.append(train_loss)
            gen_errors.append(test_error)
        results = NamedDict(train_losses=train_losses,gen_errors=gen_errors,epoch_numbers=epoch_numbers,corruption_probs=corruption_probs)
        return results

    def match_train_error(self,target_loss, mat_contents):
        train_losses = mat_contents['train_losses'][0]
        train_errors = mat_contents['train_errors'][0]
        for epoch in range(len(train_losses)):
            train_loss = train_losses[epoch]
            if abs(train_loss - target_loss) < 0.0001:
                train_error = train_errors[epoch]
                if train_error == 0.0:
                    seed = mat_contents['seed'][0][0]
                    return epoch,seed
        return -1,-1

    def get_results_from_normalized_net(self,epoch,seed_id, path_to_folder_expts):
        ''' '''
        ''' get net '''
        nets_folders = [filename for filename in os.listdir(path_to_folder_expts) if 'nets_folder' in filename]
        net_folder = [filename for filename in nets_folders if f'seed_{seed_id}' in filename][0] # note seed are unique very h.p.
        net_path = os.path.join(path_to_folder_expts,net_folder)
        net_name = [net_name for net_name in os.listdir(net_path) if f'epoch_{epoch}' in filename][0]
        net_path = os.path.join(net_path, net_name)
        net = torch.load(net_path)
        ''' normalize net '''
        net = self.normalize(net)
        ''' get train errors '''
        train_loss, train_error = evalaute_mdl_data_set(self.loss, self.error, net, self.trainloader, self.device)
        test_loss, test_error = evalaute_mdl_data_set(self.loss, self.error, net, self.testloader, self.device)
        return test_error, train_loss

    def normalize_net(self,net):
        return self.normalization_scheme(net)

def divide_params(net,norm):
    '''
    normalizes the network per layer
    '''
    params = net.named_parameters()
    dict_params = dict(params)
    for name, param in dict_params.items():
        if name in dict_params:
            W_norm = norm(param)
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
    print('start main')
    path_all_expts = '/cbcl/cbcl01/brando90/home_simulation_research/overparametrized_experiments/pytorch_experiments/test_runs_flatness4'
    ''' expt_paths '''
    list_names = []
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_NL_only_1st_layer_BIAS_True_batch_size_train_1024_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.0001_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.001_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.1_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.5_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.75_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_1.0_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0')
    ''' get results'''
    data_path = './data'
    target_loss = 0.0044
    normalizer = Normalizer(data_path)
    train_all_losses, gen_all_errors = normalizer.extract_all_results_vs_test_errors(path_all_expts,list_names,target_loss)
    print(f'train_all_losses,gen_all_errors={train_all_losses,gen_all_errors}')

if __name__ == '__main__':
    main()
    print('end')
