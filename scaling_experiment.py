import torch
import os

import scipy.io as sio

from pdb import set_trace as st

def get_corruption_label( path_to_experiment ):
    '''
    extract corrpution label from path to experiments
    e.g. flatness_June_label_corrupt_prob_0.0_exptlabel_NL_only_1st_layer_BIAS_True_batch_size_train_1024_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0
    '''
    corrupt_prob = path_to_experiment.split('corrupt_prob_')[1].split('_exptlabel_NL')
    return float(corrupt_prob)

def get_epoch_number(matlab_path):
    mat_contents = sio.loadmat(matlab_path)
    train_errors = mat_contents['train_errors'][0]
    test_errors = mat_contents['test_errors'][0]
    train_losses = mat_contents['train_losses'][0]
    seed = mat_contents['seed'][0][0]
    epoch = -1
    for epoch in range(len(test_errors)):
        
    st()

def get_everything(expt_paths):
    loss_vs_test = []
    ''' for every experiment in all experiments '''
    for path_to_experiment in expt_paths: # for every experiment NL, RLNL1, ..., RLNL10 etc
        ## construct a specific folder that points to an experiment NL/RLNL_1.0,0.75,...,0.0001
        path_to_experiment = os.path.join('pytorch_experiments/test_runs_flatness4',path_to_experiment) # e.g. pytorch_experiments/test_runs_flatness4/flatness_June_label_corrupt_prob_0.0_exptlabel_NL_only_1st_layer_BIAS_True_batch_size_train_1024_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0
        corrupt_prob = get_corruption_label( path_to_experiment )
        print(f'\npath_to_experiment = {path_to_experiment}')
        ''' for all data in current path to experiments '''
        for dirpath, dirnames, filenames in os.walk(path_to_experiment):
            if dirpath == path_to_experiment: # makes sure we loop only through current directory contents
                # filenames would contain the matlab files which we need (the net at the final iteration, we don't need those) || e.g. filenames = ['flatness_14_June_sj_570_staid_8_seed_9032528616438533_polestar-old.mat', 'net_14_June_sj_570_staid_8_seed_9032528616438533_polestar-old', 'flatness_14_June_sj_576_staid_6_seed_19910608494692730_polestar-old.mat', 'net_14_June_sj_576_staid_6_seed_19910608494692730_polestar-old', 'flatness_14_June_sj_572_staid_2_seed_33985094234217568_polestar-old.mat', 'net_14_June_sj_572_staid_2_seed_33985094234217568_polestar-old', 'flatness_14_June_sj_573_staid_3_seed_30377539497157790_polestar-old.mat', 'net_14_June_sj_573_staid_3_seed_30377539497157790_polestar-old', 'flatness_14_June_sj_571_staid_1_seed_40364281550521465_polestar-old.mat', 'net_14_June_sj_571_staid_1_seed_40364281550521465_polestar-old', 'flatness_14_June_sj_575_staid_5_seed_33458967967300360_polestar-old.mat', 'net_14_June_sj_575_staid_5_seed_33458967967300360_polestar-old', 'flatness_14_June_sj_577_staid_7_seed_21743562594991433_polestar-old.mat', 'net_14_June_sj_577_staid_7_seed_21743562594991433_polestar-old', 'flatness_14_June_sj_574_staid_4_seed_11190672253866788_polestar-old.mat', 'net_14_June_sj_574_staid_4_seed_11190672253866788_polestar-old’]
                matlab_filenames = [ filename for filename in filenames if '.mat' in filename ]
                # note this has all the nets, we'll need them to for pointing to the path to lead the net to evaluate it and modify it || e.g. dirnames = ['nets_folder_14_June_sj_573_staid_3_seed_30377539497157790_polestar-old', 'nets_folder_14_June_sj_572_staid_2_seed_33985094234217568_polestar-old', 'nets_folder_14_June_sj_576_staid_6_seed_19910608494692730_polestar-old', 'nets_folder_14_June_sj_570_staid_8_seed_9032528616438533_polestar-old', 'nets_folder_14_June_sj_571_staid_1_seed_40364281550521465_polestar-old', 'nets_folder_14_June_sj_575_staid_5_seed_33458967967300360_polestar-old', 'nets_folder_14_June_sj_577_staid_7_seed_21743562594991433_polestar-old', 'nets_folder_14_June_sj_574_staid_4_seed_11190672253866788_polestar-old']
                dirnames_nets = dirnames
                for matlab_filename in matlab_filenames:
                    ##
                    matlab_path = os.path.join(path_to_experiment,matlab_filename)
                    epoch,stid = get_epoch_number(matlab_path)
                    ##
                    #net = get_net(epoch,stid,dirnames,root_path=path)
                    #all_stats_for_expt = get_stats(net)
    #save_to_matlab(all_stats_for_expt)

def main():
    print('start main')
    #expt_paths = 'pytorch_experiments/test_runs_flatness4'
    ''' expt_paths '''
    expt_paths = []
    expt_paths.append('flatness_June_label_corrupt_prob_0.0_exptlabel_NL_only_1st_layer_BIAS_True_batch_size_train_1024_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0')
    expt_paths.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.0001_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0')
    expt_paths.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.001_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0')
    expt_paths.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.1_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0')
    expt_paths.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.5_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0')
    expt_paths.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.75_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0')
    expt_paths.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_1.0_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.1_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0')
    ''' '''
    target_loss = 0.0044
    get_everything(expt_paths,target_loss)

if __name__ == "__main__":
    main()
