'''
Module that has the list of names of the experiments to process for making figures.
'''

def DEBUG_KNOWN_NET():
    '''
    Experiment where we pre-trained on RL then NL different degrees of corruption.
    Last point is the RL points, trained only on RL.
    '''
    # net_18_July_sj_0_staid_0_seed_25144932459028958_polestar-old
    # loss_original=3.5762119288847335e-05,error_original=0.0
    # loss_restored=3.576211930291418e-05,error_restored=0.0
    data_set_type = 'mnist'
    RL_str = ''
    list_names = []
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_debug_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means__stds__batch_size_train_1024_lr_0.02_momentum_0.95_epochs_200')
    RL_str = 'DEBUG_known_net'
    return list_names, RL_str, data_set_type

def experiment_RLNL_RL():
    '''
    Experiment where we pre-trained on RL then NL different degrees of corruption.
    Last point is the RL points, trained only on RL.
    '''
    data_set = 'cifar10'
    RL_str = ''
    list_names = []
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_NL_only_1st_layer_BIAS_True_batch_size_train_1024_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.0001_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.001_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.01_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.1_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.2_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.5_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.75_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    ### list_names.append('flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_1.0_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    ### list_names.append('flatness_June_label_corrupt_prob_1.0_exptlabel_RLInits_only_1st_layer_BIAS_True_batch_size_train_1024_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    #list_names.append('flatness_June_label_corrupt_prob_1.0_exptlabel_RL_only_1st_layer_BIAS_True_batch_size_train_1024_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0')
    #RL_str ='debug'
    #RL_str = 'RL_point_'
    #RL_str = 'RL_point_and_0NL_'
    RL_str = 'Only_0NL_'
    return list_names, RL_str, data_set

def experiment_BigInits_MNIST_24u_2c_1fc():
    '''
    Experiment where we pre-trained on RL then NL different degrees of corruption.
    Last point is the RL points, trained only on RL.
    '''
    data_set = 'mnist_10classes'
    RL_str = ''
    list_names = []
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_24_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.001,0.001,0.001]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_24_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.05,0.05,0.05]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_24_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.075,0.075,0.075]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_24_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.1,0.1,0.1]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_24_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.125,0.125,0.125]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_24_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.15,0.15,0.15]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_24_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.18,0.18,0.18]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_24_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.19,0.19,0.19]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    #list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_24_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.2,0.2,0.2]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    ##list_names.append('')
    #RL_str = 'debug'
    #RL_str = 'RL_point_'
    #RL_str = 'RL_point_and_0NL_'
    #RL_str = 'Only_NL_'
    return list_names, RL_str, data_set

def experiment_BigInits_MNIST_34u_2c_1fc():
    '''
    Experiment where we pre-trained on RL then NL different degrees of corruption.
    Last point is the RL points, trained only on RL.
    '''
    RL_str = 'debug'
    RL_str = ''
    ''' data set '''
    data_set_type = 'mnist'
    ''' RL string '''
    RL_str = 'Large_Inits'
    RL_str = 'RL_point_included'
    ''' list names '''
    list_names = []
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.001,0.001,0.001]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.01,0.01,0.01]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.05,0.05,0.05]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.075,0.075,0.075]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.1,0.1,0.1]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.125,0.125,0.125]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.15,0.15,0.15]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.18,0.18,0.18]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.19,0.19,0.19]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    ####list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_24_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.2,0.2,0.2]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    if 'RL' in RL_str:
        list_names.append('flatness_July_label_corrupt_prob_1.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.01,0.01,0.01]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    return list_names, RL_str, data_set_type

def experiment_BigInits_MNIST_34u_2c_1fc_hyperparams2():
    '''
    Experiment where we pre-trained on RL then NL different degrees of corruption.
    Last point is the RL points, trained only on RL.
    '''
    RL_str = 'debug'
    RL_str = ''
    ''' data set '''
    data_set_type = 'mnist'
    ''' RL string '''
    RL_str = 'Large_Inits'
    #RL_str = 'RL_point_included'
    RL_str = 'Only_RL_vs_NL'
    ''' list names '''
    list_names = []
    #list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.001,0.001,0.001]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.0025,0.0025,0.0025]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.005,0.05,0.05]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.01,0.01,0.01]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.05,0.05,0.05]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.075,0.075,0.075]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.1,0.1,0.1]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.125,0.125,0.125]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.15,0.15,0.15]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.18,0.18,0.18]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.19,0.19,0.19]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    ###list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_24_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.2,0.2,0.2]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    if 'RL' in RL_str:
        #list_names.append('flatness_July_label_corrupt_prob_1.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.001,0.001,0.001]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
        #list_names.append('flatness_July_label_corrupt_prob_1.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.001,0.001,0.001]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
        list_names.append('flatness_July_label_corrupt_prob_1.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.05,0.05,0.05]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    return list_names, RL_str, data_set_type

def experiment_BigInits_MNIST_different_HP():
    '''
    Experiment where we pre-trained on RL then NL different degrees of corruption.
    Last point is the RL points, trained only on RL.
    '''
    RL_str = 'debug'
    RL_str = ''
    ''' data set '''
    data_set_type = 'mnist'
    ''' RL string '''
    RL_str = 'Large_Inits2'
    #RL_str = 'RL_point_included'
    #RL_str = 'RL_point_included_diff_hp'
    #RL_str = 'Only_RL_vs_NL'
    ''' list names '''
    list_names = []
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.001,0.001,0.001]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.0025,0.0025,0.0025]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.005,0.05,0.05]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.01,0.01,0.01]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.05,0.05,0.05]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.075,0.075,0.075]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.1,0.1,0.1]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.125,0.125,0.125]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.15,0.15,0.15]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.18,0.18,0.18]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.19,0.19,0.19]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    ###list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_24_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.2,0.2,0.2]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    if 'RL' in RL_str:
        list_names.append('flatness_July_label_corrupt_prob_1.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.05,0.05,0.05]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    return list_names, RL_str, data_set_type

def experiment_cifar100_big_inits():
    '''
    Experiment where we pre-trained on RL then NL different degrees of corruption.
    Last point is the RL points, trained only on RL.
    '''
    RL_str = 'debug'
    RL_str = ''
    ''' data set '''
    data_set_type = 'cifar100'
    ''' RL string '''
    RL_str = 'Large_Inits'
    #RL_str = 'RL_point_included'
    #RL_str = 'RL_point_included_diff_hp'
    #RL_str = 'Only_RL_vs_NL'
    ''' list names '''
    list_names = []
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_35_units_2_layers_only_1st_layer_BIAS_True_data_set_cifar100_means_[0,0,0]_stds_[0.2,0.2,0.2]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_4000_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_35_units_2_layers_only_1st_layer_BIAS_True_data_set_cifar100_means_[0,0,0]_stds_[0.18,0.18,0.18]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_4000_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_35_units_2_layers_only_1st_layer_BIAS_True_data_set_cifar100_means_[0,0,0]_stds_[0.001,0.001,0.001]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_4000_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_35_units_2_layers_only_1st_layer_BIAS_True_data_set_cifar100_means_[0,0,0]_stds_[0.1,0.1,0.1]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_4000_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_35_units_2_layers_only_1st_layer_BIAS_True_data_set_cifar100_means_[0,0,0]_stds_[0.19,0.19,0.19]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_4000_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_35_units_2_layers_only_1st_layer_BIAS_True_data_set_cifar100_means_[0,0,0]_stds_[0.075,0.075,0.075]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_4000_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_35_units_2_layers_only_1st_layer_BIAS_True_data_set_cifar100_means_[0,0,0]_stds_[0.15,0.15,0.15]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_4000_scheduler_milestones_200,250,300_gamma_1.0')
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_35_units_2_layers_only_1st_layer_BIAS_True_data_set_cifar100_means_[0,0,0]_stds_[0.05,0.05,0.05]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_4000_scheduler_milestones_200,250,300_gamma_1.0')
    if 'RL' in RL_str:
        list_names.append('')
    return list_names, RL_str, data_set_type

def experiment_BigInits_MNIST_different_HP_HISTOGRAM():
    '''
    Experiment where we pre-trained on RL then NL different degrees of corruption.
    Last point is the RL points, trained only on RL.
    '''
    RL_str = 'debug'
    RL_str = ''
    ''' data set '''
    data_set_type = 'mnist'
    ''' RL string '''
    RL_str = 'Large_Inits_HIST'
    #RL_str = 'RL_point_included'
    #RL_str = 'RL_point_included_diff_hp'
    #RL_str = 'Only_RL_vs_NL'
    ''' list names '''
    list_names = []
    list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.001,0.001,0.001]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.0025,0.0025,0.0025]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.005,0.05,0.05]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.01,0.01,0.01]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.05,0.05,0.05]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.075,0.075,0.075]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.1,0.1,0.1]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.125,0.125,0.125]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.15,0.15,0.15]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.18,0.18,0.18]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    # list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.19,0.19,0.19]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    ###list_names.append('flatness_July_label_corrupt_prob_0.0_exptlabel_NL_24_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.2,0.2,0.2]_batch_size_train_1024_lr_0.02_momentum_0.95_epochs_800')
    if 'RL' in RL_str:
        list_names.append('flatness_July_label_corrupt_prob_1.0_exptlabel_NL_34_units_2_layers_only_1st_layer_BIAS_True_data_set_mnist_reg_param_0.0_means_[0,0,0]_stds_[0.05,0.05,0.05]_batch_size_train_1024_lr_0.01_momentum_0.9_epochs_800')
    return list_names, RL_str, data_set_type