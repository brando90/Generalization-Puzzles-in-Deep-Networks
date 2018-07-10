clear;clc;clear;clc;
%% get path to folder
dot = '/Users/brandomiranda/home_simulation_research/overparametrized_experiments/';
path = fullfile(dot,'pytorch_experiments/test_runs_flatness2');
%NL
path_all_expts_NL = fullfile(path,'flatness_May_label_corrupt_prob_0.0_exptlabel_RadiusFlatnessNL_samples20_RLarge50')
path_all_expts_NL = fullfile(path,'flatness_May_label_corrupt_prob_0.0_exptlabel_RadiusFlatnessNL_samples20_RLarge_12.0')
%RLNL
path_all_expts_RLNL = fullfile(path,'flatness_May_label_corrupt_prob_0.0_exptlabel_RadiusFlatnessRLNL_samples20_RLarge50')
path_all_expts_RLNL = fullfile(path,'flatness_May_label_corrupt_prob_0.0_exptlabel_RadiusFlatnessRLNL_samples20_RLarge_12.0')
eps = 0.05;
nb_bins=20;
%% go through NL files
[all_train_error_r_eps,all_train_losses_r_eps,all_test_error_r_eps,all_test_losses_r_eps] = get_all_radius_arrays(path_all_expts_NL,eps);
%% plots
mean_radius_nl = mean(all_train_error_r_eps)
std_radius_nl = std(all_train_error_r_eps)
fig_nl = figure;
size(all_train_error_r_eps)
histogram(all_train_error_r_eps,nb_bins);
title('histogram of when landscape makes epsilon jump, Natural Label (NL)');
xlabel('radius random direction made epsilon jump');
ylabel('counts');
%%
[all_train_error_r_eps,all_train_losses_r_eps,all_test_error_r_eps,all_test_losses_r_eps] = get_all_radius_arrays(path_all_expts_RLNL,eps);
%% plots
mean_radius_rlnl = mean(all_train_error_r_eps)
std_radius_rlnl = std(all_train_error_r_eps)
fig_rlnl = figure;
size(all_train_error_r_eps)
histogram(all_train_error_r_eps,nb_bins);
title('histogram of when landscape makes epsilon jump, pretrained Random Labels retrained Natural Label (RLNL)');
xlabel('radius random direction made epsilon jump');
ylabel('counts');
%%
saveas(fig_nl,'fig_nl_hist');
saveas(fig_rlnl,'fig_rlnl_hist');
saveas(fig_nl,'fig_nl_hist','pdf');
saveas(fig_rlnl,'fig_rlnl_hist','pdf');
cd(dot)
%%
function [all_train_error_r_eps,all_train_losses_r_eps,all_test_error_r_eps,all_test_losses_r_eps] = get_all_radius_arrays(path_all_expts,eps)
% Extracts the set of radiuses where the epsilon jumped happened.
% return: all_X_error_r_eps = contains all the radiuses for each direction where an the error changed by epsilon [1,# directions/experiments] e.g.[1,40K]
all_train_error_r_eps = [];
all_train_losses_r_eps = [];
all_test_error_r_eps = [];
all_test_losses_r_eps = [];
%
path_plus_prefix_of_all_expts = fullfile(path_all_expts,'/flatness_*');
expt_data_files = dir(path_plus_prefix_of_all_expts);
expt_data_filenames = {expt_data_files.name};
for expt_file_name = expt_data_filenames
    %expt_file_name
    path_to_data_file = fullfile(path_all_expts,expt_file_name{1});
    load(path_to_data_file)
    %%
    rs = double(r_large).*interpolations;
    % get train error radiuses
    r_train_error_eps_s = get_all_eps_radiuses(eps,all_train_errors,rs);
    all_train_error_r_eps = [all_train_error_r_eps r_train_error_eps_s];
    % get train loss radiuses
    r_train_losses_eps_s = get_all_eps_radiuses(eps,all_train_losses,rs);
    all_train_losses_r_eps = [all_train_losses_r_eps r_train_losses_eps_s];    
    % get test error radiuses
    r_test_errors_eps_s = get_all_eps_radiuses(eps,all_test_errors,rs);
    all_test_error_r_eps = [all_test_error_r_eps r_test_errors_eps_s];    
    % get test loss radiuses
    r_test_losses_eps_s = get_all_eps_radiuses(eps,all_test_losses,rs);
    all_test_losses_r_eps = [all_test_losses_r_eps r_test_losses_eps_s];    
end
end
%
function rs_eps = get_all_eps_radiuses(eps,all_data,rs)
% collects the radiuses when all the data jumps by eps closest to the minimum
% eps = value to record when the epsilon jump is made (number,double/float)
% all_data = matrix with errors. Size = [# directions,# of samples up to R] e.g [1000,20]
% rs = radiuses [1,# samples up to R] e.g. [1,20]
[nb_dirs,nb_epochs] = size(all_data);
error_at_min = all_data(1,1);
rs_eps = -1*ones(1,nb_dirs);
for dir = 1:nb_dirs
    for epoch = 1:nb_epochs
        current_error_height = all_data(dir,epoch);
        if abs(error_at_min-current_error_height)>eps
            r_eps = rs(epoch);
            rs_eps(dir) = r_eps;
            break;
        end
    end
end
end