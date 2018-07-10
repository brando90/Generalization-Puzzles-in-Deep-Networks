clear;clc;clear;clc;
%% get path to folder
dot = '/Users/brandomiranda/home_simulation_research/overparametrized_experiments/';
path = fullfile(dot,'pytorch_experiments/test_runs_flatness2');
%NL
path_all_expts_NL = fullfile(path,'flatness_May_label_corrupt_prob_0.0_exptlabel_RadiusFlatnessNL_samples20_RLarge50')
%RLNL
path_all_expts_RLNL = fullfile(path,'flatness_May_label_corrupt_prob_0.0_exptlabel_RadiusFlatnessRLNL_samples20_RLarge50')
R = 44.7368;
index_R = get_closest(path_all_expts_NL,R);
nb_bins=50;
%% go through NL files
[all_train_error_r,all_train_losses_r,all_test_error_r,all_test_losses_r] = get_all_errors_at_radius_R(path_all_expts_NL,index_R);
%% plots
mean_error_nl = mean(all_train_error_r)
std_error_nl = std(all_train_error_r)
fig_nl = figure;
size(all_train_error_r)
histogram(all_train_error_r,nb_bins);
title_name = sprintf('Error histogram/distribution at R=%g, Natural Label (NL)',R)
title(title_name);
xlabel('Errors');
ylabel('counts');
curtick = get(gca, 'XTick');
set(gca, 'XTickLabel', cellstr(num2str(curtick(:))));
%%
[all_train_error_r,all_train_losses_r,all_test_error_r,all_test_losses_r] = get_all_errors_at_radius_R(path_all_expts_RLNL,index_R);
%% plots
error_radius_rlnl = mean(all_train_error_r)
std_error_rlnl = std(all_train_error_r)
fig_rlnl = figure;
size(all_train_error_r)
histogram(all_train_error_r,nb_bins);
title_name = sprintf('Error histogram/distribution at R=%g pretrained Random Labels retrained Natural Label (RLNL)',R)
title(title_name);
xlabel('Errors');
ylabel('counts');
curtick = get(gca, 'XTick');
set(gca, 'XTickLabel', cellstr(num2str(curtick(:))));
%%
saveas(fig_nl,'fig_nl_hist');
saveas(fig_rlnl,'fig_rlnl_hist');
saveas(fig_nl,'fig_nl_hist','pdf');
saveas(fig_rlnl,'fig_rlnl_hist','pdf');
cd(dot)
%%
function [all_train_error_r,all_train_losses_r,all_test_error_r,all_test_losses_r] = get_all_errors_at_radius_R(path_all_expts,index_R)
% Extracts the set of radiuses where the epsilon jumped happened.
% return: all_X_error_r_eps = contains all the errors are Radius R [1,# directions/experiments] e.g.[1,40K]
all_train_error_r = [];
all_train_losses_r = [];
all_test_error_r = [];
all_test_losses_r = [];
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
    r_train_error = get_errors_at_radius_R(all_train_errors,index_R);
    all_train_error_r = [all_train_error_r r_train_error];
    % get train loss radiuses
    r_train_losses = get_errors_at_radius_R(all_train_losses,index_R);
    all_train_losses_r = [all_train_losses_r r_train_losses];    
    % get test error radiuses
    r_test_errors = get_errors_at_radius_R(all_test_errors,index_R);
    all_test_error_r = [all_test_error_r r_test_errors];    
    % get test loss radiuses
    r_test_losses = get_errors_at_radius_R(all_test_losses,index_R);
    all_test_losses_r = [all_test_losses_r r_test_losses];    
end
end
%
function errors_at_r = get_errors_at_radius_R(all_data,index_R)
% returns all errors at index_R which corresponds at radisu R
% all_data = matrix with errors. Size = [# directions,# of samples up to R] e.g [1000,20]
% errors_at_r = errors at r
errors_at_r = all_data(:,index_R)'; % errors at index_R
end
%
function index_R = get_closest(path_all_expts,R_target)
% gets the index of where the r
path_plus_prefix_of_all_expts = fullfile(path_all_expts,'/flatness_*');
expt_data_files = dir(path_plus_prefix_of_all_expts);
expt_data_filenames = {expt_data_files.name};
for expt_file_name = expt_data_filenames
    %expt_file_name
    path_to_data_file = fullfile(path_all_expts,expt_file_name{1});
    load(path_to_data_file)
    %%
    rs = double(r_large).*interpolations;
    rs_diff_target = abs(rs - R_target);
    [M,index_R] = min(rs_diff_target)
    break
end
end