clear;clc;clear;clc;
%% get path to folder
dot = '/Users/brandomiranda/home_simulation_research/overparametrized_experiments/';
path = fullfile(dot,'pytorch_experiments/test_runs_flatness2');
%NL
path_all_expts = fullfile(path,'/RedoFig5_Cheby/N_train_9_N_test_100_batch_size_9_perturb_freq_2500_perturb_magnitude_0.45_momentum_0.0_iterations_switch_mode_1/expt_type_DEGREES_40')
%% go through all the expt files
[train_means,test_means,w_norms_means] = get_means_of_experiments(path_all_expts);
%%
nb_iter = length(train_means);
iterations = 1:nb_iter;
fig_errors = figure;
plot(iterations,train_means);
hold on;
plot(iterations,test_means);
title('Train/Test error vs Iterations')
xlabel('number of iterations')
ylabel('l2 Error');
%
fig_w_norms = figure;
plot(iterations,w_norms_means);
title('norm of W');
xlabel('number of iterations')
ylabel('l2 norm of W');
%%
function [train_means,test_means,w_norms_means] = get_means_of_experiments(path_all_expts)
train_means = [];
test_means = [];
w_norms_means = [];
%
path_plus_prefix_of_all_expts = fullfile(path_all_expts,'/satid_*');
expt_data_files = dir(path_plus_prefix_of_all_expts);
expt_data_filenames = {expt_data_files.name};
for expt_file_name = expt_data_filenames
    %expt_file_name
    path_to_data_file = fullfile(path_all_expts,expt_file_name{1});
    load(path_to_data_file)
    %%
    train_means = [train_means ; train_errors];
    test_means = [test_means ; test_errors];
    w_norms_means = [w_norms_means ; w_norms];
end
train_means = mean(train_means,1);
test_means = mean(test_means,1);
w_norms_means = mean(w_norms_means,1);
end