clear;clc;
%% root folder names
dot = '/Users/brandomiranda/home_simulation_research/overparametrized_experiments/';
path_all_expts = fullfile(dot,'pytorch_experiments/test_runs_flatness2');
%% experiment files names
list_names = [];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption1.0_lr_0.01_momentum_0.9";list_names=[list_names, name];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption0.8_lr_0.01_momentum_0.9";list_names=[list_names, name];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption0.7_lr_0.01_momentum_0.9";list_names=[list_names, name];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption0.6_lr_0.01_momentum_0.9";list_names=[list_names, name];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption0.5_lr_0.01_momentum_0.9";list_names=[list_names, name];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption0.4_lr_0.01_momentum_0.9";list_names=[list_names, name];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption0.3_lr_0.01_momentum_0.9";list_names=[list_names, name];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption0.2_lr_0.01_momentum_0.9";list_names=[list_names, name];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption0.1_lr_0.01_momentum_0.9";list_names=[list_names, name];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption0.05_lr_0.01_momentum_0.9";list_names=[list_names, name];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption0.01_lr_0.01_momentum_0.9";list_names=[list_names, name];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption0.001_lr_0.01_momentum_0.9";list_names=[list_names, name];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption0.0005_lr_0.01_momentum_0.9";list_names=[list_names, name];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption0.0001_lr_0.01_momentum_0.9";list_names=[list_names, name];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption5e-5_lr_0.01_momentum_0.9";list_names=[list_names, name];
name = "flatness_May_label_corrupt_prob_0.0_exptlabel_MoreRLNLmdls_label_corruption5e-6_lr_0.01_momentum_0.9";list_names=[list_names, name];
%name = "flatness_May_label_corrupt_prob_0.0_exptlabel_SGD_ManyRuns_Momentum0.9";list_names=[list_names, name];
%%
target_loss = 0.0044
%% 
%[w_norms_means,w_norms_std, gen_errors_means,gen_errors_stds] = extract_w_norms_vs_gen_errors_results_stats(path_all_expts,list_names,target_loss);
%w_norms_means
%gen_errors_means
[w_all_norms, gen_errors] = extract_w_norms_vs_gen_errors_results(path_all_expts,list_names,target_loss);
%%
fig = figure;
scatter(w_all_norms,gen_errors);
%plot(w_norms_means,gen_errors_means)
%scatter(w_norms_means,gen_errors_means)
lsline
xlabel('product norm, ||w_k||...||w_1||')
ylabel('generalization error/test error')
title('||w|| vs generalization error')
%errorbar(w_norms_means,gen_errors_means,gen_errors_stds)
%% save figure
saveas(fig,'gen_w_norms');
saveas(fig,'gen_w_norms','pdf');
%% HELPER FUNCTIONS
function [w_all_norms, gen_all_errors] = extract_w_norms_vs_gen_errors_results(path_all_expts,list_names,target_loss)
w_all_norms = [];
gen_all_errors = [];
for name = list_names
    path_to_folder_expts = fullfile(path_all_expts,name);
    %% extra all data from experiments
    [gen_errors, w_norms] = extract_results_with_target_loss(path_to_folder_expts,target_loss);
    w_all_norms = [w_all_norms w_norms];
    gen_all_errors = [gen_all_errors gen_errors];
end
end
%%
function [w_norms_means,w_norms_std, gen_errors_means,gen_errors_stds] = extract_w_norms_vs_gen_errors_results_stats(path_all_expts,list_names,target_loss)
w_norms_means = [];
w_norms_stds = [];
gen_errors_means = [];
gen_errors_stds = [];
for name = list_names
    path_to_folder_expts = fullfile(path_all_expts,name);
    %% extract stats
    [w_norms_all, gen_errors] = extract_results_with_target_loss(path_to_folder_expts,target_loss);
    w_norms_mean = mean(w_norms_all);
    w_norms_std = std(w_norms_all);
    gen_errors_mean = mean(gen_errors);
    gen_errors_std = std(gen_errors);
    %% store stats
    w_norms_means = [w_norms_means w_norms_mean];
    w_norms_stds = [w_norms_stds w_norms_std];
    gen_errors_means = [gen_errors_means gen_errors_mean];
    gen_errors_stds = [gen_errors_stds gen_errors_std];
end
end
%
function [gen_errors,w_norms_all] = extract_results_with_target_loss(path_to_folder_expts,target_loss)
gen_errors = [];
w_norms_all = [];
%%
path_plus_prefix_of_all_expts = fullfile(path_to_folder_expts,'/flatness_*');
expt_data_files = dir(path_plus_prefix_of_all_expts);
expt_data_filenames = {expt_data_files.name};
for expt_file_name = expt_data_filenames
    %expt_file_name
    path_to_data_file = fullfile(path_to_folder_expts,expt_file_name{1});
    load(path_to_data_file)
    %%
    epoch = match_train_error(target_loss,train_losses);
    if epoch ~= -1
        train_error = train_errors(epoch);
        if train_error == 0
            %w_norm = sum(w_norms(:,epoch))
            w_norm = prod(w_norms(:,epoch))
            gen_error = test_errors(epoch)
            gen_errors = [gen_errors gen_error];
            w_norms_all = [w_norms_all w_norm];
        end
    end
end
end
%
function [epoch] = match_train_error(target_loss,train_losses)
for epoch = 1:length(train_losses)
    train_loss = train_losses(epoch);
    if abs(target_loss-train_loss) < 0.0001 % catch 0.044 match 2 sign figs
        return;
    end
end
epoch = -1;
end