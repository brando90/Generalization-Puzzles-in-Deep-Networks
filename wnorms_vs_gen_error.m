clear;clc;
%% root folder names
dot = '/Users/brandomiranda/home_simulation_research/overparametrized_experiments/';
path_all_expts = fullfile(dot,'pytorch_experiments/test_runs_flatness5_ProperOriginalExpt');
%% experiment files names
list_names = [];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_NL_only_1st_layer_BIAS_True_batch_size_train_1024_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0";list_names=[list_names, name];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.0001_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0";list_names=[list_names, name];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.001_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0";list_names=[list_names, name];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.01_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0";list_names=[list_names, name];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.1_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0";list_names=[list_names, name];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.2_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0";list_names=[list_names, name];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.5_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0";list_names=[list_names, name];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.75_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0";list_names=[list_names, name];
%name = "flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_1.0_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_200,250,300_gamma_1.0";list_names=[list_names, name];
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
%%
% norm = 'forbenius_norm';
% xlabel('forbenius norm (of all weights): ||W||_F')
% title('forbenius norm ||W|| vs generalization error')
% lsline
%%
% norm = 'product_norm';
% xlabel('product norm: ||w_k||...||w_1||')
% title('product norm ||w_k||...||w_1|| vs generalization error')
%%
norm = 'log_product_norm';
xlabel('log of product norm: log||w_k||+...+log||w_1||')
title('log of product norm log||w_k||+...+log||w_1|| vs generalization error')
lsline
%%
ylabel('generalization error/test error')
%errorbar(w_norms_means,gen_errors_means,gen_errors_stds)
%% save figure
saveas(fig,['gen_w_norms' norm]);
saveas(fig,['gen_w_norms' norm],'pdf');
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
    epoch = match_train_loss(target_loss,train_losses)
    if epoch ~= -1
        train_error = train_errors(epoch);
        if train_error == 0
            %w_norm = sum( w_norms(:,epoch).^2 )^0.5;
            w1_norm = (w_norms(1,epoch)^2+w_norms(2,epoch)^2)^0.5;
            w_norm = w1_norm*prod(w_norms(3:end,epoch));
            w_norm = log(w_norm);
            gen_error = test_errors(epoch)
            gen_errors = [gen_errors gen_error];
            w_norms_all = [w_norms_all w_norm];
        end
    end
end
end
%
function [epoch] = match_train_loss(target_loss,train_losses)
for epoch = 1:length(train_losses)
    train_loss = train_losses(epoch);
    if abs(target_loss-train_loss) < 0.0001 % catch 0.044 match 2 sign figs
        return;
    end
end
epoch = -1;
end