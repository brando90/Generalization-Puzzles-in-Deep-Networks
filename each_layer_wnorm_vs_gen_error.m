clear;clc;
%% root folder names
dot = '/Users/brandomiranda/home_simulation_research/overparametrized_experiments/';
path_all_expts = fullfile(dot,'pytorch_experiments/test_runs_flatness4');
%% experiment files names
list_names = [];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_1.0_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0";list_names=[list_names, name];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.75_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0";list_names=[list_names, name];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.5_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0";list_names=[list_names, name];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.2_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0";list_names=[list_names, name];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.1_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0";list_names=[list_names, name];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.001_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0";list_names=[list_names, name];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_RLNL_0.0001_only_1st_layer_BIAS_True_batch_size_train_256_lr_0.01_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0";list_names=[list_names, name];
name = "flatness_June_label_corrupt_prob_0.0_exptlabel_NL_only_1st_layer_BIAS_True_batch_size_train_1024_lr_0.01_momentum_0.9_scheduler_milestones_[200, 250, 300]_gamma_1.0";list_names=[list_names, name];
%%
target_loss = 0.0044
%% 
%[w_norms_means,w_norms_std, gen_errors_means,gen_errors_stds] = extract_w_norms_vs_gen_errors_results_stats(path_all_expts,list_names,target_loss);
%w_norms_means
%gen_errors_means
[w_all_norms_each_layer, gen_errors] = extract_w_norms_vs_gen_errors_results(path_all_expts,list_names,target_loss);
%%
fig1 = figure;
scatter( (w_all_norms_each_layer(1,:).^2+w_all_norms_each_layer(2,:).^2).^0.5 ,gen_errors);
lsline
norm1 = 'L2_norm_1st_layer';
xlabel('L2 norm of first layer: ||W_1||')
title('L2 norm ||W_1|| vs generalization error')
ylabel('generalization error/test error')
%%
fig2 = figure;
scatter(w_all_norms_each_layer(3,:),gen_errors);
lsline
norm2 = 'L2_norm_2nd_layer';
xlabel('L2 norm of second layer: ||W_2||')
title('L2 norm ||W_2|| vs generalization error')
ylabel('generalization error/test error')
%%
fig3 = figure;
scatter(w_all_norms_each_layer(4,:),gen_errors);
lsline
norm3 = 'L2_norm_3rd_layer';
xlabel('L2 norm of third/final layer: ||W_3||')
title('L2 norm ||W_3|| vs generalization error')
ylabel('generalization error/test error')
%% save figure
saveas(fig1,['gen_w_norms' norm1]);
saveas(fig1,['gen_w_norms' norm1],'pdf');
saveas(fig2,['gen_w_norms' norm2]);
saveas(fig2,['gen_w_norms' norm2],'pdf');
saveas(fig3,['gen_w_norms' norm3]);
saveas(fig3,['gen_w_norms' norm3],'pdf');
%% HELPER FUNCTIONS
function [w_all_norms_each_layer, gen_all_errors] = extract_w_norms_vs_gen_errors_results(path_all_expts,list_names,target_loss)
w_all_norms_each_layer = [];
gen_all_errors = [];
for name = list_names
    path_to_folder_expts = fullfile(path_all_expts,name);
    %% extra all data from experiments
    [gen_errors, w_norms] = extract_results_with_target_loss(path_to_folder_expts,target_loss);
    w_all_norms_each_layer = [w_all_norms_each_layer w_norms];
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
function [gen_errors,w_norms_all_each_layer] = extract_results_with_target_loss(path_to_folder_expts,target_loss)
gen_errors = [];
w_norms_all_each_layer = [];
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
            %w_norm = prod(w_norms(:,epoch))
            each_layer_weight = w_norms(:,epoch)
            gen_error = test_errors(epoch)
            gen_errors = [gen_errors gen_error];
            w_norms_all_each_layer = [w_norms_all_each_layer each_layer_weight];
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