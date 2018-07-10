clear;clc;clear;clc;
%% get path to folder
dot = '/Users/brandomiranda/home_simulation_research/overparametrized_experiments/';
path = fullfile(dot,'pytorch_experiments/test_runs_flatness2');
%NL
path_all_expts_NL = fullfile(path,'flatness_May_label_corrupt_prob_0.0_exptlabel_RadiusFlatnessNL_samples20_RLarge50')
%RLNL
path_all_expts_RLNL = fullfile(path,'flatness_May_label_corrupt_prob_0.0_exptlabel_RadiusFlatnessRLNL_samples20_RLarge50')
%% go through NL files
[rs,train_error_means,train_error_stds] = get_average_landscape(path_all_expts_NL);
%% plots
fig_nl = figure;
errorbar(rs,train_error_means,train_error_stds)
title_name = 'Average Landscape, Natural Label (NL)'
title(title_name);
xlabel('distance from minimum, R');
ylabel('Error');
curtick = get(gca, 'XTick');
set(gca, 'XTickLabel', cellstr(num2str(curtick(:))));
%%
[rs,train_error_means,train_error_stds] = get_average_landscape(path_all_expts_RLNL);
%% plots
fig_rlnl = figure;
errorbar(rs,train_error_means,train_error_stds)
title_name = 'Average Landscape, pretrained Random Labels retrained Natural Label (RLNL)'
title(title_name);
xlabel('distance from minimum, R');
ylabel('Error');
curtick = get(gca, 'XTick');
set(gca, 'XTickLabel', cellstr(num2str(curtick(:))));
%%
saveas(fig_nl,'fig_nl_hist');
saveas(fig_rlnl,'fig_rlnl_hist');
saveas(fig_nl,'fig_nl_hist','pdf');
saveas(fig_rlnl,'fig_rlnl_hist','pdf');
cd(dot)
%%
function [rs,train_error_means,train_error_stds] = get_average_landscape(path_all_expts)
% Extracts the set of radiuses where the epsilon jumped happened.
% return: all_X_error_r_eps = contains all the errors are Radius R [1,# directions/experiments] e.g.[1,40K]
actually_all_train_errors = [];
actually_all_train_losses = [];
actually_all_test_errors = [];
actually_all_test_losses = [];
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
    actually_all_train_errors = [actually_all_train_errors; all_train_errors];
    % get train loss radiuses
    %actually_all_train_losses = [actually_all_train_losses r_train_losses];    
    % get test error radiuses
    %actually_all_test_errors = [actually_all_test_errors r_test_errors];    
    % get test loss radiuses
    %actually_all_test_losses = [actually_all_test_losses r_test_losses];    
end
[train_error_means,train_error_stds] = get_means_all_data(actually_all_train_errors);
end
%
function [error_means,error_stds] = get_means_all_data(errors)
[~,nb_epochs] = size(errors);
error_means = zeros(1,nb_epochs);
error_stds = zeros(1,nb_epochs);
for epoch=1:nb_epochs
    error_means(1,epoch) = mean(errors(:,epoch));
    error_stds(1,epoch) = std(errors(:,epoch));
end
end