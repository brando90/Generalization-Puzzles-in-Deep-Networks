clear;clc;
%% root folder names
dot = '/Users/brandomiranda/home_simulation_research/overparametrized_experiments/';
path_all_expts = fullfile(dot,'pytorch_experiments/test_runs_flatness5_ProperOriginalExpt');
%% experiment files names
name = "Large_Inits_HIST_realloss_vs_gen_errors_norm_l2_division_constant1_data_set_mnist";
%name = "Large_Inits_HIST_real_dbloss_vs_gen_errors_norm_l2_division_constant1_data_set_mnist";
%% get K_alphas
gamma = 0.01;
[K_gammas,list_train_all_losses_normalized,stds] = extract_all_margin_based_values(path_all_expts,name,gamma);
%% plot scatter of K_gamma vs normalized train loss
fig1 = figure;
str_K_gamma = ['K_{' num2str(gamma) '}'];
scatter(list_train_all_losses_normalized,K_gammas)
%lscatter(list_train_all_losses_normalized,K_gammas,stds)
xlabel('normalized train loss');
ylabel(str_K_gamma);
title([str_K_gamma ' vs normalized train loss']);
%%
name = ['K' strrep(num2str(gamma),'.','p')]
saveas(fig1,name);
saveas(fig1,name,'pdf');
%%
function [K_gammas,list_train_all_losses_normalized,stds] = extract_all_margin_based_values(path_all_expts,name,gamma)
stds = [];
K_gammas = [];
list_train_all_losses_normalized = [];
%% load data
path_to_folder_expts = fullfile(path_all_expts,name);
load(path_to_folder_expts);
[N,~,~] = size(hist_all_train_norm);
N
for i = 1:N
    std = std_inits_all(i);
    train_losses = squeeze(hist_all_train_norm(i,:,:));
    [K_gamma,max_values,second_max_values] = get_margin_based_stats(train_losses,gamma);
    %%
    stds = [stds std];
    K_gammas = [K_gammas K_gamma];
    current_train_all_losses_normalized = train_all_losses_normalized(i);
    list_train_all_losses_normalized = [list_train_all_losses_normalized current_train_all_losses_normalized];
end
end
function [K_gamma,max_values,second_max_values] = get_margin_based_stats(train_losses,gamma)
[nb_rows,nb_classes] = size(train_losses);
max_values = zeros(nb_rows,1);
second_max_values = zeros(nb_rows,1);
indicies = 1:nb_classes;
for row=1:nb_rows
    %% get largest
    confidence_f_x = train_losses(row,:);
    [models_largest_confidence, max_index] = max(confidence_f_x);
    max_values(row) = models_largest_confidence;
    %% get second largest
    selected = ~(max_index == indicies); % logical index array that selects the arrays such that only the ones without the max are left out, since it will only match once, at the max index location the rest will be 0, then negating leaves the max index out
    confidence_f_x = train_losses(row,selected);
    models_2nd_largest_confidence = max(confidence_f_x);
    second_max_values(row) = models_2nd_largest_confidence;
end
%% compute max_y f_y - max_{c!=y} f_c
K_gammas = max_values - second_max_values < gamma;
K_gamma = (1/nb_rows)*sum(K_gammas);
end