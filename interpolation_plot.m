%clear;clc;
%% root folder names
dot = "/Users/brandomiranda/home_simulation_research/overparametrized_experiments/";
path_all_expts = fullfile(dot,'pytorch_experiments/test_runs_flatness2');
%% names
%name = "flatness_May_label_corrupt_prob_0.0_exptlabel_Movie_NL0.0_RLNL0.5_WRLNL_Mdist_NL_1.0_Wrlnl_lr_0.01_momentum_0.9/flatness_23_May_sj_1213_staid_1_seed_27177842637628052_polestar-old.mat"
%name = "flatness_May_label_corrupt_prob_0.0_exptlabel_Movie_NL0.0_RLNL0.5_WRLNL_0.75_Mdist_NL_0.85925_Wrlnl_lr_0.01_momentum_0.9/flatness_23_May_sj_1211_staid_1_seed_34664358959217573_polestar-old.mat"
%name = "flatness_May_label_corrupt_prob_0.0_exptlabel_Movie_NL0.0_RLNL0.5_WRLNL_0.5_Mdist_NL_0.7184_Wrln_lr_0.01_momentum_0.9/flatness_23_May_sj_1210_staid_1_seed_67834155599723600_polestar-old.mat"
%name = "flatness_May_label_corrupt_prob_0.0_exptlabel_Movie_NL0.0_RLNL0.5_WRLNL_0.25_Mdist_NL_0.577_Wrlnl_lr_0.01_momentum_0.9/flatness_23_May_sj_1209_staid_1_seed_68711334840760903_polestar-old.mat"
%name = "flatness_May_label_corrupt_prob_0.0_exptlabel_Movie_NL0.0_RLNL0.5_WRLNL_1.0_Mdist_NL_0.437_Wrlnl_lr_0.01_momentum_0.9/flatness_23_May_sj_1214_staid_1_seed_21573795583809121_polestar-old.mat"
%% xaxis and label
%path_to_folder_expts = fullfile(path_all_expts,name)
%load(path_to_folder_expts);
%% plot interpolation
%expt_type = 'alpha';
expt_type = 'radius_distance';
if strcmp(expt_type,'alpha')
    xaxis=interpolations;
    abcissa_label = 'convex parameter \alpha';
else % strcmp(expt_type,'radius_distance')
    xaxis=rs;
    abcissa_label = 'distance from Random Labels ("bad") minima';
end
%% accs
% fig_accs = figure;
% plot(xaxis,train_accs);hold;
% plot(xaxis,test_accs);
% legend('train accuracy','test accuracy')
% title('accuracy vs interpolation parameters')
% ylabel('accuracy');
% xlabel(abcissa_label);
%% errors
fig_errors = figure;
plot(xaxis,train_errors);hold;
plot(xaxis,test_errors);
legend('train errors','test error')
title('errors vs interpolation parameters')
ylabel('error');
xlabel(abcissa_label);
xlim([xaxis(1),xaxis(end)])
%% losses
fig_losses = figure;
plot(xaxis,train_losses);hold;
plot(xaxis,test_losses);
legend('train loss','test loss')
title('loss vs interpolation parameters')
ylabel('loss');
xlabel(abcissa_label);
xlim([xaxis(1),xaxis(end)])
%%
W_rlnl = sum(w_norms(:,1))
W_nl = sum(w_norms(:,end))
%% save
saveas(fig_errors,['./fig_errors_interpolation_sj'])
saveas(fig_errors,['./fig_errors_interpolation_sj'],'pdf')
saveas(fig_losses,['./fig_losses_interpolation_sj'])
saveas(fig_losses,['./fig_losses_interpolation_sj'],'pdf')
% saveas(fig_accs,['./fig_accs_interpolation_sj' num2str(sj)])
% saveas(fig_accs,['./fig_accs_interpolation_sj' num2str(sj)],'pdf')