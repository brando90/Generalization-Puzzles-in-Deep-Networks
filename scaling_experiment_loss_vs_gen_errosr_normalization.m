clear;clc;
%load('./pytorch_experiments/test_runs_flatness4/loss_vs_gen_errors_norm_frobenius')
load('./pytorch_experiments/test_runs_flatness5_ProperOriginalExpt/loss_vs_gen_errors_norm_frobenius_final')
%% test error vs train loss (all normalized)
fig1 = figure;
lscatter(train_all_losses_normalized,gen_all_errors_normalized,corruption_all_probs)
%lsline
%title('The weights of all models are normalized')
xlabel('Train Loss (Network Normalized)')
ylabel('Test Error (Network Normalized)')
%% test loss vs train loss (all normalized)
fig2 = figure;
lscatter(train_all_losses_normalized,test_all_losses_normalized,corruption_all_probs)
%lsline
%title('The weights of all models are normalized')
xlabel('Train Loss (Network Normalized)')
ylabel('Test Loss (Network Normalized)')
%% test error (unormalized) vs train loss (normalized)
fig3 = figure;
lscatter(train_all_losses_normalized,gen_all_errors_unnormalized,corruption_all_probs)
%lsline
%title('Train Loss vs Test Error')
xlabel('Train Loss (Network Normalized)')
ylabel('Test Error (Network Unnormalized)')
%% test loss (unormalized) vs train loss (normalized), not interesting cuz unnormalized loss diverges to infinity
fig4 = figure;
lscatter(train_all_losses_normalized,test_all_losses_unnormalized,corruption_all_probs)
%lsline
%title('Train Loss vs Test Loss')
xlabel('Train Loss (Network Normalized)')
ylabel('Test Loss (Network Unnormalized)')
%% CONTROL: test loss (unormalized) vs train loss (unnormalized)
fig5 = figure;
lscatter(train_all_losses_unnormalized,gen_all_errors_unnormalized,corruption_all_probs)
%lsline
title('The weights of all models are unnormalized')
xlabel('Train Loss (Network Unnormalized)')
ylabel('Test Loss (Network Unnormalized)')
%% save
saveas(fig1,'test_error_vs_train_loss_all_normalized');
saveas(fig1,'test_error_vs_train_loss_all_normalized','pdf');
%
saveas(fig2,'test_loss_vs_train_loss_all_normalized');
saveas(fig2,'test_loss_vs_train_loss_all_normalized','pdf');
%
saveas(fig3,'test_error_vs_train_loss_unnormalized_vs_normalized');
saveas(fig3,'test_error_vs_train_loss_unnormalized_vs_normalized','pdf');
%
saveas(fig4,'test_loss_vs_train_loss_unnormalized_vs_normalized');
saveas(fig4,'test_loss_vs_train_loss_unnormalized_vs_normalized','pdf');
%
saveas(fig5,'control_test_error_vs_train_loss_all_unnormalized');
saveas(fig5,'control_test_error_vs_train_loss_all_unnormalized','pdf');