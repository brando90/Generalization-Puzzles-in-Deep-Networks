%clear;clc;
%load('./pytorch_experiments/test_runs_flatness4/loss_vs_gen_errors_norm_frobenius')
%load('./pytorch_experiments/test_runs_flatness5_ProperOriginalExpt/loss_vs_gen_errors_norm_frobenius_final')
%load('./pytorch_experiments/test_runs_flatness5_ProperOriginalExpt/loss_vs_gen_errors_norm_l1')
%load('./pytorch_experiments/test_runs_flatness5_ProperOriginalExpt/loss_vs_gen_errors_norm_l1_divided_by_10')
%load('./pytorch_experiments/test_runs_flatness5_ProperOriginalExpt/loss_vs_gen_errors_norm_l1_divided_by_100')RL_corruption_1.0_loss_vs_gen_errors_norm_l2
%load('./pytorch_experiments/test_runs_flatness5_ProperOriginalExpt/RL_corruption_1.0_loss_vs_gen_errors_norm_l2')
%%
markers = corruption_all_probs;
%markers = std_inits_all;
%%%%RLs = 8:19;
RLs = 62:73;
train_all_losses_normalized(RLs) = train_all_losses_normalized_rand(RLs);
test_all_losses_normalized(RLs) = test_all_losses_normalized_rand(RLs);
train_all_errors_unnormalized(RLs) = train_all_errors_unnormalized_rand(RLs);
gen_all_errors_unnormalized(RLs) = gen_all_errors_unnormalized_rand(RLs);

train_all_losses_unnormalized(RLs) = train_all_losses_unnormalized_rand(RLs);
test_all_losses_unnormalized(RLs) = test_all_losses_unnormalized_rand(RLs);
%% test error vs train error 
fig0 = figure;
lscatter(train_all_errors_unnormalized,gen_all_errors_unnormalized,markers)
xlim([-0.05,1])
xlabel('Train Error (Network Normalized)')
ylabel('Test Error (Network Normalized)')
%lsline
xlim([-0.05,1])
xlabel('Train Error (Network Normalized)')
ylabel('Test Error (Network Normalized)')
%% test error vs train loss (all normalized)
fig1 = figure;
lscatter(train_all_losses_normalized,gen_all_errors_normalized,markers)
%lscatter(all_train_errors,gen_all_errors_normalized,markers)
%lsline
%title('The weights of all models are normalized')
xlabel('Train Loss (Network Normalized)')
ylabel('Test Error (Network Normalized)')
%% IMPORTANT: test loss vs train loss (all normalized) - shows the linear correlation of the train loss and test loss
fig2 = figure;
scatter(train_all_losses_normalized,test_all_losses_normalized)
hold;
%
X = train_all_losses_normalized;
y = test_all_losses_normalized;
% X = [train_all_losses_normalized(1:7) train_all_losses_normalized(RLs)];
% y = [test_all_losses_normalized(1:7) test_all_losses_normalized(RLs)];
n = length(X);
Intercept = (1/n)*sum(y-X) % (1/n)*sum(y-x)
%
yCalc1 = X+Intercept;
Rsq1 = 1 - sum((y - yCalc1).^2)/sum((y - mean(y)).^2)
RMSE = sqrt(mean((y - yCalc1).^2))  % Root Mean Squared Error
%
plot(X,yCalc1);
%
xlabel('Train Loss (Network Normalized)')
ylabel('Test Loss (Network Normalized)')
%% IMPORTANT: test error (unnormalized) vs train loss (normalized), this checks if we can predict test error from train loss
fig3 = figure;
lscatter(train_all_losses_normalized,gen_all_errors_unnormalized,markers)
%lsline
%title('Train Loss vs Test Error')
xlabel('Train Loss (Network Normalized)')
ylabel('Test Error (Network Unnormalized)')
%% test loss (unormalized) vs train loss (normalized), not interesting cuz unnormalized loss diverges to infinity
fig4 = figure;
lscatter(train_all_losses_normalized,test_all_losses_unnormalized,markers)
%lsline
%title('Train Loss vs Test Loss')
xlabel('Train Loss (Network Normalized)')
ylabel('Test Loss (Network Unnormalized)')
%% CONTROL1: test loss (unormalized) vs train loss (unnormalized)
fig5 = figure;
scatter(train_all_losses_unnormalized,test_all_losses_unnormalized)
% h = scatter(train_all_losses_unnormalized,test_all_losses_unnormalized)
% c = get(h,'Color')
% c{1}
% c{2}

%lsline
title('Control 1: The weights of all models are unnormalized')
xlabel('Train Loss (Network Unnormalized)')
ylabel('Test Loss (Network Unnormalized)')
xlim([-2.5e-4,9e-3])
%% CONTROL2: test error (unormalized) vs train loss (unnormalized)
fig6 = figure;
lscatter(train_all_losses_unnormalized,gen_all_errors_unnormalized,markers)
%lsline
title('Control 2: The weights of all models are unnormalized')
xlabel('Train Loss (Network Unnormalized)')
ylabel('Test Error (Network Unnormalized)')
%% save
saveas(fig0,'test_error_vs_train_error_all_unnormalized');
saveas(fig0,'test_error_vs_train_error_all_unnormalized','pdf');
%
saveas(fig1,'test_error_vs_train_loss_all_normalized');
saveas(fig1,'test_error_vs_train_loss_all_normalized','pdf');
%
saveas(fig2,'important_test_loss_vs_train_loss_all_normalized');
saveas(fig2,'important_test_loss_vs_train_loss_all_normalized','pdf');
%
saveas(fig3,'important_test_error_vs_train_loss_unnormalized_vs_normalized');
saveas(fig3,'important_test_error_vs_train_loss_unnormalized_vs_normalized','pdf');
%
saveas(fig4,'test_loss_vs_train_loss_unnormalized_vs_normalized');
saveas(fig4,'test_loss_vs_train_loss_unnormalized_vs_normalized','pdf');
%
saveas(fig5,'control1_test_error_vs_train_loss_all_unnormalized');
saveas(fig5,'control1_test_error_vs_train_loss_all_unnormalized','pdf');
%
saveas(fig6,'control2_test_error_vs_train_loss_all_unnormalized');
saveas(fig6,'control2_test_error_vs_train_loss_all_unnormalized','pdf');