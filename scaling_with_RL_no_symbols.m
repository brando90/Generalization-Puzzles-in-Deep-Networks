% plot RL vs NL
% markers for NL indicate the amount of std in init.
% 1) markers for RL indicate the amount of corruption of label (all have the same type of initialization)
% 2) markers for RL indicate the size of initialization (but can have a
% constant amount of corruption first and then increase it later. Note that
% to indicate this increase in corruption in label we would need perhaps
% colors of dots increasing in darkness for more corrupted)
%%
%markers = corruption_all_probs;
markers = std_inits_all;
RL = [];
for i=1:length(corruption_all_probs)
    %fprintf('corruption_all_probs(i) = %d\n',corruption_all_probs(i))
    if corruption_all_probs(i) == 1.0
        markers(i) = 1;
        RL = [RL i]
    end
end
%% test error vs train error 
fig0 = figure;
lscatter(train_all_errors_unnormalized,gen_all_errors_unnormalized,markers)
%scatter(train_all_errors_unnormalized,gen_all_errors_unnormalized)
% hold;
% scatter(train_all_errors_unnormalized(RL),gen_all_errors_unnormalized(RL));
% hold;
xlim([-0.05,1])
xlabel('Train Error (Network Unnormalized)')
ylabel('Test Error (Network Unnormalized)')
%lsline
% xlim([-0.05,1])
% xlabel('Train Error (Network Normalized)')
% ylabel('Test Error (Network Normalized)')
%% test error vs train loss (all normalized)
fig1 = figure;
lscatter(train_all_losses_normalized,gen_all_errors_normalized,markers)
xlabel('Train Loss (Network Normalized)')
ylabel('Test Error (Network Normalized)')
%% IMPORTANT: test loss vs train loss (all normalized) - shows the linear correlation of the train loss and test loss
fig2 = figure;
scatter(train_all_losses_normalized,test_all_losses_normalized)
%lscatter(train_all_losses_normalized,test_all_losses_normalized,markers)
hold;
%%
% hl = lsline;
% B = [ones(size(hl.XData(:))), hl.XData(:)]\hl.YData(:);
% Slope = B(2)
% Intercept = B(1)
% X = train_all_losses_normalized;y = test_all_losses_normalized;
% mdl = fitlm(X,y);
% RMSE = mdl.RMSE
% Rsquared_Ordinary = mdl.Rsquared.Ordinary
% Rsquared_adjusted = mdl.Rsquared
%%
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
yCalc1 = X+Intercept;
Rsq1 = 1 - sum((y - yCalc1).^2)/sum((y - mean(y)).^2)
RMSE = sqrt(mean((y - yCalc1).^2))  % Root Mean Squared Error
%%
xlabel('Train Loss (Network Normalized)')
ylabel('Test Loss (Network Normalized)')
%% IMPORTANT TEST ERROR CORRELATION: test error (unormalized) vs train loss (normalized)
fig3 = figure;
lscatter(train_all_losses_normalized,gen_all_errors_unnormalized,markers)
%scatter(train_all_losses_normalized,gen_all_errors_unnormalized)
xlabel('Train Loss (Network Normalized)')
ylabel('Test Error (Network Unnormalized)')
% 
% fig3 = figure;
% %lscatter(test_all_losses_unnormalized,gen_all_errors_unnormalized,markers)
% scatter(test_all_losses_unnormalized,gen_all_errors_unnormalized)
% %scatter(test_all_losses_unnormalized,gen_all_errors_unnormalized)
% xlabel('Test Loss (Network Unnormalized)')
% ylabel('Test Error (Network Unnormalized)')
%% CONTROL1: test loss (unormalized) vs train loss (unnormalized)
fig5 = figure;
%lscatter(train_all_losses_unnormalized,test_all_losses_unnormalized,markers)
scatter(train_all_losses_unnormalized,test_all_losses_unnormalized)
title('Control 1: The weights of all models are unnormalized')
xlabel('Train Loss (Network Unnormalized)')
ylabel('Test Loss (Network Unnormalized)')
%xlim([-2.5e-4,9e-3])
%% CONTROL2: test error (unormalized) vs train loss (unnormalized)
fig6 = figure;
%lscatter(train_all_losses_unnormalized,gen_all_errors_unnormalized,markers)
scatter(train_all_losses_unnormalized,gen_all_errors_unnormalized)
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
saveas(fig5,'control1_test_vs_train_loss_all_unnormalized');
saveas(fig5,'control1_test_vs_train_loss_all_unnormalized','pdf');
%
saveas(fig6,'control2_test_error_vs_train_loss_all_unnormalized');
saveas(fig6,'control2_test_error_vs_train_loss_all_unnormalized','pdf');