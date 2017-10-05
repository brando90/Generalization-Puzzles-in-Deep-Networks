clear;
%%
lambda = [0.1, 0.05, 0.025, 0.01, 0.005] %big go first
one_over_lambda = 1./lambda %small to big
%%
save_bool = 1
%%
test_errors = [13.79, 5.6,  2.95, 1.7, 1.1]
train_erros = [0.11, 0.03, 0.011, 0.002, 0.0005]
%%
lambda = [9000, 10000, 15000, 17000 ,20000] %big go first
one_over_lambda = lambda
%one_over_lambda = 1./lambda %small to big
%%
save_bool = 1
%%
test_errors = [0.642, 0.392, 0.300054,  0.2505, 0.2459]
train_erros = [0.001716, 0.00119, 0.0001249,  0.00003194, 0.00001133]
%% SUBPLOTS
% fig = figure
% subplot(2,1,1)
% plot(one_over_lambda,train_erros)
% title('1/ \lambda vs train errors')
% xlabel('1/ \lambda')
% ylabel('Error')
% 
% subplot(2,1,2)
% plot(one_over_lambda,test_errors)
% title('1/ \lambda vs test errors')
% xlabel('1/ \lambda')
% ylabel('Error')
% filename='one_over_lambda_vs_train_test_subplots'
%% plot train test same plot
fig = plot_test_train_same_plot( one_over_lambda,train_erros,test_errors );