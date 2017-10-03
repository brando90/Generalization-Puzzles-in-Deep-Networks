clear;
%%
lambda = [600, 800, 1000, 1200, 1400] %big go first
one_over_lambda = lambda
%one_over_lambda = 1./lambda %small to big
%%
save_bool = 0
%%
lambda = [0.1, 0.05, 0.025, 0.01, 0.005] %big go first
one_over_lambda = 1./lambda %small to big
%%
save_bool = 1
%%
test_errors = [13.79, 5.6,  2.95, 1.7, 1.1]
train_erros = [0.11, 0.03, 0.011, 0.002, 0.0005]
test_errors = 0.02*test_errors
%train_erros = log(train_erros)
%test_errors = log(test_errors)
%erm_lambda = [0.11,0.03]
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
plot_test_train_same_plot( one_over_lambda,train_erros,test_errors );
if save_bool
    saveas(fig,filename)
    saveas(fig,filename,'pdf')
end
%%
% fig = figure
% plot(one_over_lambda,train_erros)
% title('Regularization param vs train errors')
% xlabel('1/ \lambda')
% ylabel('Error')
% filename='one_over_lambda_vs_train_errors'
% saveas(fig,filename)
% saveas(fig,filename,'pdf')
% %%
% fig = figure
% plot(one_over_lambda,test_errors)
% title('Regularization param vs test errors')
% xlabel('1/ \lambda')
% ylabel('Error')
% filename='one_over_lambda_vs_test_errors'
% saveas(fig,filename)
% saveas(fig,filename,'pdf')
%%
% fig = figure
% plot(one_over_lambda,erm_lambda)
% title('Regularization param vs train+regularization errors')
% xlabel('1/ \lambda')
% ylabel('Error')
% filename='one_over_lambda_vs_erm'
% saveas(fig,filename)
% saveas(fig,filename,'pdf')