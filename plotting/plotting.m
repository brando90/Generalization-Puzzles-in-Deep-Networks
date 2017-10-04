clear;
%%
load('experiment.mat')
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
% if save_bool
%     saveas(fig,filename)
%     saveas(fig,filename,'pdf')
% end
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