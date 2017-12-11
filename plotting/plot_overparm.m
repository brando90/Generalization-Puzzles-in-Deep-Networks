%clear;
disp('--------------')
%%
filename='overfit_param_pinv_7';
%filename='overfit_param_pinv_keep';
load( ['./results/' filename])
%%
x_axis = monomials
train_errors
test_errors
[N_train,~] = size(X_train);
title_fig = ['Training data size: ' sprintf('%d',N_train)]
%%
fig = figure;
fig.PaperPositionMode = 'auto';
title(title_fig)
%
plot(x_axis,train_errors,'-ob');
hold on;
plot(x_axis,test_errors,'-*r');
vline( double(N_train),'--g','# Training data');
vline( double(26),'--c','# monomial of target function');
legend('Training Error','Test Error')
title(title_fig);
xlabel('Number of Model Params');ylabel('Error');
%ylim([0 110])
%%
% fig = figure;
% fig.PaperPositionMode = 'auto';
% title(title_fig)
% xlabel('Number of Model Params');ylabel('Error');
% %
% subplot(2,1,1)
% plot(x_axis,train_errors,'-ob');
% vline( double(N_train),'--g','# Training data');
% 
% subplot(2,1,2)
% plot(x_axis,test_errors,'-*r');
% vline( double(N_train),'--g','# Training data');
% 
% legend('Training Error','Test Error')
%%
saveas(fig,strcat('fig_',filename))
saveas(fig,strcat('fig_',filename),'pdf')