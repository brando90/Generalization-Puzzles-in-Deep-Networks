clear;
disp('--------------')
%%
filename='overfit_param_pinv_2';
load( ['./results/' filename])
%%
x_axis = monomials
train_errors
test_errors
%%
fig = figure;
fig.PaperPositionMode = 'auto';
title(title_fig)
xlabel('Number of Model Params');ylabel('Error');
%
plot(x_axis,train_errors,'-ob');
hold on;
plot(x_axis,test_errors,'-*r');
vline( double(N_train),'--g','# Training data');
legend('Training Error','Test Error')
title(title_fig);
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