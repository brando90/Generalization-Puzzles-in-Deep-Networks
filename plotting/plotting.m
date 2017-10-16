clear;
disp('--------------')
%%
prefix_fname='experiment_iter_oct7_9529777';
%prefix_fname='experiment_iter_oct7_9530606';
%experiment_iter_tmp.mat
%prefix_fname='experiment_iter_tmp';
filename = [prefix_fname '.mat']
load( ['./results/' filename])
%%
if contains(filename,'lambda')
    title_name_train = '1/ \lambda vs train errors';
    title_name_test = '1/ \lambda vs test errors';
    xlabel_name = '1/ \lambda';
    ylabel_name = 'Error';
    %%
    x_axis = one_over_lambdas;
    train_errors = train_means;
    train_errors_bars = train_stds;
    test_errors = test_means;
    test_errors_bars = test_stds;
else
    title_name_train = 'iterations vs train errors';
    title_name_test = 'iterations vs test errors';
    xlabel_name = 'iterations';
    ylabel_name = 'Error';
    %%
    x_axis = iterations;
    train_errors = train_means;
    train_errors_bars = train_stds;
    test_errors = test_means;
    test_errors_bars = test_stds;
end
%%
fprintf('size(x_axis) = %d %d', size(x_axis))
x_axis
train_errors
train_errors_bars
test_errors
test_errors_bars
%% SUBPLOTS
fig = figure;
fig.PaperPositionMode = 'auto';
subplot(2,1,1)
errorbar(x_axis,train_errors,train_errors_bars,'-bx','MarkerEdgeColor','blue')
title(title_name_train)
xlabel(xlabel_name);ylabel(ylabel_name);
legend('train');
%ylim([0 1.5*10^-3])

subplot(2,1,2)
errorbar(x_axis,test_errors,test_errors_bars,'-rx','MarkerEdgeColor','red')
title(title_name_test)
xlabel(xlabel_name);ylabel(ylabel_name)
legend('test');
%ylim([0.2 0.6])

%%
saveas(fig,strcat('fig_',prefix_fname))
saveas(fig,strcat('fig_',prefix_fname),'pdf')