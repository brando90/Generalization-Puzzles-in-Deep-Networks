clear;
%%
prefix_fname='experiment_lambdas';
filename = strcat(prefix_fname,'.mat');
load(filename)
%%
if strcmp(filename,'experiment_lambdas.mat')
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
    disp('TODO')
    %%
%     x_axis = iterations;
%     train_errors = train_means;
%     train_errors_bars = train_stds;
    %% TODO
end
%%
x_axis
train_errors
train_errors_bars
test_errors
test_errors_bars
%% SUBPLOTS
fig = figure;
subplot(2,1,1)
errorbar(x_axis,train_errors,train_errors_bars,'-bx','MarkerEdgeColor','blue')
title(title_name_train)
xlabel(xlabel_name);ylabel(ylabel_name);
legend('train');

subplot(2,1,2)
errorbar(x_axis,test_errors,test_errors_bars,'-rx','MarkerEdgeColor','red')
title(title_name_test)
xlabel(xlabel_name);ylabel(ylabel_name)
legend('test');

%%
saveas(fig,strcat('fig_',prefix_fname))
saveas(fig,strcat('fig_',prefix_fname),'pdf')