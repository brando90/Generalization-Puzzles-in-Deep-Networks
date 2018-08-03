%nb_train_acts = numel(hist_all_train_norm);
%nb_test_acts = numel(hist_all_test_norm);
%%
gamma = 0.05;
[K_gamma,max_values,second_max_values] = get_margin_based_stats(hist_all_train_norm,gamma);
margin = max_values - second_max_values;
%% plot max hist
fig1 = figure;
histogram(max_values)
xlabel('f(x) value');
ylabel('counts/frequency');
title('histogram of max f(x) for normalized net on train data MNIST');
%xlim([-0.06,0.06]);
%xlim([0,0.06]);
%% plot 2nd max hist
fig2 = figure;
histogram(second_max_values)
xlabel('f(x) value');
ylabel('counts/frequency');
title('histogram of 2nd max f(x) for normalized net on train data MNIST');
%xlim([-0.06,0.06]);
%xlim([0,0.06]);
%% plot max - 2nd max hist
fig3 = figure;
histogram(margin)
xlabel('f(x) value');
ylabel('counts/frequency');
title('histogram of margin for normalized net on train data MNIST');
%xlim([-0.06,0.06]);
%xlim([0,0.06]);
%% save
% saveas(fig3,'hist_all_test_un');
% saveas(fig3,'hist_all_test_un','pdf');
%%
function [K_gamma,max_values,second_max_values] = get_margin_based_stats(train_losses,gamma)
[nb_rows,~] = size(train_losses);
train_losses2 = repmat(train_losses,1);
max_values = zeros(nb_rows,1);
second_max_values = zeros(nb_rows,1);
for row=1:nb_rows
    %% get largest
    confidence_f_x = train_losses(row,:);
    [models_largest_confidence, max_index] = max(confidence_f_x);
    train_losses2(row,max_index) = -inf; 
    max_values(row) = models_largest_confidence;
    %% get second largest
    confidence_f_x = train_losses2(row,:);
    models_2nd_largest_confidence = max(confidence_f_x);
    second_max_values(row) = models_2nd_largest_confidence;
end
%% compute max_y f_y - max_{c!=y} f_c
K_gammas = max_values - second_max_values < gamma;
K_gamma = (1/nb_rows)*sum(K_gammas);
end