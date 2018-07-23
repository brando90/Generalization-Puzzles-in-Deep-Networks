nb_train_acts = numel(hist_all_train_norm);
nb_test_acts = numel(hist_all_test_norm);
%% train norm
fig0 = figure;
hist(hist_all_train_norm);
hist_all_train_norm2 = reshape(hist_all_train_norm,[1 nb_train_acts]);
%hist_all_train_norm2 = list_of_max_values(hist_all_train_norm);
xlabel('f(x) value');
ylabel('counts/frequency');
title('histogram of f(x) for normalized net on train data MNIST');
fprintf('mean(hist_all_train_norm) = %d\n',mean(hist_all_train_norm2));
fprintf('std(hist_all_train_norm) = %d\n',std(hist_all_train_norm2));
xlim([-0.06,0.06]);
%% test norm
fig1 = figure;
hist(hist_all_test_norm)
%hist_all_test_norm2 = reshape(hist_all_test_norm,[1 nb_test_acts]);
xlabel('f(x) value');
ylabel('counts/frequency');
title('histogram of f(x) for normalized net on test data MNIST');
fprintf('mean(hist_all_test_norm) = %d\n',mean(hist_all_test_norm2));
fprintf('std(hist_all_test_norm) = %d\n',std(hist_all_test_norm2));
xlim([-0.06,0.06]);
%%
disp(' ');
%% train unnorm
fig2 = figure;
hist(hist_all_train_un)
hist_all_train_un3 = reshape(hist_all_train_un,[1 nb_train_acts]);
xlabel('f(x) value');
ylabel('counts/frequency');
title('histogram of f(x) for unnormalized net on train data MNIST');
fprintf('mean(hist_all_train_un) = %d\n',mean(hist_all_train_un3));
fprintf('std(hist_all_train_un) = %d\n',std(hist_all_train_un3));
xlim([-150,150]);
%% test unnorm
fig3 = figure;
hist(hist_all_test_un)
hist_all_test_un4 = reshape(hist_all_test_un,[1 nb_test_acts]);
xlabel('f(x) value');
ylabel('counts/frequency');
title('histogram of f(x) for unnormalized net on test data MNIST');
fprintf('mean(hist_all_test_un) = %d\n',mean(hist_all_test_un4));
fprintf('std(hist_all_test_un) = %d\n',std(hist_all_test_un4));
xlim([-150,150]);
%% save
saveas(fig0,'hist_all_train_norm');
saveas(fig0,'hist_all_train_norm','pdf');
%
saveas(fig1,'hist_all_test_norm');
saveas(fig1,'hist_all_test_norm','pdf');
%
saveas(fig2,'hist_all_train_un');
saveas(fig2,'hist_all_train_un','pdf');
%
saveas(fig3,'hist_all_test_un');
saveas(fig3,'hist_all_test_un','pdf');
%%
function [max_values] = list_of_max_values(list)
[nb_rows,~] = size(list);
max_values = zeros(nb_rows,1);
for row=1:nb_rows
    confidence_f_x = list(row,:)
    models_largest_confidence = max(confidence_f_x)
    max_values(row) = models_largest_confidence;
end
end

