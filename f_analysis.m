nb_train_acts = numel(hist_all_train_norm);
nb_test_acts = numel(hist_all_test_norm);
%% train norm
fig0 = figure;
hist(hist_all_train_norm)
hist_all_train_norm = reshape(hist_all_train_norm,[1 nb_train_acts]);
xlabel('f(x) value');
ylabel('counts/frequency');
title('histogram of f(x) for normalized net on train data MNIST');
fprintf('mean(hist_all_train_norm) = %d\n',mean(hist_all_train_norm));
fprintf('std(hist_all_train_norm) = %d\n',std(hist_all_train_norm));
%% test norm
fig1 = figure;
hist(hist_all_test_norm)
hist_all_test_norm = reshape(hist_all_test_norm,[1 nb_test_acts]);
xlabel('f(x) value');
ylabel('counts/frequency');
title('histogram of f(x) for normalized net on test data MNIST');
fprintf('mean(hist_all_test_norm) = %d\n',mean(hist_all_test_norm));
fprintf('std(hist_all_test_norm) = %d\n',std(hist_all_test_norm));
%% train unnorm
fig2 = figure;
hist(hist_all_train_un)
hist_all_train_un = reshape(hist_all_train_un,[1 nb_train_acts]);
xlabel('f(x) value');
ylabel('counts/frequency');
title('histogram of f(x) for unnormalized net on train data MNIST');
fprintf('mean(hist_all_train_un) = %d\n',mean(hist_all_train_un));
fprintf('std(hist_all_train_un) = %d\n',std(hist_all_train_un));
%% test unnorm
fig3 = figure;
hist(hist_all_test_un)
hist_all_test_un = reshape(hist_all_test_un,[1 nb_test_acts]);
xlabel('f(x) value');
ylabel('counts/frequency');
title('histogram of f(x) for unnormalized net on test data MNIST');
fprintf('mean(hist_all_test_un) = %d\n',mean(hist_all_test_un));
fprintf('std(hist_all_test_un) = %d\n',std(hist_all_test_un));
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