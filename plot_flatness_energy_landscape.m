%%
random_direction = 1;
%% plot landscape from random direction
rs = double(r_large).*interpolations;
xaxis = rs;
train_accs = all_train_accs(random_direction,:);
test_accs = all_test_accs(random_direction,:);
train_errors = all_train_errors(random_direction,:);
test_errors = all_test_errors(random_direction,:);
train_losses = all_train_losses(random_direction,:);
test_losses = all_test_losses(random_direction,:);
%% accs
% fig_accs = figure;
% plot(xaxis,train_accs);hold;
% plot(xaxis,test_accs);
% legend('train accuracy','test accuracy')
% title('accuracy vs interpolation parameters')
% ylabel('accuracy');
% xlabel('distance from minima');
%% errors
fig_errors = figure;
plot(xaxis,train_errors);hold;
plot(xaxis,test_errors);
legend('train errors','test error')
title('errors vs interpolation parameters')
ylabel('error');
xlabel('distance from minima');
%% losses
fig_losses = figure;
plot(xaxis,train_losses);hold;
plot(xaxis,test_losses);
legend('train loss','test loss')
title('loss vs interpolation parameters')
ylabel('loss');
xlabel('distance from minima');
%%
saveas(fig_errors,['./fig_errors_radius_landscape_sj' num2str(sj)],'pdf')
saveas(fig_losses,['./fig_losses_radius_landscape_sj' num2str(sj)],'pdf')
% saveas(fig_accs,['./fig_accs_radius_landscape_sj' num2str(sj)],'pdf')