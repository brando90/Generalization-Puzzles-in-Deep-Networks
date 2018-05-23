%% plot SGD training path
xaxis = 1:length(test_errors);
fig_errors = figure;
plot(xaxis,train_errors);hold;
plot(xaxis,test_errors);
legend('train errors','test error')
title('errors vs epochs')
fig_losses = figure;
plot(xaxis,train_losses);hold;
plot(xaxis,test_losses);
legend('train loss','test loss')
title('losses vs epochs')
%saveas(fig_errors,['./fig_errors_seed' num2str(seed)],'pdf')
%saveas(fig_losses,['./fig_losses_seed' num2str(seed)],'pdf')
nb_epochs
hours
nb_params
train_errors(end)