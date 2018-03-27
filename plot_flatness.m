%%
xaxis = 1:length(test_errors);
figure;
plot(xaxis,train_errors);hold;
plot(xaxis,test_errors);
legend('train errors','test error')
title('errors vs epochs')
figure;
plot(xaxis,train_losses);hold;
plot(xaxis,test_losses);
legend('train loss','test loss')
title('losses vs epochs')