function [ fig ] = plot_test_train_same_plot( one_over_lambda,train_erros,test_errors )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
fig = figure;
plot(one_over_lambda,train_erros)
hold on
plot(one_over_lambda,test_errors)
legend('train','test')

% title('iterations vs train,test errors')
% xlabel('iterations')
% ylabel('Error')
% filename='iterations_vs_train_test_same_plot'
% saveas(fig,filename)
% saveas(fig,filename,'pdf')

title('1/\lambda vs train,test errors')
xlabel('1/\lambda')
ylabel('Error')
filename='one_over_lambda_vs_train_test_same_plot'
saveas(fig,filename)
saveas(fig,filename,'pdf')



end

