function [ output_args ] = two_plots( one_over_lambda,train_erros,test_errors )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
fig = figure
subplot(2,1,1)
plot(one_over_lambda,train_erros)
title('1/ \lambda vs train errors')
xlabel('1/ \lambda')
ylabel('Error')

subplot(2,1,2)
plot(one_over_lambda,test_errors)
title('1/ \lambda vs test errors')
xlabel('1/ \lambda')
ylabel('Error')
filename='one_over_lambda_vs_test_errors'

end

