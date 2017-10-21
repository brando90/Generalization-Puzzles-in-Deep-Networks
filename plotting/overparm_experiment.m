clear;clc;
%% parameters for script
lb=-1;ub=1;
D0=1;
N_train=32;
N_test=500;
%% get X
X_train = linspace(lb,ub,N_train);
X_test = linspace(lb,ub,N_test);
Degree_data_set = 25;
nb_monomials_data = nchoosek(D0+Degree_data_set,Degree_data_set)
%% target function
freq=1.5;
f_target = @(x) cos(freq*2*pi*x);
Y_test = f_target(X_test);
[p,S] = polyfit(X_test,Y_test,Degree_data_set); % p=[1,deg+1], returns in descending order [x^n, ... , 1]
%%
figure
%plot(X_test,Y_test,'o');hold;
plot(X_test,polyval(p,X_test));hold;
%plot(X_test,Y_test);
%% degrees to test



%%