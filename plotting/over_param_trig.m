%% Generate the data
N_train=60;
N_test=200;
%D=1;
%
lb=-1;
ub=1;
%x_train = linspace(lb,ub,N_train);
x_train = get_chebyshev_nodes(lb,ub,N_train);
y_train = sin(2*pi*4*x_train)';
%x_test = 1:N_test';
x_test = linspace(lb,ub,N_test);
y_test = sin(2*pi*4*x_test)';
%% get all the models from LOW to HIGH complexity
lb_deg = 1;
ub_deg = 150;
%
degrees = lb_deg:ub_deg;
[train_errors,test_errors] = get_models(x_train,x_test,y_train,y_test, degrees);
%% Plot
figure;
%plot(degrees,train_errors,'-*');
plot(degrees,train_errors);
hold on;
%plot(degrees,test_errors,'-*');
plot(degrees,test_errors);
legend('train','test')
%ylim(2);
%% helper subfunctions
function [train_errors,test_errors] = get_models(x_train,x_test,y_train,y_test, degrees)
    [N_train,~] = size(x_train);
    [N_test,~] = size(x_test);
    %%
    train_errors = zeros(1,length(degrees));
    test_errors = zeros(1,length(degrees));
    %%
    for deg = degrees
        Kern_train = polynomial_features(x_train,deg);
        Kern_test = polynomial_features(x_test,deg);
        % Kw = y =? w = K^+y
        %w_train_min_norm = Kern_train\y_train;
        w_train_min_norm = lsqminnorm(Kern_train,y_train);
        %%
        train_errors(deg) = (1/N_train)*norm(Kern_train*w_train_min_norm - y_train)^2;
        test_errors(deg) = (1/N_test)*norm(Kern_test*w_train_min_norm - y_test)^2;
    end
end
function poly_feat = polynomial_features(x,degree)
    N=length(x);
    D=degree+1;
    poly_feat = zeros(N,D);
    for n = 1:N
        for deg=0:degree
            poly_feat(n,deg+1) = x(n)^deg;
        end
    end
end
function chebyshev_nodes = get_chebyshev_nodes(lb,ub,N)
    k = 1:N+1;
    chebyshev_nodes = 0.5*(lb+ub)+0.5*(ub-lb)*cos((pi*2*k-1)/(2*N));
end