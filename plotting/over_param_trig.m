%% Generate the data
N_train=76;
N_test=600;
%D=1;
%
lb=-1;
ub=1;
%x_train = linspace(lb,ub,N_train);
x_train = get_chebyshev_nodes(lb,ub,N_train);
y_train = sin(2*pi*4*x_train);
%y_train = 10*x_train.^3 + 5*x_train.^2 + x_train + 1;
y_train = y_train';
%x_test = 1:N_test';
x_test = linspace(lb,ub,N_test);
y_test = sin(2*pi*4*x_test);
%y_test = 10*x_test.^3 + 5*x_test.^2 + x_test + 1;
y_test = y_test';
%% get all the models from LOW to HIGH complexity
lb_deg = 1;
ub_deg = 300;
%
degrees = lb_deg:2:ub_deg;
[train_errors,test_errors] = get_models(x_train,x_test,y_train,y_test, degrees);
%% Plot
fig = figure;
%plot(degrees,train_errors,'-*');
%plot(degrees+1,train_errors(lb_deg:ub_deg));
vline( double(N_train),'--g','# Training data');
hold on;
plot(degrees+1,train_errors);
hold on;
%plot(degrees,test_errors,'-*');
%plot(degrees+1,test_errors(lb_deg:ub_deg));
plot(degrees+1,test_errors);
title('train/test vs parameters/monomials')
legend('train','test')
ylabel('L2 error')
xlabel('# of parameters/monomials')
%ylim(2);
%xlim([lb_deg+1 ub_deg+1]);
saveas(fig,'over_param_plot','fig');
saveas(fig,'over_param_plot','pdf');
%% polys
function [train_errors,test_errors] = get_models(x_train,x_test,y_train,y_test, degrees)
    N_train = length(x_train);
    N_test = length(x_test);
    %%
    train_errors = zeros(0);
    test_errors = zeros(0);
    %%
    for deg = degrees
        deg
        %Kern_train = hermite_features(x_train,deg);
        %Kern_test = hermite_features(x_test,deg);
        Kern_train = polynomial_features(x_train,deg);
        Kern_test = polynomial_features(x_test,deg);
        %Kern_train = trig_kernel_matrix(x_train,deg);
        %Kern_test = trig_kernel_matrix(x_test,deg);
        % Kw = y =? w = K^+y
        %w_train_min_norm = Kern_train\y_train;
        w_train_min_norm = pinv(Kern_train)*y_train;
        %w_train_min_norm = lsqminnorm(Kern_train,y_train);
        %%
        %train_errors(deg) = (1/N_train)*norm(Kern_train*w_train_min_norm - y_train)^2;
        %test_errors(deg) = (1/N_test)*norm(Kern_test*w_train_min_norm - y_test)^2;
        train_errors =[train_errors (1/N_train)*norm(Kern_train*w_train_min_norm - y_train)^2];
        test_errors = [test_errors (1/N_test)*norm(Kern_test*w_train_min_norm - y_test)^2];
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
function poly_feat = hermite_features(x,degree)
    N=length(x);
    D=degree+1;
    poly_feat = zeros(N,D);
    for n = 1:N
        poly_feat(n,:) = hermiteH(0:degree,x(n));
    end
end
function chebyshev_nodes = get_chebyshev_nodes(lb,ub,N)
    k = 1:N+1;
    chebyshev_nodes = 0.5*(lb+ub)+0.5*(ub-lb)*cos((pi*2*k-1)/(2*N));
end
function Kern = trig_kernel_matrix(x,deg)
    N = length(x);
    Kern = zeros(N,2*deg+1);
    for n = 1:N
        for d = 1:deg+1
            Kern(n,d) = cos((d-1)*x(n));
        end
        for d = 1:deg
            Kern(n,deg+d) = sin(d*x(n));
        end
    end
end