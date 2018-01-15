function [ ] = run_GDL_wedge_perturbations( SLURM_JOBID,SLURM_ARRAY_TASK_ID,print_hist )
%% computation time params
D = 2;
nbins = 100;
c = 250000;
%c = 100;
iter = c;
%iter = c*nbins^D;
%% energy landscape
lb = -8;ub = 8;
%N = 200;
%K = 4000; % number of centers wedge
% get centers
%i_coord = 2;
%offset_i_coord = -1;
%t = get_centers(K,D,i_coord,offset_i_coord+1,lb-2.5,ub+2.5); % K x D
%tt = sum(t.^2,2)';
% get C's weights
%C = -1*ones(K,1)/575;
% get Gaussian precision
stddev = 2.0;
beta = 1/(2*stddev^2);
%% params of loss surface
%params = struct('t',t,'tt',tt,'C',C,'beta',beta);
%% RBF N batch
%ind_mini_batch = ones(length([C;C_p]),1);
%f_N_batch = @(x) f_batch_new(x,ind_mini_batch,params);
%ind_mini_batch = ones(1,K);
%f = @(x) f_batch_new_wedge(x,ind_mini_batch,params);
f = @(x) -exp(-beta*x(2)^2);
%% GDL & mdl params
g_eps = 0.00000001;
eta = 0.04;
B = 12;
%
%A = 0.05;
A = 0;
gdl_mu_noise = 0.0;
gdl_std_noise = 1.0;
%% init
%W = 0.0*ones(1,D);
mu_init=0;
std_init=0.05;
W = mu_init+std_init*randn(1,D);
%W = [0,0];
%W = [0,4];
%% histogram
%filename = sprintf('current_gdl_run_%dD_A%.2d',D,A);
save_figs = 1;
edges = linspace(0,B,nbins);
Normalization = 'probability';
%Normalization = 'pdf';
%Normalization = 'count';
%%
datetime('now')
tic
%% perturbation
mu_pert = 0.0;
%frac_norm = 2.3;
std_pert = 1.6;
perturbation_freq = 1100;
%%
train_errors = zeros(iter+1,1);
W_history = zeros(iter+1,D);
g_history = zeros(iter+1,D);
w_norms = zeros(iter+1,1);
%%
w_norms(1) = norm(W,2);
train_errors(1,:) = f(W);
W_history(1,:) = W;
W_hist_counts = zeros(size(edges)-[0,1]);
for i=2:iter+1
    %% gradient
    %g = get_gradient(W,mu1,std1,mu2,std2);
    g = CalcNumericalFiniteDiff(W,f,g_eps);
    %eps = normrnd(gdl_mu_noise,gdl_std_noise,[1,D]);
    %% SGD update
    %W = mod(W - eta*g, B);
    %W = mod(W - eta*g + A*eps, B);
    %W = W - eta*g + A*eps;
    W = W - eta*g;
    %% collect stats
    train_errors(i) = f(W);
    w_norms(i) = norm(W,2);
    %2D
    W_history(i,:) = W;
    g_history(i,:) = g;
    [W_hist_counts_current, edges2] = histcounts(W,edges);
    W_hist_counts = W_hist_counts + W_hist_counts_current;
    %% perturb
    if mod(i,perturbation_freq) == 0
        %pert_noise = mu_pert + (frac_norm*norm(W,2))*randn(size(W))
        pert_noise = mu_pert + std_pert*randn(size(W));
        %pert_noise = mu_pert + std_pert*rand(size(W))-0.5;
        W = W + pert_noise;
    end
end
elapsedTime = toc;
fprintf('D: %d, nbins: %f, c: %f, iter=c*nbins^D=%d*%d^%d = %d \n',D,nbins,c, c,nbins,D, iter);
fprintf('elapsedTime %f seconds, %f minutes, %f hours \n', elapsedTime,elapsedTime/60,elapsedTime/(60*60));
%W_history
%%
%dir_path = './test_runs/unit_test_job_name_iter_'
%dir_path = './test_runs/unit_OM_test_job_name_iter_'
dir_path = './test_runs/job_name_iter_'
dir_path = [dir_path num2str(iter) '_eta_' num2str(eta) '_mu_pert_' num2str(mu_pert) '_std_pert_' num2str(std_pert) 'perturbation_freq' num2str(perturbation_freq) ]
mkdir(dir_path)
filename = [ dir_path '/GDL_pert' '_jid_' num2str(SLURM_JOBID) '_satid_' num2str(SLURM_ARRAY_TASK_ID) ];
save(filename)
%% Visualize simulation
if print_hist
    %% visualize landscape
    fig = visualize_surf_single(f,100,lb,ub);title('Energy Landscape');
    fname = 'energy_landscape';
    saveas(fig,fname);saveas(fig,fname,'pdf');
    %%
    fig = figure;
    plot(1:iter+1,train_errors)
    title('Train error vs Iteration');
    fname = 'train_errors';
    saveas(fig,fname);saveas(fig,fname,'pdf');
    %%
    fig = figure;
    plot(1:iter+1,w_norms)
    title('Norm of Weights ||W||');
    fname = 'norm of Weights';
    saveas(fig,fname);saveas(fig,fname,'pdf');
    if D==2
        fig = figure;
        hist3(W_history,[nbins,nbins]);
        ylabel('Weight W_2')
        xlabel('Weight W_1')
        zlabel(sprintf('Normalization: %s',Normalization))
%         if strcmp(Normalization, 'count') == 0
%             zlim([0,1])
%         end
        f = sprintf('W_%dD',D);
        saveas(fig,f)
        saveas(fig,f,'pdf')
    end
    for d=1:D
        %normalizations = {'count','pdf','probability'};
        normalizations = {'probability'};
        for i=1:length(normalizations) % goes through the types of plots
            Normalization = normalizations{i};
            fig = figure;histogram(W_history(:,d),nbins,'Normalization',Normalization);
            xlabel('Weights');ylabel(sprintf('%s',Normalization))
            title(sprintf('Histogram of W_%d for %d D experiment',d,D));
            if strcmp(Normalization, 'probability')
                ylim([0,1]);
            elseif strcmp(Normalization, 'pdf')
                ylim([0,3]);
            end
            if save_figs
                fname = sprintf('W%d_%dD_A%.3f_%s',d,D,A,Normalization);
                fname = strrep(fname,'.','p');
                saveas(fig,fname);saveas(fig,fname,'pdf');
            end
        end
    end
end
%%
beep;
end

