clc;clear;
%% script folder
dot = '/Users/brandomiranda/home_simulation_research/overparametrized_experiments/';
cd(dot);
%% Get perturbation errors for natural label experiment
path_natural_label='/pytorch_experiments/test_runs_flatness/flatness_29_March_label_corrupt_prob_0.0_exptlabel_pert_retrained_Random_Label_on_Natural_Label_BoixNet';
expt_path = fullfile(dot,path_natural_label);
results_natural_label = get_results(expt_path);
%% Get perturbation errors for random label experiment
path_random_label='/pytorch_experiments/test_runs_flatness/flatness_29_March_label_corrupt_prob_0.0_exptlabel_pert_Natural_Label_BoixNet';
expt_path = fullfile(dot,path_natural_label);
results_random_label = get_results(expt_path);
%% Plot Results for random label
noises = cell2list( keys(results_random_label.M_tests_accs) );
[means_test, stds_test] = get_stats_experiment(results_random_label.M_tests_accs);
[means_train, stds_train] = get_stats_experiment(results_random_label.M_train_accs);
figure;
errorbar(noises,means_test,stds_test)
figure;
errorbar(noises,means_train,stds_train)
%% return to script folder
cd(dot)
%% Helper function
function results = get_results(expt_path)
    results = struct();
    M_tests_accs = containers.Map('KeyType','double','ValueType','any');
    M_train_accs = containers.Map('KeyType','double','ValueType','any');
    M_tests_losses = containers.Map('KeyType','double','ValueType','any');
    M_train_losses = containers.Map('KeyType','double','ValueType','any');
    %%
    cd(expt_path)
    files = dir('noise*');
    directoryNames = {files([files.isdir]).name};
    for expt_dir_name = directoryNames
        name_expt_dir = expt_dir_name{1};
        noise = str2double(name_expt_dir(7:end)); % extract the noise from the name
        %%
        all_test_accs = [];
        all_train_accs = [];
        all_test_losses = [];
        all_train_lossses = [];
        %%
        noise_dir_path = fullfile(name_expt_dir,'noise*');
        files = dir(noise_dir_path);
        files = {files.name};
        for expt_file_name = files
            noise_matlab_file = fullfile(name_expt_dir,expt_file_name{1});
            load(noise_matlab_file);
            all_test_accs = [all_test_accs, test_accs];
            all_train_accs = [all_train_accs, train_accs];
            all_test_losses = [all_test_losses, test_losses];
            all_train_lossses = [all_train_lossses, train_losses];
        end
        M_tests_accs(noise) = all_test_accs;
        M_train_accs(noise) = all_train_accs;
        M_tests_losses(noise) = all_test_losses;
        M_train_losses(noise) = all_train_lossses;
    end
    results.M_tests_accs = M_tests_accs;
    results.M_train_accs = M_train_accs;
    results.M_tests_losses = M_tests_losses;
    results.M_train_losses = M_train_losses;
end
function [means, stds] = get_stats_experiment(M)
    noises = keys(M);
    vals = values(M);
    means = length(noises);
    stds = length(noises);
    for i=length(noises)
        mu = mean(vals{1});
        stddev = std(vals{1});
        means(i) = mu;
        stds(i) = stddev;
    end
end
function list = cell2list(M)
    list = zeros(1,length(M));
    for i = 1:length(M)
        m = M(i);
        list(i) = m{1};
    end
end