%% get path to folder
dot = '/Users/brandomiranda/home_simulation_research/overparametrized_experiments/';
path = fullfile(dot,'pytorch_experiments/test_runs_flatness');
%NL
path_all_expts = fullfile(path,'flatness_4_April_label_corrupt_prob_0.0_exptlabel_RadiusFlatnessNL_samples15_RLarge50')
%RLNL
%path_all_expts = fullfile(path,'flatness_4_April_label_corrupt_prob_0.0_exptlabel_RadiusFlatnessRLNL_samples15_RLarge50')
%% go through files
eps = 0.05;
%%
path_plus_prefix_of_all_expts = fullfile(path_all_expts,'/flatness_*');
files = dir(path_plus_prefix_of_all_expts);
expt_data_filenames = {expt_data_files.name};
for expt_file_name = expt_data_file_names
end
cd(dot)c