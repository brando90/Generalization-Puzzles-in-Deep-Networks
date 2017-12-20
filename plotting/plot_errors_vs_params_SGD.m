path = '../pytorch_experiments/test_runs/unit_const_noise_pert_expt_fig13_reps1_reg__expt_type_SP_fig4_N_train_9_M_9_frac_norm_0.0_logging_freq_1_perturbation_freq_4000/fig4_expt_lambda_0_it_250000'
directory_names = dir([path '/deg_*']); % [deg_1, ..., deg_N]
for expt = directory_names' % expt.name = [deg_1, ..., deg_N]
    expt.name
    specific_simulation_loc = [path '/' expt.name '/' 'satid_*'];
    specific_simulation = dir(specific_simulation_loc); % [staid_sim1,...,staid_simK] 
    for simulation = specific_simulation' % simulation.name = [staid_sim1,...,staid_simK] 
        simulation.name
    end
end