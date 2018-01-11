%path = '../pytorch_experiments/test_runs/unit_const_noise_pert_expt_fig13_reps1_reg__expt_type_SP_fig4_N_train_9_M_9_frac_norm_0.0_logging_freq_1_perturbation_freq_4000/fig4_expt_lambda_0_it_250000'
%path = '../pytorch_experiments/test_runs/const_noise_pert_expt_fig13_reps1_reg__expt_type_SP_fig4_N_train_9_M_9_frac_norm_0.0_logging_freq_1_perturbation_freq_4000/fig4_expt_lambda_0_it_250000'
%path = '../pytorch_experiments/test_runs/const_noise_pert_expt_fig13_reps1_reg__expt_type_SP_fig4_N_train_9_M_9_frac_norm_0.0_logging_freq_1_perturbation_freq_4000/fig4_expt_lambda_0_it_750000'
%path = '../pytorch_experiments/test_runs/unit_trig_pert_expt_fig13_reps1_reg__expt_type_SP_fig4_N_train_25_M_25/fig4_expt_lambda_0_it_500000'
%path = '../pytorch_experiments/test_runs/unit_trig_pert_expt_fig13_reps1_reg__expt_type_SP_fig4_N_train_25_M_25_frac_norm_0.0_logging_freq_1_perturbation_freq_4000/fig4_expt_lambda_0_it_500000'
path = '../pytorch_experiments/test_runs/unit_trig_pert_expt_fig13_reps1_reg__expt_type_SP_fig4_N_train_25_M_25_frac_norm_0.0_logging_freq_1_perturbation_freq_4000/fig4_expt_lambda_0_it_5000000'
directory_names = dir([path '/deg_*']); % [deg_1, ..., deg_N]
nb_terms = 100;
%%
train_errors = zeros(1,nb_terms);
test_errors = zeros(1,nb_terms);
directory_names
for expt = directory_names' % expt.name = [deg_1, ..., deg_N]
    expt.name
    specific_simulation_loc = [path '/' expt.name '/' 'satid_*'];
    specific_simulation = dir(specific_simulation_loc); % [staid_sim1,...,staid_simK] 
    for simulation = specific_simulation' % simulation.name = [staid_sim1,...,staid_simK] 
        simulation.name
        [path '/' expt.name '/' simulation.name]
        load([path '/' expt.name '/' simulation.name]);
        %%
%         fig = figure;
%         plot(1:length(train_loss_list_WP),train_loss_list_WP)
%         hold on;
%         plot(1:length(test_loss_list_WP),test_loss_list_WP)
%         legend('train','test')
%         title(['Degree mdl ' num2str(Degree_mdl)])
        %%
%         train_errors(Degree_mdl) = train_error_WP
%         test_errors(Degree_mdl) = test_error_WP;
        train_errors(Degree_mdl) = train_loss_list_WP(end);
        test_errors(Degree_mdl) = test_loss_list_WP(end);
    end
end
fig = figure;
plot(1:nb_terms,train_errors)
hold on;
plot(1:nb_terms,test_errors)
legend('train','test')
title('train/test vs degree trig')
xlabel('# of degree trig')
ylabel('L2 error')
saveas(fig,'error_vs_params','fig');
saveas(fig,'error_vs_params','pdf');