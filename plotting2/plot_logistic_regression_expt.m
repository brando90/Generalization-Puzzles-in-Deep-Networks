path='/Users/brandomiranda/home_simulation_research/overparametrized_experiments/pytorch_experiments2/test_runs_debug/unit_logistic_regression/N_train_81_N_test_121_batch_size_2_perturb_freq_1000_perturb_magnitude_0_momentum_0.99/expt_type_NB_VEC_ELEMENTS_1/satid_1_sid_1_February_9.mat';
load(path);
iterations=1:length(w_norms);
%% plot w_norms
fig_w=figure;
plot(iterations,w_norms)
legend('||w_1||','||w_2||')
title('Weight norm vs iterations (for both classes)')
%% losses
fig_loss=figure;
plot(iterations,[train_losses;test_losses])
legend('train loss','test loss')
title('train/test loss vs iterations)')
%% errors
fig_error=figure;
plot(iterations,[test_errors;train_errors])
legend('train error','test error')
title('train/test error vs iterations)')
%%
saveas(fig_w,'fig_w')
saveas(fig_w,'fig_w','pdf')
saveas(fig_loss,'fig_loss')
saveas(fig_loss,'fig_loss','pdf')
saveas(fig_error,'fig_error')
saveas(fig_error,'fig_error','pdf')