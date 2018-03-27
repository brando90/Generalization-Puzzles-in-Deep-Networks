path='/Users/brandomiranda/home_simulation_research/overparametrized_experiments/pytorch_experiments/test_runs_flatness/flatness_26_March_label_corrupt_prob_0_sj_10406481'
%%
files = dir(path);
i=0;
for file = files'
    csv = load(file.name);
    xaxis = 1:length(test_errors);
    fig_errors = figure;
    plot(xaxis,train_errors);hold;
    plot(xaxis,test_errors);
    legend('train errors','test error')
    title('errors vs epochs')
    fig_losses = figure;
    plot(xaxis,train_losses);hold;
    plot(xaxis,test_losses);
    legend('train loss','test loss')
    title('losses vs epochs')
    %%
    saveas(fig_errors,['fig_errors' 0]);
    saveas(fig_losses,['fig_losses' 1]);
end