%% xaxis and label
expt_type = 'alpha';
%expt_type = 'radius_distance';
if strcmp(expt_type,'alpha')
    xaxis=interpolations;
    abcissa_label = 'convex parameter \alpha';
else % strcmp(expt_type,'radius_distance')
    xaxis=rs;
    abcissa_label = 'convex parameter \alpha';
end
%% accs
fig_accs = figure;
plot(xaxis,train_accs);hold;
plot(xaxis,test_accs);
legend('train accuracy','test accuracy')
title('accuracy vs interpolation parameters')
ylabel('accuracy');
xlabel(abcissa_label);
%% errors
fig_errors = figure;
plot(xaxis,train_errors);hold;
plot(xaxis,test_errors);
legend('train errors','test error')
title('errors vs interpolation parameters')
%% losses
fig_losses = figure;
plot(xaxis,train_losses);hold;
plot(xaxis,test_losses);
legend('train loss','test loss')
title('loss vs interpolation parameters')
saveas(fig_errors,['./fig_errors_interpolation_sj' num2str(sj)],'pdf')
saveas(fig_losses,['./fig_losses_interpolation_sj' num2str(sj)],'pdf')
saveas(fig_accs,['./fig_accs_interpolation_sj' num2str(sj)],'pdf')