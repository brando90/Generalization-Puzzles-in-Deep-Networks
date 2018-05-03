params1 = net1.named_parameters()
params2 = pert.named_parameters()
params_pert_net = net_perturbed.named_parameters()
''' '''
dict_pert = dict(params2)  # <- net with perturbations!
dict_pert_net = dict(params_pert_net)  # <- net we are updating
for param_name, W in params1:
    if param_name in params_pert_net:
        pert = dict_pert[param_name]
        dict_pert_net[param_name] = W + pert
interpolated_net.load_state_dict(dict_params_interpolated)
return interpolated_net