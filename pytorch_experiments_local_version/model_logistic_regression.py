import torch

# def f_mdl(x,mdl):
#     w_pos = mdl[0].weight[:,0].data.numpy()
#     w_neg = mdl[0].weight[:,1].data.numpy()
#     score_pos = np.dot(w_pos,x)
#     score_pos = np.dot(w_pos,x)

def get_logistic_regression_mdl(in_features,n_classes,bias):
    '''
    '''
    mdl_sgd = torch.nn.Sequential(
        torch.nn.Linear(in_features, n_classes, bias=bias)
    )
    return mdl_sgd

def get_polynomial_logistic_reg(Degree_mdl):
    ## TODO
    poly_feat = PolynomialFeatures(degree=Degree_mdl)
    Kern_train, Kern_test = poly_feat.fit_transform(X_train), poly_feat.fit_transform(X_test) # N by D, [1,x,x^2,...,x^D]
    nb_terms = Kern_train.shape[1]
    ## get model
    bias = False # cuz the kernel/feature vector has a 1 [..., 1]
    n_classes = 2
    mdl_sgd = torch.nn.Sequential(
        torch.nn.Linear(Kern_train.shape[1], n_classes, bias=bias)
    )
    loss = torch.nn.CrossEntropyLoss(size_average=True) #TODO fix
    optimizer = torch.optim.SGD(mdl_sgd.parameters(), lr=eta, momentum=0.98) #TODO fix
    ## data to TORCH
    data = get_data_struct_classification(X_train,Y_train,X_test,Y_test,Kern_train,Kern_test,dtype_x,dtype_y)
    data.X_train, data.X_test = data.Kern_train, data.Kern_test
    ##
    nb_monomials = int(scipy.misc.comb(D0+Degree_mdl,Degree_mdl))
    ##
    legend_mdl = 'logistic_regression_mdl'
    ##
    reg_lambda = 0
    #frac_norm = 0.6
    frac_norm = 0.0
    logging_freq = 1
    perturbation_freq = 600
    ##
    c_pinv = None
    return
