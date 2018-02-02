


def train_SGD_with_perturbations_optim(arg, mdl,data,optimizer,loss, M,eta,nb_iter,A ,logging_freq ,dtype_x,dtype_y, perturbation_freq, frac_norm):
    '''
    '''
    nb_module_params = len( list(mdl.parameters()) )
    loss_list, grad_list =  [], [ [] for i in range(nb_module_params) ]
    erm_lamdas = []
    test_loss_list = []
    func_diff = []
    w_norms = [ [] for i in range(nb_module_params) ]
    train_accs,test_accs = [],[]
    ##
    N_train, _ = tuple( data.X_train.size() )
    for i in range(0,nb_iter):
        # Reset gradient
        optimizer.zero_grad()
        # Forward pass: compute predicted Y using operations on Variables
        batch_xs, batch_ys = get_batch2(data.X_train,data.Y_train,M,(dtype_x,dtype_y)) # [M, D], [M, 1]
        ## FORWARD PASS
        y_pred = mdl(batch_xs)
        batch_loss = loss(input=y_pred,target=batch_ys)
        ## Check vectors have same dimension
        #if vectors_dims_dont_match(batch_ys,y_pred):
        #    raise ValueError('You vectors don\'t have matching dimensions. It will lead to errors.')
        ## BACKARD PASS
        batch_loss.backward() # Use autograd to compute the backward pass. Now w will have gradients
        # Update parameters
        optimizer.step()
        ## train stats
        if i % (nb_iter/10) == 0 or i == 0:
            current_train_loss = loss(input=mdl(data.X_train),target=data.Y_train).data.numpy()
            max_vals, max_indices = torch.max(mdl(data.X_train),1)
            train_acc = (max_indices == data.Y_train).sum().data.numpy()/max_indices.size()[0]
            current_test_loss = loss(input=mdl(data.X_test),target=data.Y_test).data.numpy()
            max_vals, max_indices = torch.max(mdl(data.X_test),1)
            test_acc = (max_indices == data.Y_test).sum().data.numpy()/max_indices.size()[0]
            print('-------------')
            print(f'i = {i}, current_train_loss = {current_train_loss}')
            print(f'i = {i}, train_acc = {train_acc}')
            print(f'i = {i}, current_test_loss = {current_test_loss}')
            print(f'i = {i}, test_acc = {test_acc}')
        ## stats logger
        if i % logging_freq == 0 or i == 0:
            c_pinv = mdl[0].weight
            reg_lambda = -1
            stats_logger_optim_loss(arg, mdl, loss, data, eta,loss_list,test_loss_list,grad_list,func_diff,erm_lamdas, train_accs,test_accs, i,c_pinv, reg_lambda)
            for index, W in enumerate(mdl.parameters()):
                w_norms[index].append( W.data.norm(2) )
        ## DO OP
        if i % perturbation_freq == 0 and frac_norm != 0 and i != 0:
            for W in mdl.parameters():
                #pdb.set_trace()
                Din,Dout = W.data.size()
                std = frac_norm
                noise = torch.normal(means=0.0*torch.ones(Din,Dout),std=std)
                W.data.copy_(W.data + noise)
    return loss_list,test_loss_list,grad_list,func_diff,erm_lamdas,nb_module_params,w_norms, train_accs,test_accs

def stats_logger_optim_loss(arg, mdl, loss, data, eta, loss_list,test_loss_list, grad_list, func_diff,erm_lamdas, train_accs,test_accs, i,c_pinv, reg_lambda):
    N_train,_ = tuple(data.X_train.size())
    N_test,_ = tuple(data.X_test.size())
    ## log: TRAIN ERROR
    y_pred_train = mdl(data.X_train)
    current_train_loss = loss(input=y_pred_train,target=data.Y_train).data.numpy()
    loss_list.append( float(current_train_loss) )
    ##
    y_pred_test = mdl(data.X_test)
    current_test_loss = loss(input=y_pred_test,target=data.Y_test).data.numpy()
    test_loss_list.append( float(current_test_loss) )
    ## log: GEN DIFF/FUNC DIFF
    gen_diff = -1
    func_diff.append( float(gen_diff) )
    ## ERM + regularization
    #erm_reg = get_ERM_lambda(arg,mdl,reg_lambda,X=data.X_train,Y=data.Y_train,l=2).data.numpy()
    reg = get_regularizer_term(arg, mdl,reg_lambda,X=data.X_train,Y=data.Y_train,l=2)
    erm_reg = loss(input=y_pred_test,target=data.Y_test) + reg_lambda*reg
    erm_lamdas.append( float(erm_reg.data.numpy()) )
    ##
    max_vals, max_indices = torch.max(mdl(data.X_train),1)
    train_acc = (max_indices == data.Y_train).sum().data.numpy()/max_indices.size()[0]
    train_accs.append(train_acc)
    ##
    max_vals, max_indices = torch.max(mdl(data.X_test),1)
    test_acc = (max_indices == data.Y_test).sum().data.numpy()/max_indices.size()[0]
    test_accs.append(test_acc)
    ##
    for index, W in enumerate(mdl.parameters()):
        delta = eta*W.grad.data
        grad_list[index].append( W.grad.data.norm(2) )
        if is_NaN(W.grad.data.norm(2)) or is_NaN(current_train_loss):
            print('\n----------------- ERROR HAPPENED \a')
            print('reg_lambda', reg_lambda)
            print('error happened at: i = {} current_train_loss: {}, grad_norm: {},\n ----------------- \a'.format(i,current_train_loss,W.grad.data.norm(2)))
            #sys.exit()
            #pdb.set_trace()
            raise ValueError('Nan Detected')
