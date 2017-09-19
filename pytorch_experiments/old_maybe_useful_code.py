def old_SGD_update():
    ## SGD update
    for W in mdl_sgd.parameters():
        gdl_eps = torch.randn(W.data.size()).type(dtype)
        #clip=0.001
        #torch.nn.utils.clip_grad_norm(mdl_sgd.parameters(),clip)
        #delta = torch.clamp(eta*W.grad.data,min=-clip,max=clip)
        #print(delta)
        #W.data.copy_(W.data - delta + A*gdl_eps)
        delta = eta*W.grad.data
        W.data.copy_(W.data - delta + A*gdl_eps) # W - eta*g + A*gdl_eps

def L2_norm_2(f,g,lb=0,ub=1,D=1):
    if D==1:
        f_g_2 = lambda x: (f(x) - g(x))**2
        result = integrate.quad(func=f_g_2, a=lb,b=ub)
        integral_val = result[0]
    elif D==2:
        gfun,hfun = lambda x: -1, lambda x: 1
        def f_g_2(x,y):
            #pdb.set_trace()
            x_vec = np.array([[x,y]])
            return (f(x_vec) - g(x_vec))**2
        result = integrate.dblquad(func=f_g_2, a=lb,b=ub, gfun=gfun,hfun=hfun)
        integral_val = result[0]
    else:
        raise ValueError(' D {} is not handled yet'.format(D))
    return integral_val

if D0 == 1:
    f_sgd = lambda x: f_mdl_eval(x,mdl_sgd,dtype)
    f_pinv = lambda x: f_mdl_LA(x,c_pinv)
    print('||f_sgd - f_pinv||^2_2 = ', L2_norm_2(f=f_sgd,g=f_pinv,lb=lb,ub=ub,D=1) )
    #print('||f_avg - f_pinv||^2_2 = ', L2_norm_2(f=f_avg,g=f_pinv,lb=0,ub=1))
elif D0 == 2:
    #f_sgd = lambda x: f_mdl_eval(x,mdl_sgd,dtype)
    #f_pinv = lambda x: f_mdl_LA(x,c_pinv,D_mdl=D_pinv)
    #print('||f_sgd - f_pinv||^2_2 = ', L2_norm_2(f=f_sgd,g=f_pinv,lb=lb,ub=ub,D=2))
    pass
else:
    pass

def get_RLS_soln( X,Y,lambda_rls):
    N,D = X.shape
    XX_lI = np.dot(X.transpose(),X) + lambda_rls*N*np.identity(D)
    w = np.dot( np.dot( np.linalg.inv(XX_lI), X.transpose() ), Y)
    return w

if i % logging_freq == 0 or i == 0:
    current_loss = (1/N_train)*(mdl_sgd.forward(X) - Y).pow(2).sum().data.numpy()
    #current_loss = loss.data.numpy()[0]
    loss_list.append(current_loss)
    if i!=0:
        if collect_functional_diffs:
            f_sgd = lambda x: f_mdl_eval(x,mdl_sgd,dtype)
            f_pinv = lambda x: f_mdl_LA(x,c_pinv,D_mdl=D_pinv)
            func_diff.append( L2_norm_2(f=f_sgd,g=f_pinv,lb=lb,ub=ub,D=2) )
        elif collect_generalization_diffs:
            y_test_sgd = mdl_sgd.forward(X_pytorch_test)
            #y_test_pinv = torch.FloatTensor( np.dot( Kern_test, c_pinv) )
            y_test_pinv = Variable( torch.FloatTensor( np.dot( Kern_test, c_pinv) ) )
            loss = (1/N_test)*(y_test_sgd - y_test_pinv).pow(2).sum()
            func_diff.append( loss.data.numpy() )
        else:
            func_diff.append(-1)
    if debug_sgd:
        print('\ni =',i)
        print('current_loss = ',current_loss)
    for index, W in enumerate(mdl_sgd.parameters()):
        grad_norm = W.grad.data.norm(2)
        delta = eta*W.grad.data
        grad_list[index].append( W.grad.data.norm(2) )
        if debug_sgd:
            print('-------------')
            print('-> grad_norm: ',grad_norm)
            #print('----> eta*grad_norm: ',eta*grad_norm)
            print('------> delta: ', delta.norm(2))
            #print(delta)
        if is_NaN(grad_norm) or is_NaN(current_loss):
            print('\n----------------- ERROR HAPPENED')
            print('error happened at: i = {}'.format(i))
            print('current_loss: {}, grad_norm: {},\n -----------------'.format(current_loss,grad_norm) )
            #print('grad_list: ', grad_list)
            print('\a')
            sys.exit()

fig = plt.figure()
p_func_diff, = plt.plot(np.arange(len(func_diff)), func_diff,color='g')
if collect_functional_diffs:
    plt.legend([p_func_diff],[' L2 functional distance: SGD minus minimum norm solution'])
    plt.title('Functional L2 difference between minimum norm and SGD functions')
elif collect_generalization_diffs:
    plt.legend([p_func_diff],[' L2 generalization distance: SGD minus minimum norm solution, number test points = {}'.format(N_test)])
    plt.title('Generalization L2 difference between minimum norm and SGD functions')
else:
    raise ValueError('Plot Functional not supported.')
