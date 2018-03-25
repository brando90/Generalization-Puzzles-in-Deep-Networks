import pytorch_over_approx_high_dim as my_pytorch

import unittest

#SLURM_ARRAY_TASK_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])

def get_lambda_to_run(hyper_params,repetitions,satid):
    start_next_bundle_batch_jobs=1
    for job_number in range(1,len(hyper_params)):
        start_next_bundle_batch_jobs+= job_number*repetitions[job_number]
        if start_next_bundle_batch_jobs > SLURM_ARRAY_TASK_ID:
            return hyper_params[job_number-1]
    raise ValueError('There is something wrong with the number of jobs you submitted compared.')

def main():
    ## config params
    N_lambdas = 3
    lb,ub=1,3
    lambdas = np.linspace(lb,ub,N_lambdas)
    repetitions = len(lambdas)*[2]

    ##
    nb_iterations = 10000

    ##dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
    dtype = torch.FloatTensor
    #
    debug = True
    debug_sgd = False
    ## Hyper Params SGD weight parametrization
    M = 3
    eta = 0.002 # eta = 1e-6
    if 'nb_iterations_WP' in kwargs:
        nb_iter = kwargs['nb_iterations_WP']
    else:
        nb_iter = int(80*1000)
    #nb_iter = int(15*1000)
    A = 0.0
    if 'reg_lambda_WP' in kwargs:
        reg_lambda_WP = kwargs['reg_lambda_WP']
    else:
        reg_lambda_WP = 0.0
    ##
    logging_freq = 100
    #### Get Data set
    ##
    truth_filename='data_gen_type_mdl=WP_D_layers_[3, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_8_N_test_20_lb_-1_ub_1_act_poly_act_degree2_nb_params_5_msg_'
    data_filename='data_numpy_type_mdl=WP_D_layers_[3, 1, 1]_nb_layers3_bias[None, True, False]_mu0.0_std5.0_N_train_8_N_test_20_lb_-1_ub_1_act_poly_act_degree2_nb_params_5_msg_.npz'
    if truth_filename is not None:
        mdl_truth_dict = torch.load('./data/'+truth_filename)
        D_layers_truth=extract_list_filename(truth_filename)
    ##
    data = np.load( './data/{}'.format(data_filename) )
    X_train, Y_train = data['X_train'], data['Y_train']
    #X_train, Y_train = X_train[0:6], Y_train[0:6]
    X_test, Y_test = data['X_test'], data['Y_test']
    D_data = X_test.shape[1]
    ## get nb data points
    N_train,_ = X_train.shape
    N_test,_ = X_test.shape
    ## activation params
    #adegree = 2
    #alb, aub = -100, 100
    #aN = 100
    #act = get_relu_poly_act(degree=adegree,lb=alb,ub=aub,N=aN) # ax**2+bx+c
    ##
    adegree = 2
    ax = np.concatenate( (np.linspace(-20,20,100), np.linspace(-10,10,1000)) )
    aX = np.concatenate( (ax,np.linspace(-2,2,100000)) )
    act, c_pinv_relu = get_relu_poly_act2(aX,degree=adegree) # ax**2+bx+c, #[1, x^1, ..., x^D]
    print('c_pinv_relu = ', c_pinv_relu)
    #act = relu
    #act = lambda x: x
    #act.__name__ = 'linear'
    ## plot activation
    # palb, paub = -20, 20
    # paN = 1000swqb
    # print('Plotting activation function')
    # plot_activation_func(act,lb=palb,ub=paub,N=paN)
    # plt.show()
    #### 2-layered mdl
    D0 = D_data

    H1 = 12
    D0,D1,D2 = D0,H1,1
    D_layers,act = [D0,D1,D2], act

    # H1,H2 = 20,20
    # D0,D1,D2,D3 = D0,H1,H2,1
    # D_layers,act = [D0,D1,D2,D3], act

    # H1,H2,H3 = 15,15,15
    # D0,D1,D2,D3,D4 = D0,H1,H2,H3,1
    # D_layers,act = [D0,D1,D2,D3,D4], act

    # H1,H2,H3,H4 = 25,25,25,25
    # D0,D1,D2,D3,D4,D5 = D0,H1,H2,H3,H4,1
    # D_layers,act = [D0,D1,D2,D3,D4,D5], act

    nb_layers = len(D_layers)-1 #the number of layers include the last layer (the regression layer)
    biases = [None] + [True] + (nb_layers-1)*[False] #bias only in first layer
    #biases = [None] + (nb_layers)*[True] # biases in every layer
    #pdb.set_trace()
    start_time = time.time()
    ##
    np.set_printoptions(suppress=True)
    lb, ub = -1, 1
    ## mdl degree and D
    nb_hidden_layers = nb_layers-1 #note the last "layer" is a summation layer for regression and does not increase the degree of the polynomial
    Degree_mdl = adegree**( nb_hidden_layers ) # only hidden layers have activation functions
    ## Lift data/Kernelize data
    poly_feat = PolynomialFeatures(degree=Degree_mdl)
    Kern_train = poly_feat.fit_transform(X_train)
    Kern_test = poly_feat.fit_transform(X_test)
    ## LA models
    c_pinv = np.dot(np.linalg.pinv( Kern_train ),Y_train)
    ## inits
    init_config = Maps( {'w_init':'w_init_normal','mu':0.0,'std':0.01, 'bias_init':'b_fill','bias_value':0.01,'biases':biases ,'nb_layers':len(D_layers)} )
    w_inits_sgd, b_inits_sgd = get_initialization(init_config)
    init_config_standard_sgd = Maps( {'mu':0.0,'std':0.001, 'bias_value':0.01} )
    mdl_stand_initializer = lambda mdl: lifted_initializer(mdl,init_config_standard_sgd)
    ## SGD models
    if truth_filename:
        mdl_truth = NN(D_layers=D_layers_truth,act=act,w_inits=w_inits_sgd,b_inits=b_inits_sgd,biases=biases)
        mdl_truth.load_state_dict(mdl_truth_dict)
    mdl_sgd = NN(D_layers=D_layers,act=act,w_inits=w_inits_sgd,b_inits=b_inits_sgd,biases=biases)
    mdl_standard_sgd = get_sequential_lifted_mdl(nb_monomials=c_pinv.shape[0],D_out=1, bias=False)
    ## data to TORCH
    data = get_data_struct(X_train,Y_train,X_test,Y_test,Kern_train,Kern_test,dtype)
    data_stand = get_data_struct(X_train,Y_train,X_test,Y_test,Kern_train,Kern_test,dtype)
    data_stand.X_train, data_stand.X_test = data_stand.Kern_train, data_stand.Kern_test
    ## DEBUG PRINTs
    print('>>norm(Y): ', ((1/N_train)*torch.norm(data.Y_train)**2).data.numpy()[0] )
    print('>>l2_loss_torch: ', (1/N_train)*( data.Y_train - mdl_sgd.forward(data.X_train)).pow(2).sum().data.numpy()[0] )
    ## check number of monomials
    nb_monomials = int(scipy.misc.comb(D0+Degree_mdl,Degree_mdl))
    if c_pinv.shape[0] != int(scipy.misc.comb(D0+Degree_mdl,Degree_mdl)):
       raise ValueError('nb of monomials dont match D0={},Degree_mdl={}, number of monimials fron pinv={}, number of monomials analyticall = {}'.format( D0,Degree_mdl,c_pinv.shape[0],int(scipy.misc.comb(D0+Degree_mdl,Degree_mdl)) )    )
    ########################################################################################################################################################
    if 'reg_type_wp' in kwargs:
        reg_type_wp = kwargs['reg_type_wp']
    else:
        reg_type_wp = 'tikhonov'
    print('reg_type_wp = ', reg_type_wp)
    ##
    arg = Maps(reg_type=reg_type_wp)
    keep_training=True
    while keep_training:
        try:
            train_loss_list_WP,test_loss_list_WP,grad_list_weight_sgd,func_diff_weight_sgd,erm_lamdas_WP,nb_module_params = train_SGD(
                arg,mdl_sgd,data, M,eta,nb_iter,A ,logging_freq ,dtype,c_pinv, reg_lambda_WP
            )
            keep_training=False
        except ValueError:
            print('Nan was caught, going to restart training')
            w_inits_sgd, b_inits_sgd = get_initialization(init_config)
            mdl_sgd = NN(D_layers=D_layers,act=act,w_inits=w_inits_sgd,b_inits=b_inits_sgd,biases=biases)

class TestStringMethods(unittest.TestCase):

    def test_get_lambda_to_run(self):
        hyper_params = [1,2,3]
        repetitions = [5,5,5]
        satid = 1
        for hp_i in range(1,len(hyper_params)):
            for r_i in range(1,len(repetitions)):
                hyper_param = get_lambda_to_run(hyper_params,repetitions,satid)
                self.assertEqual(hyper_param,hyper_params[satid])
                satid+=1

if __name__ == '__main__':
    #main()
    unittest.main()
