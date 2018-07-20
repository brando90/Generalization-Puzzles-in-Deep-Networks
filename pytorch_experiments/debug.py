def debug():
    print(f'path_to_folder_expts = {path_to_folder_expts}')
    print(f'net_filename = {net_filename}')
    print(f'seed = {seed}')
    print(f'matlab_filename = {matlab_filename}')
    ''' '''
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    ''' '''
    data_set = 'mnist'
    data_eval_type = 'evalaute_mdl_on_full_data_set'
    evalaute_mdl_data_set = get_function_evaluation_from_name(data_eval_type)
    ''' data '''
    data_path = './data'
    print(f'data_set = {data_set}')
    trainset, testset, classes = data_class.get_data_processors(data_path, 0.0, dataset_type=data_set, standardize=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=10)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=10)
    trainloader = self.trainloader
    testloader = self.testloader
    ''' Criterion '''
    error_criterion = metrics.error_criterion
    criterion = torch.nn.CrossEntropyLoss()
    iterations = math.inf
    ''' Nets'''
    net_path = os.path.join(path_to_folder_expts, net_filename)
    net = utils.restore_entire_mdl(net_path).cuda()
    # net2 = utils.restore_entire_mdl(path).cuda()
    # net3 = utils.restore_entire_mdl(path).cuda()
    ''' stats about the nets '''
    train_loss_epoch, train_error_epoch = evalaute_mdl_data_set(criterion, error_criterion, net, trainloader, device)
    test_loss_epoch, test_error_epoch = evalaute_mdl_data_set(criterion, error_criterion, net, testloader, device)
    nb_params = nn_mdls.count_nb_params(net)
    ''' print net stats '''
    print(
        f'train_loss_norm, train_error_norm, test_loss_norm, test_error_norm = {train_loss_norm, train_error_norm, test_loss_norm, test_error_norm}')
    print(
        f'train_loss_un, train_error_un, test_loss_un, test_error_un = {train_loss_un, train_error_un, test_loss_un, test_error_un}')
    print(f'train_loss_epoch, train_error_epoch  = {train_loss_epoch}, {train_error_epoch}')
    print(f'test_loss_epoch, test_error_epoch  = {test_loss_epoch}, {test_error_epoch}')
    print(f'nb_params {nb_params}')