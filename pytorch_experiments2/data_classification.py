import numpy as np

import data_utils

import torch
import torchvision
import torchvision.transforms as transforms

from pdb import set_trace as st

def get_quadratic_plane_classification_data_set(N_train,N_test,lb,ub,D0):
    '''
    data set with feature space phi(x)=[x0,x1,x2]=[1,x,x^2]
    separating hyperplane = [0,1,-2]
    corresponds to line x2 = 0.5x1
    '''
    ## target function
    freq_sin = 4
    #f_target = lambda x: np.sin(2*np.pi*freq_sin*x)
    #f_target = lambda x: (x-0.25)*(x-0.75)*(x+0.25)*(x+0.75)
    def f_target(x):
        poly_feat = PolynomialFeatures(degree=2)
        x_feature = poly_feat.fit_transform(x) # N x D, [1, x, x^2]
        normal = np.zeros((1,x_feature.shape[1])) # 1 x D
        normal[:,[0,1,2]] = [0,1,-2]
        score = np.dot(normal,x_feature.T)
        label = score > 0
        return label.astype(int)
    ## define x
    X_train = np.linspace(lb,ub,N_train).reshape((N_train,D0))
    X_test = np.linspace(lb,ub,N_test).reshape((N_test,D0))
    ## get y's
    Y_train = f_target(X_train)
    Y_test = f_target(X_test)
    return X_train,X_test, Y_train,Y_test

def separte_data_by_classes(Xtr,Ytr):
    N,D = Xtr.shape
    X_pos = []
    X_neg = []
    for n in range(Xtr.shape[0]):
        x = Xtr[n,:]
        if Ytr[n]==1:
            #np.append(X_pos,x,axis=0)
            X_pos.append(x)
        else:
            #np.append(X_neg,x,axis=0)
            X_neg.append(x)
    return np.array(X_pos),np.array(X_neg)

def get_2D_classification_data(N_train,N_val,N_test,lb,ub,f_target):
    '''
    Returns x in R^2 classification data
    '''
    ''' make mesh grid'''
    Xtr_grid,Ytr_grid = data_utils.generate_meshgrid(N_train,lb,ub)
    Xv_grid,Yv_grid = data_utils.generate_meshgrid(N_val,lb,ub)
    Xt_grid,Yt_grid = data_utils.generate_meshgrid(N_test,lb,ub)
    ''' make data set with target function'''
    Xtr,Ytr = data_utils.make_mesh_grid_to_data_set_with_f(f_target,Xtr_grid,Ytr_grid)
    Xv,Yv = data_utils.make_mesh_grid_to_data_set_with_f(f_target,Xv_grid,Yv_grid)
    Xt,Yt = data_utils.make_mesh_grid_to_data_set_with_f(f_target,Xt_grid,Yt_grid)
    ''' Convert to ints '''
    Ytr,Yv,Yt = np.int64(Ytr).reshape(Ytr.shape[0]) ,np.int64(Yv).reshape(Yv.shape[0]), np.int64(Yt).reshape(Yt.shape[0])
    return Xtr,Ytr, Xv,Yv, Xt,Yt

#####

def get_cifer_data_processors(data_path,batch_size_train,batch_size_test,num_workers):
    '''
        The output of torchvision datasets are PILImage images of range [0, 1].
        We transform them to Tensors of (gau)normalized range [-1, 1].

        Params:
            num_workers = how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    '''
    ''' converts (HxWxC) in range [0,255] to [0.0,1.0] '''
    to_tensor = transforms.ToTensor()
    ''' Given meeans (M1,...,Mn) and std: (S1,..,Sn) for n channels, input[channel] = (input[channel] - mean[channel]) / std[channel] '''
    gaussian_normalize = transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )
    ''' transform them to Tensors of normalized range [-1, 1]. '''
    transform = transforms.Compose([to_tensor,gaussian_normalize])
    ''' train data processor '''
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,shuffle=True, num_workers=num_workers)
    ''' test data processor '''
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,shuffle=False, num_workers=num_workers)
    ''' classes '''
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    ''' return trainer processors'''
    return trainset,trainloader, testset,testloader, classes

def get_error_loss_test(testloader, net):
    correct,total = 0,0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return correct,total
