import numpy as np

import data_utils

import os

import torch
from torchvision import datasets, transforms
#from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

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

class IndxCifar10(torch.utils.data.Dataset):
    def __init__(self,transform):
        self.cifar10 = datasets.CIFAR10(root='./data',download=False,train=True,transform=transform)

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)

def get_standardized_transform():
    '''
    transform is to tensor follwed by x-mu/sigma
    '''
    ''' converts (HxWxC) in range [0,255] to [0.0,1.0] '''
    to_tensor = transforms.ToTensor()
    ''' Given meeans (M1,...,Mn) and std: (S1,..,Sn) for n channels, input[channel] = (input[channel] - mean[channel]) / std[channel] '''
    gaussian_normalize = transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )
    ''' transform them to Tensors of normalized range [-1, 1]. '''
    #transform = transforms.Compose([to_tensor,gaussian_normalize])
    transform = transforms.Compose([to_tensor])
    return transform

class CIFAR10RandomLabels(torchvision.datasets.CIFAR10):
  """CIFAR10 dataset, with support for randomly corrupt labels.
  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
    super(CIFAR10RandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)

  def corrupt_labels(self, corrupt_prob):
    labels = np.array(self.train_labels if self.train else self.test_labels)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]

    if self.train:
      self.train_labels = labels
    else:
      self.test_labels = labels

class MNISTRandomLabels(torchvision.datasets.MNIST):
  """MNIST dataset, with support for randomly corrupt labels.
  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
    super(MNISTRandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    self.corrupt_prob = corrupt_prob
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)

  def corrupt_labels(self, corrupt_prob):
    labels = np.array(self.train_labels if self.train else self.test_labels)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]

    if self.train:
      self.train_labels = labels
    else:
      self.test_labels = labels

####

def other_preprocessing():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train, transform_test

def get_data_processors(data_path,label_corrupt_prob,dataset_type,standardize=False,type_standardize='default'):
    '''
        The output of torchvision datasets are PILImage images of range [0, 1].
        We transform them to Tensors of (gau)normalized range [-1, 1].

        Params:
            num_workers = how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    '''
    transform = []
    ''' converts (HxWxC) in range [0,255] to [0.0,1.0] '''
    to_tensor = transforms.ToTensor()
    transform.append(to_tensor)
    ''' Given meeans (M1,...,Mn) and std: (S1,..,Sn) for n channels, input[channel] = (input[channel] - mean[channel]) / std[channel] '''
    if standardize:
        if dataset_type == 'cifar10' or dataset_type == 'cifar100':
            if type_standardize == 'default':
                gaussian_normalize = transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )
                transform.append(gaussian_normalize)
            elif type_standardize == 'data_stats':
                print('>>>>>> using DATA STATS')
                gaussian_normalize = transforms.Normalize( (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) )
                transform.append(gaussian_normalize)
            else:
                raise ValueError(f'Error what type of standardization, type_standardize = {type_standardize} not available')
        elif dataset_type == 'mnist':
            gaussian_normalize = transforms.Normalize( (0.1307,), (0.3081,) )
            transform.append(gaussian_normalize)
    ''' transform them to Tensors of normalized range [-1, 1]. '''
    transform = transforms.Compose(transform)
    ''' get cifar data '''
    if dataset_type == 'cifar10':
        ''' train sets '''
        trainset = CIFAR10RandomLabels(root=data_path, train=True, download=True,transform=transform, num_classes=10, corrupt_prob=label_corrupt_prob)
        testset = CIFAR10RandomLabels(root=data_path, train=False, download=True, transform=transform, num_classes=10, corrupt_prob=label_corrupt_prob)
        ''' classes '''
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif dataset_type == 'cifar100':
        if label_corrupt_prob != 0:
            raise ValueError('label_corrupt_prob not implemented yet, have it be zero.')
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True,download=True, transform=transform)
        #trainset = CIFAR100RandomLabels(root=data_path, train=True, download=True, transform=transform, num_classes=100,
        #                               corrupt_prob=label_corrupt_prob)
        #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=shuffle_train,
        #                                          num_workers=num_workers)
        ''' test data processor '''
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False,download=True, transform=transform)
        #testset = CIFAR100RandomLabels(root=data_path, train=False, download=True, transform=transform, num_classes=100,
        #                              corrupt_prob=label_corrupt_prob)
        ''' classes '''
        ## TODO
        classes = list(range(100))
    elif dataset_type == 'mnist':
        trainset = MNISTRandomLabels(root=data_path, train=True, download=True,transform=transform, num_classes=10, corrupt_prob=label_corrupt_prob)
        testset = MNISTRandomLabels(root=data_path, train=False, download=True, transform=transform, num_classes=10, corrupt_prob=label_corrupt_prob)
        ''' classes '''
        classes = list(range(10))
    else:
        raise ValueError(f'dataset_type = {dataset_type}')
    ''' return trainer processors'''
    print(f'------> classes = {classes}')
    return trainset, testset, classes

##

class MyData(torch.utils.data.Dataset):
    def __init__(self,path_train,eps=1,path_test=None,transform=None,dtype='float32'):
        self.transform = transform
        ## train
        np_train_data = np.load(path_train) # sorted according to smallest to largest
        self.X_train = np_train_data['X_train'].astype(dtype)
        Neps = int(self.X_train.shape[0]*eps)
        self.Y_train = np_train_data['Y_train'].astype('int')
        self.X_train = self.X_train[0:Neps,:,:,:]
        self.Y_train = self.Y_train[0:Neps]
        ## test
        # self.test = None
        # if path_test is not None:
        #     np_train_data = np.load(path_train)
        #     self.X_train = np_train_data['X_test'].astype(dtype)
        #     self.Y_train = np_train_data['Y_test']

    def __getitem__(self, index):
        data = self.X_train[index]
        target = self.Y_train[index]
        if self.transform is not None:
            data = self.transform(data)
        return data,target
        #return data, target, index

    def __len__(self):
        return len(self.X_train)

def load_only_train(path_train,eps,batch_size_train,shuffle_train,num_workers):
    '''
        Loads the train data that was made from the bottom lowest scores N*eps.
    '''
    trainset = MyData(path_train,eps=eps,transform=get_standardized_transform())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,shuffle=shuffle_train, num_workers=num_workers)
    return trainset,trainloader

#####

def imagenet():
    lql_imagenet_loc = '/cbcl/scratch01/datasets/imagenet2013'
    # Data loading code
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
