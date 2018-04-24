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
    transform = []
    ''' converts (HxWxC) in range [0,255] to [0.0,1.0] '''
    to_tensor = transforms.ToTensor()
    transform.append(to_tensor)
    ''' Given meeans (M1,...,Mn) and std: (S1,..,Sn) for n channels, input[channel] = (input[channel] - mean[channel]) / std[channel] '''
    gaussian_normalize = transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )
    transform.append(gaussian_normalize)
    ''' transform them to Tensors of normalized range [-1, 1]. '''
    transform = transforms.Compose(transform)
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

def get_cifer_data_processors(data_path,batch_size_train,batch_size_test,num_workers,label_corrupt_prob,shuffle_train=True,suffle_test=False,standardize=False):
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
        gaussian_normalize = transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )
        transform.append(gaussian_normalize)
    ''' transform them to Tensors of normalized range [-1, 1]. '''
    transform = transforms.Compose(transform)
    ''' train data processor '''
    #trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,download=True, transform=transform)
    trainset = CIFAR10RandomLabels(root=data_path, train=True, download=True,transform=transform, num_classes=10,corrupt_prob=label_corrupt_prob)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,shuffle=shuffle_train, num_workers=num_workers)
    ''' test data processor '''
    #testset = torchvision.datasets.CIFAR10(root=data_path, train=False,download=True, transform=transform)
    testset = CIFAR10RandomLabels(root=data_path, train=False, download=True,transform=transform, num_classes=10,corrupt_prob=label_corrupt_prob)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,shuffle=suffle_test, num_workers=num_workers)
    ''' classes '''
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    ''' return trainer processors'''
    return trainset,trainloader, testset,testloader, classes

#####

def get_MNIST_data_processor(data_path,batch_size_train,batch_size_test,num_workers,label_corrupt_prob):
    # TODO
    kwargs = {'num_workers': 1, 'pin_memory': True}
    ''' converts (HxWxC) in range [0,255] to [0.0,1.0] '''
    to_tensor = transforms.ToTensor()
    ''' Given meeans (M1,...,Mn) and std: (S1,..,Sn) for n channels, input[channel] = (input[channel] - mean[channel]) / std[channel] '''
    gaussian_normalize = transforms.Compose( transforms.Normalize((0.1307,),(0.3081,)) )
    ''' transform them to Tensors of normalized range [-1, 1]. '''
    transform = transforms.Compose([to_tensor,gaussian_normalize])
    ''' train data processor '''
    trainset = datasets.MNIST(root=data_path, train=True,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,shuffle=True, num_workers=num_workers)
    ''' test data processor '''
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,shuffle=False, num_workers=num_workers)
