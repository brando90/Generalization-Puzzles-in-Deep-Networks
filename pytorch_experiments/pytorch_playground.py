import torch
from torch import nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import copy

def get_cifar10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader,classes

def combine_nets(net_train,net_no_train,net_place_holder):
    '''
        Combine nets in a way train net is trainable
    '''
    params_train = net_no_train.named_parameters()
    dict_params_place_holder = dict( net_place_holder.named_parameters() )
    dict_params_no_train = dict(net_train.named_parameters())
    for name,param_train in params_train:
        if name in dict_params_place_holder:
            param_no_train = dict_params_no_train[name]
            delattr(net_place_holder, name)
            W_new = param_train + param_no_train # notice addition is just chosen for the sake of an example
            setattr(net_place_holder, name, W_new)
    return net_place_holder

def combining_nets_lead_to_error():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ''' create three musketeers '''
    net_train = nn.Sequential(
          nn.Conv2d(3,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        ).to(device)
    net_no_train = copy.deepcopy(net_train).to(device)
    net_place_holder = copy.deepcopy(net_train).to(device)
    ''' prepare train, hyperparams '''
    trainloader,classes = get_cifar10()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    ''' train '''
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            optimizer.zero_grad() # zero the parameter gradients
            inputs, labels = inputs.device(), labels.to(device)
            # combine nets
            net_place_holder = combine_nets(net_train,net_no_train,net_place_holder)
            #
            outputs = net_place_holder(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    ''' DONE '''
    print('Done \a')

