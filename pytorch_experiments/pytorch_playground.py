import torch
from torch import nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from collections import OrderedDict

import copy

from pdb import set_trace as st

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def dont_train(net):
    '''
    set training parameters to false.
    '''
    for param in net.parameters():
        param.requires_grad = False
    return net

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
            layer_name, param_name = name.split('.')
            ## get place holder layer
            layer_place_holder = getattr(net_place_holder,layer_name)
            delattr(layer_place_holder, param_name)
            ## get new param
            param_no_train = dict_params_no_train[name]
            W_new = param_train + param_no_train # notice addition is just chosen for the sake of an example
            ## store param in placehoder net
            setattr(layer_place_holder, param_name, W_new)
    return net_place_holder

def main():
    '''
    Intention is to only train the net with trainable params.
    Placeholde rnet is a dummy net, it doesn't actually do anything except hold the combination of params and its the
    net that does the forward pass on the data.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ''' create three musketeers '''
    net_train = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(3,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('Flatten', Flatten()),
          ('fc0',nn.Linear(36864,10))
        ])).to(device)
    net_no_train = copy.deepcopy(net_train).to(device)
    net_place_holder = copy.deepcopy(net_train).to(device)
    ''' prepare train, hyperparams '''
    trainloader,classes = get_cifar10()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net_train.parameters(), lr=0.001, momentum=0.9)
    ''' train '''
    net_train.train()
    net_no_train.eval()
    net_place_holder.eval()
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            optimizer.zero_grad() # zero the parameter gradients
            inputs, labels = inputs.to(device), labels.to(device)
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

if __name__ == '__main__':
    main()