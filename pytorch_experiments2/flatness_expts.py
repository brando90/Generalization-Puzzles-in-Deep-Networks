"""
training an image classifier so that it overfits
----------------------------

"""
import time
from datetime import date
import calendar

import os
import sys

current_directory = os.getcwd() #The method getcwd() returns current working directory of a process.
sys.path.append(current_directory)

import torch

from torch.autograd import Variable
import torch.optim as optim

import data_classification as data_class

import nn_models as nn_mdls
import training_algorithms as tr_alg
import save_to_matlab_format as save2matlab

def main():
    nb_epochs = 2
    batch_size = 4
    batch_size_train,batch_size_test = batch_size,batch_size
    data_path = './data'
    num_workers = 2 # how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    ''' get (gau)normalized range [-1, 1]'''
    trainset,trainloader, testset,testloader, classes = data_class.get_cifer_data_processors(data_path,batch_size_train,batch_size_test,num_workers)
    ''' get NN '''
    ## conv params
    nb_filters1,nb_filters2 = 6, 16
    kernel_size1,kernel_size2 = 5,5
    ## fc params
    nb_units_fc1,nb_units_fc2,nb_units_fc3 =120,84,len(classes)
    C,H,W = 3,32,32
    net = nn_mdls.BoixNet(C,H,W,nb_filters1,nb_filters2, kernel_size1,kernel_size2, nb_units_fc1,nb_units_fc2,nb_units_fc3)
    ''' Cross Entropy + Optmizer'''
    lr = 0.001
    momentum = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    ''' Train the Network '''
    # We simply have to loop over our data iterator, and feed the inputs to the network and optimize.
    tr_alg.train_cifar(nb_epochs, trainloader,testloader, net,optimizer,criterion)
    seconds,minutes,hours = utils.report_times(start_time)
    print(f'Finished Training, hours={hours}')
    ''' Test the Network on the test data '''
    correct,total = data_class.get_error_loss_test(testloader, net)
    print(f'test_error={100*correct/total} )')

if __name__ == '__main__':
    main()
