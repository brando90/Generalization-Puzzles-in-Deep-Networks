"""
training an image classifier so that it overfits
----------------------------

"""
import torch
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

def main():
    nb_epochs = 2
    batch_size = 4
    data_path = './data'
    num_workers = 2 # how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    ''' get (gau)normalized range [-1, 1]'''
    trainset,trainloader, testset,testloader, classes = get_cifer_data_processors(data_path,batch_size_train,batch_size_test,num_workers)
    ''' get NN '''
    ## conv params
    nb_filters1,nb_filters2 = 6, 16
    kernel_size1,kernel_size2 = 5,5
    ## fc params
    nb_units_fc1,nb_units_fc2,nb_units_fc3 =120,84,len(classes)
    net = BoixNet(nb_filters1,nb_filters2, kernel_size1,kernel_size2, nb_units_fc1,nb_units_fc2,nb_units_fc3)
    ''' Cross Entropy + Optmizer'''
    lr = 0.001
    momentum = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    ''' Train the Network '''
    # We simply have to loop over our data iterator, and feed the inputs to the network and optimize.
    train_cifar(nb_epochs, trainloader,testloader, net,optimizer,criterion)
    print('Finished Training')
    ''' Test the Network on the test data '''
    correct,total = get_error_loss_test(testloader, net)
    print(f'test_error={100*correct/total} ))
