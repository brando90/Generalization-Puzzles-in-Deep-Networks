import torch

import pickle

import utils
import data_classification as data_class

def save_index_according_to_criterion(path,dataloader,criterion):
    '''

    :param dataloader:
    :param criteron:
    :return:
    '''
    ''' produce scores list [(i,score)] '''
    enable_cuda = False
    index_scores = []
    for i,data_train in enumerate(dataloader):
        inputs, labels = utils.extract_data(enable_cuda,data_train,wrap_in_variable=False)
        score = criterion(inputs)
        index_scores.append( (i,score) )
    ''' sort scores list based on scores '''
    sorting_criterion = lambda tup: tup[1]
    sorted_by_second = sorted(index_scores, key=sorting_criterion)
    ''' pickle the index,score sorted array to '''
    with open(path,'wb+') as array_file: #wb+ Opens a file for both writing and reading in binary format. Overwrites the existing file if the file exists. If the file does not exist, creates a new file for reading and writing.
        pickle.dump(sorted_by_second, array_file)

def other():
    cifar_dataset = torchvision.datasets.CIFAR10(root='./data', transform=transform)
    train_indices =  # select train indices according to your rule
    test_indices =  # select test indices according to your rule
    train_loader = torch.utils.data.DataLoader(cifar_dataset, batch_size=32, shuffle=True,
                                               sampler=SubsetRandomSampler(train_indices))
    test_loader = torch.utils.data.DataLoader(cifar_dataset, batch_size=32, shuffle=True,
                                              sampler=SubsetRandomSampler(test_indices))

def main():
    ''' get data loaders '''
    #TODO
    standardize = True # x - mu / std , [-1,+1]
    trainset, trainloader, testset, testloader, classes_data = data_class.get_cifer_data_processors(data_path,batch_size_train,batch_size_test,num_workers,args.label_corrupt_prob,suffle_test=suffle_test,standardize=standardize)
    ''' load net for the cirterion '''
    path_train = '.data/'
    ''' '''
    save_data_according_to_criterion(dataloader,citerion)