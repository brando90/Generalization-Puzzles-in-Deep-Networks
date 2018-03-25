import numpy as np

import torch
from torch.autograd import Variable
#from torch import optim

import pdb

def create_classification_sin_data(N_train,N_test, lb,ub, freq_sin):
    ## define x
    x_train = np.linspace(lb,ub,N_train)
    x_test = np.linspace(lb,ub,N_test)
    ## get y's
    f_target = lambda x: np.sin(2*np.pi*freq_sin*x)
    y_train = (f_target(x_train) > 0).astype(int)
    y_test = (f_target(x_test) > 0).astype(int)
    return x_train,x_test, y_train,y_test

def build_model(input_dim, output_dim, bias):
    '''
        We don't need the softmax layer here since CrossEntropyLoss already
        uses it internally.
    '''
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim, bias=bias)
    )
    return model

def index_batch(X,batch_indices,dtype):
    '''
    returns the batch indexed/sliced batch
    '''
    if len(X.shape) == 1: # i.e. dimension (M,) just a vector
        batch_xs = torch.FloatTensor(X[batch_indices]).type(dtype)
    else:
        batch_xs = torch.FloatTensor(X[batch_indices,:]).type(dtype)
    return batch_xs

def get_batch2(X,Y,M,dtype):
    '''
    get batch for pytorch model
    '''
    # TODO fix and make it nicer, there is pytorch forum question
    # https://discuss.pytorch.org/t/how-to-get-mini-batches-in-pytorch-in-a-clean-and-efficient-way/10322
    # https://stackoverflow.com/questions/45113245/how-to-get-mini-batches-in-pytorch-in-a-clean-and-efficient-way
    X,Y = X.data.numpy(), Y.data.numpy()
    N = len(Y)
    valid_indices = np.array( range(N) )
    batch_indices = np.random.choice(valid_indices,size=M,replace=False)
    batch_xs = index_batch(X,batch_indices,dtype)
    batch_ys = index_batch(Y,batch_indices,dtype)
    return Variable(batch_xs, requires_grad=False), Variable(batch_ys, requires_grad=False)

def main():
    #torch.manual_seed(42)
    X_train, X_test, Y_train, Y_test = load_mnist(onehot=False)
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    Y_train = torch.from_numpy(Y_train).long()
    ##
    model = build_model(n_features, n_classes)
    loss = torch.nn.CrossEntropyLoss(size_average=True)
    ##
    for i in range(100):
        epoch_loss = 0
        num_batches = n_examples // batch_size
        for k in range(num_batches):

            current_loss = train(model, loss, optimizer, trX[start:end], trY[start:end])
        ##
        pred_y_train = predict(model, X_train)
        pred_y_test = predict(model, X_test)
        ##
        epoch_nb = i + 1,
        current_loss = cost/num_batches
        acc_train = 100*np.mean(predY == Y_test)
        acc_test = 100*np.mean(predY == Y_train)
        print(f"epoch_nb={epoch_nb},cost={current_loss},acc={acc}"
