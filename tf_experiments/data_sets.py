# Loading the CIFAR-10 datasets
from keras.datasets import cifar10

from keras.utils import np_utils

def load_cifar10(num_classes,standardize=False):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train - training data(images), y_train - labels(digits)
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train  /= 255
    x_test /= 255
    ''' '''
    if standardize:
        x_train = (x_train - 0.5)/0.5
        x_test = (x_test - 0.5)/0.5
    return (x_train, y_train), (x_test, y_test)
