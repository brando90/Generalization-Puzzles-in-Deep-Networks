import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace as st

class Net(nn.Module):
    # Cifar/MNIST example
    def __init__(self):
        super(Net, self).__init__()
        # in_channels = # channels from previous layer
        # out_channels = # of filters (since thats the # filters for next layer)
        # kernel_size = tuple (H,W) in pytorch
        self.conv1 = nn.Conv2d(3, 6, 5) #(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

##

class BoixNet(nn.Module):
    ## The network has 2 convolutional layers followed by 3 fully connected.
    ## Use ReLUs, and no batch normalization or regularizers.
    ## Trained with cross-entropy
    ## https://discuss.pytorch.org/t/when-creating-new-neural-net-from-scratch-how-does-one-statically-define-what-the-size-of-the-a-flatten-layer-is-not-at-runtime/14235
    def __init__(self,C,H,W, nb_filters1,nb_filters2, kernel_size1,kernel_size2, nb_units_fc1,nb_units_fc2,nb_units_fc3,do_bn=False):
        super(BoixNet, self).__init__()
        self.do_bn = do_bn
        ''' Initialize conv layers'''
        self.conv1 = nn.Conv2d(3,nb_filters1, kernel_size1) #(in_channels, out_channels, kernel_size)
        if self.do_bn: self.bn_conv1 = nn.BatchNorm2d(nb_filters1)
        self.conv2 = nn.Conv2d(nb_filters1,nb_filters2, kernel_size2)
        if self.do_bn: self.bn_conv2 = nn.BatchNorm2d(nb_filters2)
        CHW = ((H-kernel_size1+1)-kernel_size2+1) * ((W-kernel_size1+1)-kernel_size2+1) * nb_filters2
        ''' '''
        self.fc1 = nn.Linear(CHW, nb_units_fc1)
        if self.do_bn: self.fc1_bn = nn.BatchNorm1d(nb_units_fc1)
        self.fc2 = nn.Linear(nb_units_fc1,nb_units_fc2)
        if self.do_bn: self.fc2_bn = nn.BatchNorm1d(nb_units_fc2)
        self.fc3 = nn.Linear(nb_units_fc2,nb_units_fc3)
        if self.do_bn: self.fc3_bn = nn.BatchNorm1d(nb_units_fc3)

    def forward(self, x):
        ''' conv layers'''
        pre_act1 = self.bn_conv1(self.conv1(x)) if self.do_bn else self.conv1(x)
        a_conv1 = F.relu(pre_act1)
        ##
        pre_act2 = self.bn_conv2(self.conv2(a_conv1)) if self.do_bn else self.conv2(a_conv1)
        a_conv2 = F.relu(pre_act2)
        ''' FC layers '''
        _,C,H,W = a_conv2.size()
        a_flat_conv2 = a_conv2.view(-1,C*H*W)
        ##
        pre_act_fc1 = self.fc1_bn(self.fc1(a_flat_conv2)) if self.do_bn else self.fc1(a_flat_conv2)
        a_fc1 = F.relu(pre_act_fc1)
        pre_act_fc2 = self.fc2_bn(self.fc2(a_fc1)) if self.do_bn else self.fc2(a_fc1)
        a_fc2 = F.relu(pre_act_fc2)
        pre_act_fc3 = self.fc3_bn(self.fc3(a_fc2)) if self.do_bn else self.fc3(a_fc2)
        a_fc3 = pre_act_fc3
        return a_fc3
##

class LiaoNet(nn.Module):
    ## 5 conv net FC
    def __init__(self,C,H,W, Fs, Ks, FC,do_bn=False):
        super(LiaoNet, self).__init__()
        self.do_bn = do_bn
        self.nb_conv_layers = len(Fs)
        ''' Initialize Conv layers '''
        self.convs = []
        self.bns = []
        out = Variable(torch.FloatTensor(1, C,H,W))
        in_channels = C
        for i in range(self.nb_conv_layers):
            F,K = Fs[i], Ks[i]
            conv = nn.Conv2d(in_channels,F,K) #(in_channels, out_channels, kernel_size)
            if self.do_bn:
                bn = nn.BatchNorm2d(F)
                setattr(self,f'bn{i}',bn)
                self.bns.append(bn)
            ##
            setattr(self,f'conv{i}',conv)
            self.convs.append(conv)
            ##
            in_channels = F
            out = conv(out)
        ''' Initialize FC layers'''
        CHW = out.numel()
        self.fc = nn.Linear(CHW,FC)
        if self.do_bn:
            self.bn_fc = nn.BatchNorm1d(FC)

    def forward(self, x):
        ''' conv layers '''
        for i in range(self.nb_conv_layers):
            conv = self.convs[i]
            ##
            z = conv(x)
            if self.do_bn:
                bn = self.bns[i]
                z = bn(z)
            x = F.relu(z)
        _,C,H,W = x.size()
        x = x.view(-1,C*H*W)
        ''' FC layers '''
        x = self.fc(x)
        if self.do_bn:
            x = self.bn_fc(x)
        return x
