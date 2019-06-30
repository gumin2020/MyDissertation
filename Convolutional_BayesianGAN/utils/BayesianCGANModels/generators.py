import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.BBBlayers import BBBConv2d, BBBLinearFactorial, FlattenLayer

import math

class _netG(nn.Module):#not changed, kept as a comparation.
    def __init__(self, ngpu, nz=100, ngf=64, nc=3, batch_norm_layers=[], affine=True):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        if batch_norm_layers == []:
            print("Initializing the Batch Norm layers. Affine = {}".format(affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 8, affine=affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 4, affine=affine))
            batch_norm_layers.append(nn.BatchNorm2d(ngf * 2, affine=affine))
        else:
            assert len(batch_norm_layers) == 3
            #print("Reusing the Batch Norm Layers")
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            #nn.BatchNorm2d(ngf * 8),
            batch_norm_layers[0],
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 4),
            batch_norm_layers[1],
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 2),
            batch_norm_layers[2],
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     nc, 4, 2, 1, bias=False),
            # state size. (nc) x 32 x 32
            nn.Tanh()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class _BayesianNetG(nn.Module):
    '''
    In progress, need to define a new deconvolution layer in the BBBlayers.py
    '''
    def __init__(self, noize, ):
        super(_BayesianNetG, self).__init__()

        #due to the convlayer change have to assume the logvar initial value.
        self.q_logvar_init = 0.05
        self.p_logvar_init = math.log(0.05)

        self.conv1 = BBBConv2d(self.q_logvar_init, self.p_logvar_init, noize, 128, kernel_size=3, stride=1, padding=1)
        self.soft = nn.Softplus()
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(self.q_logvar_init, self.p_logvar_init,128, 64, kernel_size=3, stride=1, padding=1)
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BBBConv2d(self.q_logvar_init, self.p_logvar_init,64, 32,  kernel_size=3, stride=1, padding=1)


        self.conv4 = BBBConv2d(self.q_logvar_init, self.p_logvar_init,32, 16,  kernel_size=3, stride=1, padding=1)

        self.conv5 = BBBConv2d(self.q_logvar_init, self.p_logvar_init,16, 3,  kernel_size=3, stride=1, padding=1)

        '''self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = BBBLinearFactorial(5 * 5 * 16, 120)
        self.soft3 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(120, 84)
        self.soft4 = nn.Softplus()

        self.fc3 = BBBLinearFactorial(84, outputs)'''

        '''layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2,
                  self.flatten, self.fc1, self.soft3, self.fc2, self.soft4, self.fc3]'''
        
        layers = [self.conv1,self.soft,self.conv2,self.soft,self.conv3,self.soft,self.conv4,self.soft,self.conv5,self.soft]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x = F.interpolate(x, scale_factor=2, mode="bilinear")
                x, _kl, = layer.convprobforward(x)
                #x = F.leaky_relu(x,negative_slope=0.2)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        return logits, kl