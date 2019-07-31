import torch
import torch.nn as nn
from utils.BBBlayers import BBBConv2d, BBBLinearFactorial, FlattenLayer
import numpy as np
import math

class DebugNetD(nn.Module):
    def __init__(self):
        super(DebugNetD, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3072*64, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 32),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        validity = validity.view(validity.size(0),-1)

        return validity

class _netD(nn.Module):
    def __init__(self, ngpu, num_classes=1, nc=3, ndf=64):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.num_classes = num_classes
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            # conv2D(in_channels, out_channels, kernelsize, stride, padding)
            nn.Conv2d(nc, ndf , 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2 x 2
            nn.Conv2d(ndf * 8, num_classes, 2, 1, 0, bias=False),
            # out size = batch x num_classes x 1 x 1
        )

        if self.num_classes == 1:
          self.main.add_module('prob', nn.Sigmoid())
          # output = probability
        else:
          pass
          # output = scores

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        #print(input.shape)
        return output.view(input.size(0), self.num_classes).squeeze(1)

class _BayesianAlexNetD(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(_BayesianAlexNetD, self).__init__()

        self.q_logvar_init = 0.05
        self.p_logvar_init = math.log(0.05)
 
        self.classifier = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init, 1* 1 * 128, outputs)

        self.conv1 = BBBConv2d(self.q_logvar_init, self.p_logvar_init, inputs, 64, kernel_size=11, stride=4, padding=5)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(self.q_logvar_init,  self.p_logvar_init, 64, 192, kernel_size=5, padding=2)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BBBConv2d(self.q_logvar_init, self.p_logvar_init, 192, 384, kernel_size=3, padding=1)
        self.soft3 = nn.Softplus()

        self.conv4 = BBBConv2d(self.q_logvar_init, self.p_logvar_init, 384, 256, kernel_size=3, padding=1)
        self.soft4 = nn.Softplus()

        self.conv5 = BBBConv2d(self.q_logvar_init, self.p_logvar_init, 256, 128, kernel_size=3, padding=1)
        self.soft5 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.flatten = FlattenLayer(1 * 1 * 128)
        # self.fc1 = BBBLinearFactorial(q_logvar_init, N, p_logvar_init, 1* 1 * 128, outputs)


        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2, self.conv3, self.soft3,
                  self.conv4, self.soft4, self.conv5, self.soft5, self.pool3]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
            else:
                x = layer.forward(x)
        x = x.view(x.size(0), -1)
        x, _kl = self.classifier.fcprobforward(x)
        kl += _kl
        logits = x
        return logits, kl

class _BayesianLeNetD(nn.Module):
    def __init__(self, outputs, inputs,):
        super(_BayesianLeNetD, self).__init__()

        self.outputs = outputs
        #due to the convlayer change have to assume the logvar initial value.
        self.q_logvar_init = 0.05
        self.p_logvar_init = math.log(0.05)

        self.conv1 = BBBConv2d(self.q_logvar_init, self.p_logvar_init,inputs,6, 5, stride=1)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(self.q_logvar_init, self.p_logvar_init,6, 16, 5, stride=1)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init,5 * 5 * 16, 120)
        self.soft3 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init,120, 84)
        self.soft4 = nn.Softplus()

        self.fc3 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init,84, outputs)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2,
                  self.flatten, self.fc1, self.soft3, self.fc2, self.soft4, self.fc3]

        #not sure if this is right, test drive.
        self.prob = nn.Sigmoid()
        if outputs == 1:
            layers.append(self.prob)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):#used name: probforward
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        #return logits, kl
        return logits.view(x.size(0), self.outputs).squeeze(1), kl

class _ClassifierD(nn.Module):
    '''
    the ACGAN part, the code is doing the classification here
    '''
    def __init__(self, outputs, inputs,):
        super(_ClassifierD, self).__init__()
        self.outputs = outputs
        #due to the convlayer change have to assume the logvar initial value.
        self.q_logvar_init = 0.05
        self.p_logvar_init = math.log(0.05)

        self.conv1 = BBBConv2d(self.q_logvar_init, self.p_logvar_init,inputs,6, 5, stride=1)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(self.q_logvar_init, self.p_logvar_init,6, 16, 5, stride=1)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init,5 * 5 * 16, 120)
        self.soft3 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init,120, 84)
        self.soft4 = nn.Softplus()

        #self.fc3 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init,84, outputs)

        self.fcA = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init,84, 1)
        self.fcB = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init,84, 1)
        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2,
                  self.flatten, self.fc1, self.soft3, self.fc2, self.soft4]

        #not sure if this is right, test drive.
        self.prob = nn.Sigmoid()
        if outputs == 1:
            layers.append(self.prob)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):#used name: probforward
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        #logits = x
        #return logits, kl
        logitsA, klA = self.fcA.fcprobforward(x)
        logitsB, klB = self.fcB.fcprobforward(x)
        return logitsA, logitsB, klA+kl, klB+kl