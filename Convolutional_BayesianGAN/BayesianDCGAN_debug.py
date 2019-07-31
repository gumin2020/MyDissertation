from utils.BayesianCGANModels.generators import _netG
from utils.BayesianCGANModels.discriminators import _BayesianAlexNetD 
import sys
import pandas as pd
from ComplementCrossEntropyLoss import ComplementCrossEntropyLoss
from partial_dataset import PartialDataset
import os
import pickle
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from statsutil import AverageMeter, accuracy

# Default Parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--imageSize', type=int, default=32)
parser.add_argument('--batchSize', type=int,
                    default=64, help='input batch size')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=1000,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, default=0.0002')
parser.add_argument('--cuda', type=int, default=1, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--outf', default='modelfiles/pytorch_demo3',
                    help='folder to output images and model checkpoints')
parser.add_argument('--numz', type=int, default=1,
                    help='The number of set of z to marginalize over.')
#parser.add_argument('--num_mcmc', type=int, default=10,help='The number of MCMC chains to run in parallel')
parser.add_argument('--num_semi', type=int, default=4000,help='The number of semi-supervised samples')
parser.add_argument('--gnoise_alpha', type=float, default=0.0001, help='')
parser.add_argument('--dnoise_alpha', type=float, default=0.0001, help='')
parser.add_argument('--d_optim', type=str, default='adam',
                    choices=['adam', 'sgd'], help='')
parser.add_argument('--g_optim', type=str, default='adam',
                    choices=['adam', 'sgd'], help='')
parser.add_argument('--stats_interval', type=int, default=10,
                    help='Calculate test accuracy every interval')
parser.add_argument('--tensorboard', type=int, default=1, help='')
parser.add_argument('--bayes', type=int, default=1,
                    help='Do Bayesian GAN or normal GAN')
#newly added
parser.add_argument('--semi_supervised_boost', type=bool,
                    default=False, help='whether use semi or not')
parser.add_argument('--is_bayesian_generator', type=bool, default=False, help='use bayesian generator or not')
parser.add_argument('--is_using_classification', type=bool, default=False, help='use classifier or not')
parser.add_argument('--debug', type=bool, default=False, help='debug')
parser.add_argument('--Legacy', type=bool, default=False, help='legacy code')
parser.add_argument('--BayesianCNN_debug', type=bool, default=True, help='normal loss')
sys.argv = ['']
#reload(sys)
opt = parser.parse_args()
try:
    os.makedirs(opt.outf)
    os.makedirs(opt.outf+"/debug")
except Exception as e:
    print(str(e), opt.outf)


normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform_opt = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
dataset = dset.CIFAR10(root="./cifar10", download=True,
                       transform=transform_opt)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=0)

dataset_partial = PartialDataset(dataset, opt.num_semi)


dataset_test = dset.CIFAR10(root="./cifar10",
                            train=False,
                            transform=transform_opt)
dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=opt.batchSize, shuffle=False, pin_memory=True, num_workers=0)

dataloader_semi = torch.utils.data.DataLoader(dataset_partial, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=0)


netG = _netG(ngpu=opt.ngpu, nz=opt.nz)

num_classes = 11
netD = _BayesianAlexNetD(num_classes, 3)


if opt.d_optim == 'adam':
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
elif opt.d_optim == 'sgd':
    optimizerD = torch.optim.SGD(netD.parameters(), lr=opt.lr,
                                 momentum=0.9,
                                 nesterov=True,
                                 weight_decay=1e-4)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))

# Fixed noise for data generation
fixed_noise = torch.FloatTensor(
    opt.batchSize, opt.nz, 1, 1).normal_(0, 1).cuda()
fixed_noise = Variable(fixed_noise)

if opt.cuda:
    netD.cuda()
    netG.cuda()

# define the normal ELBO producing function
def elbo(out, y, kl, beta):
    '''
    out: output; t:target; kl: kl-divergence; beta: beta produced by get_beta
    '''
    loss = F.cross_entropy(out, y)
    return loss + beta * kl

# the get_beta function
def get_beta(epoch_idx, N):
    return 1.0 / N

def BayesianCNN_debug():
    Input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize).cuda()
    Noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1).cuda()
    Label = torch.FloatTensor(opt.batchSize).cuda()
    real_label = 1.0
    fake_label = 0.0
    iteration = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader):
            iteration += 1
            # real input
            optimizerD.zero_grad()
            _input, _label = data
            batch_size = _input.size(0)
            if opt.cuda:
                _input = _input.cuda()
            Input.resize_as_(_input).copy_(_input)
            Label.resize_(batch_size).fill_(real_label)
            Inputv = Variable(Input)
            Labelv = Variable(Label)
            output, kl = netD(Inputv)
            # --- the backprop for bayesian conv ---
            Labelv = Labelv.type(torch.cuda.LongTensor)
            errD_real = elbo(output, Labelv, kl, get_beta(epoch, len(dataloader)))
            errD_real.backward()

            # fake input
            Noise.resize_(batch_size, opt.nz, 1, 1).normal_(0, 1)
            Noisev = Variable(Noise)
            _fake = netG(Noisev)
            fake = _fake
            output, kl = netD(fake.detach()) 
            Labelv = Variable(torch.LongTensor(fake.data.shape[0]).cuda().fill_(fake_label))
            # --- the backprop for bayesian conv ---
            errD_fake = elbo(output, Labelv, kl, get_beta(epoch, len(dataloader)))
            errD_fake.backward()

            #semi_supervised
            for ii, (input_sup, target_sup) in enumerate(dataloader_semi):
                input_sup, target_sup = input_sup.cuda(), target_sup.cuda()
                break
            input_sup_v = Variable(input_sup.cuda())
            # convert target indicies from 0 to 9 to 1 to 10
            target_sup_v = Variable((target_sup + 1).cuda())
            output_sup, kl_sup = netD(input_sup_v)
            # --- the backprop for bayesian conv ---
            err_sup = elbo(output_sup, target_sup_v, kl_sup,
                        get_beta(epoch, len(dataloader_semi)))
            err_sup.backward()
            errD = errD_real + errD_fake 
            optimizerD.step()
            
            # training generator
            optimizerG.zero_grad()
            Labelv = Variable(torch.LongTensor(
                fake.data.shape[0]).cuda().fill_(real_label))
            output, kl = netD(fake)
            errG = elbo(output, Labelv, kl, get_beta(epoch, len(dataloader)))
            errG.backward()
            optimizerG.step()
            print("iteration:%d|total:%d errD:%f errG:%f"%(i,len(dataloader),errD,errG))
        #save fake img
        fake = netG(fixed_noise)
        print("saving image")
        vutils.save_image(fake.data, "%s/debug/fake_samples_epoch_%03d_G.png"%(opt.outf,epoch,),normalize=True)

if opt.BayesianCNN_debug == True:
    BayesianCNN_debug()
    sys.exit()
