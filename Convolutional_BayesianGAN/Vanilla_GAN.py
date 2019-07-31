from utils.BayesianCGANModels.generators import _BayesianNetG, _netG
from utils.BayesianCGANModels.discriminators import _BayesianLeNetD, _ClassifierD, _BayesianAlexNetD , _netD
import sys
from plotnine import *
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from utils.BayesianCGANModels.bayes import NoiseLoss, PriorLoss
from utils.BayesianCGANModels.distributions import Normal
from ComplementCrossEntropyLoss import ComplementCrossEntropyLoss
from statsutil import weights_init
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
from tensorboard_logger import configure, log_value

# Default Parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--imageSize', type=int, default=32)
parser.add_argument('--batchSize', type=int,
                    default=64, help='input batch size')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=15,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002,
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
parser.add_argument('--debug', type=bool, default=True, help='debug')
parser.add_argument('--Legacy', type=bool, default=False, help='legacy code')
parser.add_argument('--BayesianCNN_debug', type=bool, default=False, help='normal loss')
sys.argv = ['']
from importlib import reload
reload(sys)
opt = parser.parse_args()
try:
    os.makedirs(opt.outf)
except Exception as e:
    print(str(e), opt.outf)
if opt.tensorboard:
    configure(opt.outf)

# First, we construct the data loader for full training set
# as well as the data loader of a partial training set for semi-supervised learning
# transformation operator
normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform_opt = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
# get training set and test set
# train=True by default so don't panic
dataset = dset.CIFAR10(root="./cifar10", download=True,
                       transform=transform_opt)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=0)

def NormalGAN():
    img_shape  = (64,3,32,32)
    from utils.BayesianCGANModels.generators import DebugNetG
    from utils.BayesianCGANModels.discriminators import DebugNetD
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,shuffle=True, num_workers=0)
    netG = DebugNetG().cuda()
    netD = DebugNetD().cuda()
    optimizerG = torch.optim.SGD(netG.parameters(),lr=0.05,)
    optimizerD = torch.optim.SGD(netD.parameters(),lr=0.05,)
    loss = nn.BCELoss()
    for epoch in range(100):
        for i, (image, label) in enumerate(dataloader):
            real_img = Variable(torch.Tensor(image.data)).cuda()
            real_img = real_img.view(1,-1)
            real = Variable(torch.Tensor(32).fill_(1.0)).cuda()
            fake = Variable(torch.Tensor(32).fill_(0.0)).cuda()
            z = Variable(torch.Tensor(np.random.normal(0, 1, (3,100)))).cuda()
            generated_img = netG(z).cuda().detach()
            generated_img = generated_img.view(1,-1)
            #discriminator
            netD.zero_grad()
            Dloss_real = loss(netD(real_img), real)
            Dloss_fake = loss(netD(generated_img), fake)
            D_loss = Dloss_real + Dloss_fake
            D_loss.backward()
            optimizerD.step()
            #generator
            netG.zero_grad()
            G_loss = loss(netD(generated_img),real)
            G_loss.backward()
            optimizerG.step()
            print("NOW :%d/%d"%(i,len(dataloader)), "G_loss: %f"%G_loss,"D_loss: %f"%D_loss)
            if i == 780:
                break
        vutils.save_image(netG(z), "./%2d_%2d.jpg"%(epoch,i))
if opt.debug == True:
    NormalGAN()
    sys.exit()


