# # Training
#
# We show that Bayesian GAN can capture the data distribution by measuring its performance in the semi-supervised setting. We will perform the posterior update as outline in Algorithm 1 in Saatchi (2017). This algorithm can be implemented quite simply by adding noise to standard optimizers such as SGD with momentum and keep track of the parameters we sample from the posterior.

# ![Posterior Sampling Algorithm](figs/bgan_alg1.png)

# ### SGHMC by Optimizing a Noisy Loss
#
# First, observe that the update rules are similar to momentum SGD except for the noise $\boldsymbol{n}$. In fact, without $\boldsymbol{n}$, this is equivalent to performing momentum SGD with the loss is $- \sum_{i=1}{J_g} \sum_{k=1}^{J_d} \log \text{posterior} $. We will describe the case where $J_g = J_d=1$ for simplicity.
#
# We use the main loss $\mathcal{L} = - \log p(\theta | ..)$ and add a noise loss $\mathcal{L}_\text{noise} = \frac{1}{\eta} \theta \cdot \boldsymbol{n}$ where $\boldsymbol{n} \sim \mathcal{N}(0, 2 \alpha \eta I)$ so that optimizing the loss function $\mathcal{L} + \mathcal{L}_\text{noise}$ with momentum SGD is equivalent to performing the SGHMC update step.
#
# Below (Equation 3 and 4) are the posterior probabilities where each error term corresponds its negative log probability.

# ![Posterior Distributions](figs/posterior_eqs2.png)

#from __future__ import print_function
from utils.BayesianCGANModels.generators import _BayesianNetG  # , _netG
from utils.BayesianCGANModels.discriminators import _BayesianLeNetD  # , _netD
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
parser.add_argument('--niter', type=int, default=2,
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
parser.add_argument('--num_mcmc', type=int, default=10,
                    help='The number of MCMC chains to run in parallel')
parser.add_argument('--num_semi', type=int, default=4000,
                    help='The number of semi-supervised samples')
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
sys.argv = ['']
del sys
opt = parser.parse_args()
try:
    os.makedirs(opt.outf)
except OSError:
    print("Error Making Directory", opt.outf)
    pass
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

# partial dataset for semi-supervised training
dataset_partial = PartialDataset(dataset, opt.num_semi)


# test set for evaluation
dataset_test = dset.CIFAR10(root="./cifar10",
                            train=False,
                            transform=transform_opt)
dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=opt.batchSize, shuffle=False, pin_memory=True, num_workers=0)

dataloader_semi = torch.utils.data.DataLoader(dataset_partial, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=0)


# Now we initialize the distributions of G and D
##### Generator ######
# opt.num_mcmc is the number of MCMC chains that we run in parallel
# opt.numz is the number of noise batches that we use. We also use different parameter samples for different batches
# we construct opt.numz * opt.num_mcmc initial generator parameters
# We will keep sampling parameters from the posterior starting from this set
# Keeping track of many MCMC chains can be done quite elegantly in Pytorch
netGs = []
for _idxz in range(opt.numz):
    for _idxm in range(opt.num_mcmc):
        netG = _BayesianNetG(noize=opt.nz)
        # netG.apply(weights_init)
        netGs.append(netG)

##### Discriminator ######
# We will use 1 chain of MCMCs for the discriminator
# The number of classes for semi-supervised case is 11; that is,
# index 0 for fake data and 1-10 for the 10 classes of CIFAR.
num_classes = 11  # not used here
#netD = _netD(opt.ngpu, num_classes=num_classes)
netD = _BayesianLeNetD(num_classes, 3)

# In order to calculate errG or errD_real, we need to sum the probabilities over all the classes (1 to K)
# ComplementCrossEntropyLoss is a loss function that performs this task
# We can specify a default except_index that corresponds to a fake label. In this case, we use index=0
criterion = nn.CrossEntropyLoss()
# use the default index = 0 - equivalent to summing all other probabilities
criterion_comp = ComplementCrossEntropyLoss(except_index=0)


# Finally, initialize the ``optimizers''
# Since we keep track of a set of parameters, we also need a set of
# ``optimizers''
if opt.d_optim == 'adam':
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
elif opt.d_optim == 'sgd':
    optimizerD = torch.optim.SGD(netD.parameters(), lr=opt.lr,
                                 momentum=0.9,
                                 nesterov=True,
                                 weight_decay=1e-4)
optimizerGs = []
for netG in netGs:
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerGs.append(optimizerG)

# since the log posterior is the average per sample, we also scale down the prior and the noise
gprior_criterion = PriorLoss(prior_std=1., observed=1000.)
gnoise_criterion = NoiseLoss(params=netGs[0].parameters(
), scale=math.sqrt(2*opt.gnoise_alpha/opt.lr), observed=1000.)
dprior_criterion = PriorLoss(prior_std=1., observed=50000.)
dnoise_criterion = NoiseLoss(params=netD.parameters(), scale=math.sqrt(
    2*opt.dnoise_alpha*opt.lr), observed=50000.)


# Fixed noise for data generation
fixed_noise = torch.FloatTensor(
    opt.batchSize, opt.nz, 1, 1).normal_(0, 1).cuda()
fixed_noise = Variable(fixed_noise)

# initialize input variables and use CUDA (optional)
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    for netG in netGs:
        netG.cuda()
    criterion.cuda()
    criterion_comp.cuda()
    input, label = input.cuda(), label.cuda()
    noise = noise.cuda()


# fully supervised
netD_fullsup = _BayesianLeNetD(num_classes, 3)
#netD_fullsup = _netD(opt.ngpu, num_classes=num_classes)
# netD_fullsup.apply(weights_init)  #was not commented out
criterion_fullsup = nn.CrossEntropyLoss()
if opt.d_optim == 'adam':
    optimizerD_fullsup = optim.Adam(
        netD_fullsup.parameters(), lr=opt.lr, betas=(0.5, 0.999))
else:
    optimizerD_fullsup = optim.SGD(netD_fullsup.parameters(), lr=opt.lr,
                                   momentum=0.9,
                                   nesterov=True,
                                   weight_decay=1e-4)
if opt.cuda:
    netD_fullsup.cuda()
    criterion_fullsup.cuda()


# We define a class to calculate the accuracy on test set
# to test the performance of semi-supervised training
def get_test_accuracy(model_d, iteration, label='semi'):
    # don't forget to do model_d.eval() before doing evaluation
    top1 = AverageMeter()
    for i, (input, target) in enumerate(dataloader_test):
        torch.no_grad()
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)
        output = model_d(input_var)[0]

        probs = output.data[:, 1:]  # discard the zeroth index
        prec1 = accuracy(probs, target, topk=(1,))[0]
        #top1.update(prec1[0], input.size(0))
        top1.update(prec1, input.size(0))
        if i % 50 == 0:
            print("{} Test: [{}/{}]\t Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                label, i, len(dataloader_test), top1=top1))
    print('{label} Test Prec@1 {top1.avg:.2f}'.format(label=label, top1=top1))
    log_value('test_acc_{}'.format(label), top1.avg, iteration)

# define the ELBO producing function


def elbo(out, y, kl, beta):
    loss = F.cross_entropy(out, y)
    return loss + beta * kl

# the get_beta function


def get_beta(epoch_idx, N):
    return 1.0 / N / 100


# --- the training part ---
iteration = 0
torch.autograd.set_detect_anomaly(True)
for epoch in range(opt.niter):
    top1 = AverageMeter()
    top1_weakD = AverageMeter()
    for i, data in enumerate(dataloader):
        iteration += 1
        #######
        # 1. real input
        netD.zero_grad()
        _input, _ = data
        batch_size = _input.size(0)
        if opt.cuda:
            _input = _input.cuda()
        input.resize_as_(_input).copy_(_input)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output, kl = netD(inputv)
        #errD_real = criterion_comp(output)
        # errD_real.backward()

        # --- the backprop for bayesian conv ---
        label = label.type(torch.LongTensor)
        label = label.cuda()
        errD_real = elbo(output, label, kl, get_beta(epoch, len(dataset)))
        #print(errD_real)

        errD_real.backward()
        # calculate D_x, the probability that real data are classified
        D_x = 1 - torch.nn.functional.softmax(output,dim=1).data[:, 0].mean()

        #######
        # 2. Generated input
        fakes = []
        for _idxz in range(opt.numz):
            noise.resize_(batch_size, opt.nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            for _idxm in range(opt.num_mcmc):
                idx = _idxz*opt.num_mcmc + _idxm
                netG = netGs[idx]
                _fake = netG(noisev)[0]
                fakes.append(_fake)
        fake = torch.cat(fakes)
        output, kl = netD(fake.detach())
        labelv = Variable(torch.LongTensor(
            fake.data.shape[0]).cuda().fill_(fake_label))
        #errD_fake = criterion(output, labelv)
        # errD_fake.backward()

        # --- the backprop for bayesian conv ---
        errD_fake = elbo(output, labelv, kl, get_beta(epoch, len(dataset)))
        errD_fake.backward()
        D_G_z1 = 1 - torch.nn.functional.softmax(output,dim=1).data[:, 0].mean()

        #######
        # 3. Labeled Data Part (for semi-supervised learning)
        for ii, (input_sup, target_sup) in enumerate(dataloader_semi):
            input_sup, target_sup = input_sup.cuda(), target_sup.cuda()
            break
        input_sup_v = Variable(input_sup.cuda())
        # convert target indicies from 0 to 9 to 1 to 10
        target_sup_v = Variable((target_sup + 1).cuda())
        output_sup, kl_sup = netD(input_sup_v)
        #err_sup = criterion(output_sup, target_sup_v)
        # --- the backprop for bayesian conv ---
        err_sup = elbo(output_sup, target_sup_v, kl_sup,
                       get_beta(epoch, len(dataset)))
        err_sup.backward()
        prec1 = accuracy(output_sup.data, target_sup + 1, topk=(1,))[0]
        #top1.update(prec1[0], input_sup.size(0))
        top1.update(prec1, input_sup.size(0))
        if opt.bayes:
            errD_prior = dprior_criterion(netD.parameters())
            errD_prior.backward()
            errD_noise = dnoise_criterion(netD.parameters())
            errD_noise.backward()
            errD = errD_real + errD_fake + err_sup + errD_prior + errD_noise
        else:
            errD = errD_real + errD_fake + err_sup
        optimizerD.step()

        # 4. Generator
        for netG in netGs:
            netG.zero_grad()
        labelv = Variable(torch.LongTensor(
            fake.data.shape[0]).cuda().fill_(real_label))
        output, kl = netD(fake)
        #errG = criterion_comp(output)
        #print(labelv) #the out put is all 1, not sure why in the original code they put it in a float tensor.
        errG = elbo(output, labelv, kl, get_beta(epoch, len(dataset)))
        if opt.bayes:
            for netG in netGs:
                errG += gprior_criterion(netG.parameters())
                errG += gnoise_criterion(netG.parameters())
        errG.backward()
        D_G_z2 = 1 - torch.nn.functional.softmax(output,1).data[:, 0].mean()
        for optimizerG in optimizerGs:
            optimizerG.step()

        # 5. Fully supervised training (running in parallel for comparison)
        netD_fullsup.zero_grad()
        input_fullsup = Variable(input_sup)
        target_fullsup = Variable((target_sup + 1))
        output_fullsup, kl_fullsup = netD_fullsup(input_fullsup)
        #err_fullsup = criterion_fullsup(output_fullsup, target_fullsup)
        # --- the backprop for bayesian conv ---
        err_fullsup = elbo(output_fullsup, target_fullsup,
                           kl_fullsup, get_beta(epoch, len(dataset)))

        optimizerD_fullsup.zero_grad()
        err_fullsup.backward()
        optimizerD_fullsup.step()

        # 6. get test accuracy after every interval
        if iteration % opt.stats_interval == 0:
            # get test accuracy on train and test
            netD.eval()
            get_test_accuracy(netD, iteration, label='semi')
            get_test_accuracy(netD_fullsup, iteration, label='sup')
            netD.train()

        # 7. Report for this iteration
        cur_val, ave_val = top1.val, top1.avg
        log_value('train_acc', top1.avg, iteration)
        #print('[%d/%d][%d/%d] Loss_D: %.2f Loss_G: %.2f D(x): %.2f D(G(z)): %.2f / %.2f | Acc %.1f / %.1f' % (epoch, opt.niter, i, len(dataloader),errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2, cur_val, ave_val))
        print('[%d/%d][%d/%d] Loss_D: %.2f Loss_G: %.2f D(x): %.2f D(G(z)): %.2f / %.2f | Acc %.1f / %.1f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data, errG.data, D_x, D_G_z1, D_G_z2, cur_val, ave_val))
    # after each epoch, save images
    vutils.save_image(_input,
                      '%s/real_samples.png' % opt.outf,
                      normalize=True)
    for _zid in range(opt.numz):
        for _mid in range(opt.num_mcmc):
            idx = _zid*opt.num_mcmc + _mid
            netG = netGs[idx]
            fake = netG(fixed_noise)[0]
            vutils.save_image(fake.data,
                              '%s/fake_samples_epoch_%03d_G_z%02d_m%02d.png' % (
                                  opt.outf, epoch, _zid, _mid),
                              normalize=True)
    for ii, netG in enumerate(netGs):
        torch.save(netG.state_dict(), '%s/netG%d_epoch_%d.pth' %
                   (opt.outf, ii, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD_fullsup.state_dict(),
               '%s/netD_fullsup_epoch_%d.pth' % (opt.outf, epoch))


#from tensorflow.python.summary import event_accumulator
ea = event_accumulator.EventAccumulator(opt.outf)
ea.Reload()

_df1 = pd.DataFrame(ea.Scalars('test_acc_semi'))
_df2 = pd.DataFrame(ea.Scalars('test_acc_sup'))
df = pd.DataFrame()
df['Iteration'] = pd.concat([_df1['step'], _df2['step']])
df['Accuracy'] = pd.concat([_df1['value'], _df2['value']])
df['Classification'] = ['BayesGAN'] * \
    len(_df1['step']) + ['Baseline']*len(_df2['step'])


# The results show that the Bayesian discriminator trained with the Bayesian generator outperforms the discriminator trained on partial data.


p = ggplot(df, aes(x='Iteration', y='Accuracy', color='Classification',
                   label='Classification')) + geom_point(size=0.5)
print(p)
