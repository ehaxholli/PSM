############################################################################################Import Packages################################################################################


import os
import logging
import argparse
import glob
from PIL import Image
import numpy as np
import matplotlib
import math
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tforms
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import time
import torch.nn.functional as F
from torchvision.utils import save_image
import unet

##############################################################################################Set arguments#################################################################################
start_time=time.time()


parser = argparse.ArgumentParser()
parser.add_argument('--dud', type=int, default=0)#ignore
parser.add_argument('--name', type=str, default="unet")#Name of the file. Becomes more detailed later
parser.add_argument('--data', type=str, default="cifar") #The dataset. Options: (cifar, celeba, imagenet64)
parser.add_argument('--nblocks', type=int, default=1000)#The total number of CNF blocks
parser.add_argument('--integral', type=int, default=5)# b(t)=10t thus integral of b(t) from 0 to 1 is 5*t^2, that is,  args.int*t^2. This means that args.int defines b(t)=2*args.int*t
parser.add_argument('--niters', type=int, default=50001)# number of training iterations
parser.add_argument('--lr', type=float, default=1e-3)#learning rate
parser.add_argument('--tr_batchsz', type=int, default=128)#the batch size used to train the models.
parser.add_argument('--gen_batchsz', type=int, default=100)# number of images to be generated using the reverse CNF approach
parser.add_argument('--gensde_batchsz', type=int, default=100)# number of images to be generated using the reverse SDE approach
parser.add_argument('--te_batchsz', type=int, default=200)#batch size during likelihood estimation
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--netsize', type=int, default=64)#sets the size of the Unet defined in Denoising Score Matching by Jonathan Ho et al.
parser.add_argument('--device', type=str, default="cuda:0")#set device
parser.add_argument('--gen_sde', type=bool, default=1)#choose 1 to generate pictures using the reverse SDE approach. 0 not to.
parser.add_argument('--gen', type=bool, default=1)#choose 1 to generate pictures using the reverse CNF approach. 0 not to.
parser.add_argument('--lkh', type=bool, default=1)#choose 1 perform likelihood estimation approach. 0 not to.
parser.add_argument('--inter_steps', type=int, default=5)#Interpolation steps between two scores in likelihood estimation

args = parser.parse_args()
############################################################################################Create directories###############################################################################
args.name='dataset_'+args.data+'_'+args.name+'_iterations_'+str(args.niters)+'_lr_'+str(args.lr)+'_sampnr_'+str(args.tr_batchsz)+'_nblocks_'+str(args.nblocks)

if False:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


try: #main directory where all information produced from running this file will be saved
    os.mkdir(args.name) 
except OSError as error: 
    print('Folder exists')  
try: #images generated using the reverse SDE approach are saved here
    os.mkdir(args.name+'/gen_sde') 
except OSError as error: 
    print('Folder exists')  
try:  #images generated using the reverse CNF approach are saved here
    os.mkdir(args.name+'/gen_cnf') 
except OSError as error: 
    print('Folder exists')  
try:  #likelihood estimation logs are saved here
    os.mkdir(args.name+'/est_lkh') 
except OSError as error: 
    print('Folder exists')  

############################################################################################ Define log file#################################################################################
logging.basicConfig(filename=args.name+'/est_lkh/'+"lkh.log", level=logging.INFO)
logging.info(args)

###################################################################################Tools to calculate the likelihood#################################################################################

#given point z, gives the log standard normal likelihood at z
def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

#Hutchinson's estimation of the divergence
def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx

#A Rademacher sample for the estimation of the divergence
def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1



###################################################################################Import Data#################################################################################

if args.data=='cifar':
    def import_data():
        im_size=32
        train_set = dsets.CIFAR10(
                root="~/diffusion_data", train=True, transform=tforms.Compose([
                    tforms.Resize(im_size),
                    tforms.RandomHorizontalFlip(),
                    tforms.ToTensor(),
                ]), download=True
            )
            
        test_set = dsets.CIFAR10(
                root="~/diffusion_data", train=False, transform=tforms.Compose([
                    tforms.Resize(im_size),
                    tforms.RandomHorizontalFlip(),
                    tforms.ToTensor(),
                ]), download=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.tr_batchsz, shuffle=True) 
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.te_batchsz, shuffle=False) 
        
        trainmean=torch.load('cifar_mean')


        return train_loader, test_loader, trainmean, im_size
        
        
if args.data=='celeba':
    def import_data():
        im_size=64
        train_set = dsets.CelebA(root='~/diffusion_data',
            split='train', download=True, transform=tforms.Compose([
                tforms.Resize(im_size),
                tforms.CenterCrop(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                ])
            )
        test_set = dsets.CelebA(root='~/diffusion_data',
            split='test', download=True, transform=tforms.Compose([

                tforms.Resize(im_size),
                tforms.CenterCrop(im_size),
                tforms.ToTensor(),

                ])
        )


        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.tr_batchsz, shuffle=True) 
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.te_batchsz, shuffle=False) 
        
        trainmean=torch.load('celeba_mean')


        return train_loader, test_loader, trainmean, im_size
  
if args.data=='imagenet64':
    def import_data():
        im_size=64
        test_set = dsets.ImageFolder(root='~/diffusion_data/imagenet/val', transform=tforms.Compose([
                tforms.Resize(im_size),
                tforms.CenterCrop(im_size),
                tforms.ToTensor(),
                ])
        )


        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.num_samples, shuffle=False) 
        
        trainmean=torch.load('imagenet64_mean')


        return 1, test_loader, trainmean, im_size    
    
train_loader, test_loader, trainmean, im_size = import_data()

###################################################################################Create model instance, and set device#################################################################################


device = torch.device(args.device)
func = unet.UNet(3,3, args.netsize).to(device)#func = CNF().to(device)
im_dim=im_size*im_size*3 # the dimensionality of  data sample

###################################################################################Generation and Likelihood Estimation#################################################################################
epsilon=1/args.nblocks

if args.gen_sde:  
    batch=torch.randn(args.gensde_batchsz,3,im_size,im_size).to(device)
    for model_no in reversed(range(1,args.nblocks+1)):

        try:
            func.load_state_dict(torch.load(args.name + '/models/'+str(model_no), map_location=device))
            print('model loaded')
        except:
            print('model does not exist')


        time_arr=np.linspace(0,1,args.nblocks+1)
        #print(time_arr)
        ctime=time_arr[model_no]
        b_t=torch.tensor(args.integral*ctime*2).type(torch.float32).to(device)
        integ_b_t=torch.tensor(args.integral*ctime**2).type(torch.float32).to(device)
        f=-b_t*(0.5*batch - (1/torch.sqrt(1-torch.exp(-integ_b_t)))*func(batch)).detach()+torch.sqrt(b_t/epsilon)*torch.randn(batch.shape).type(torch.float32).to(device)
        batch=batch-epsilon*f

        save_image(batch.detach()+trainmean.to(device), './'+args.name+'/gen_sde/generated'+str(model_no)+'.jpg', nrow=10)


if args.gen:  
    batch=torch.randn(args.gensde_batchsz,3,im_size,im_size).to(device)
    for model_no in reversed(range(1,args.nblocks+1)):


        try:
            func.load_state_dict(torch.load(args.name + '/models/'+str(model_no), map_location=device))

            print('model loaded')
        except:
            print('model does not exist')

        for step in range(args.inter_steps):
            alpha=step/args.inter_steps
            time_arr=np.linspace(0,1,args.nblocks+1)
            
            ctime=time_arr[model_no]
            b_t=torch.tensor(args.integral*ctime*2).type(torch.float32).to(device)
            integ_b_t=torch.tensor(args.integral*ctime**2).type(torch.float32).to(device)
            f=-0.5*b_t*(batch - (1/torch.sqrt(1-torch.exp(-integ_b_t)))*func(batch)).detach()

            
            batch=batch-(epsilon/args.inter_steps)*f

            save_image(batch.detach()+trainmean.to(device), './'+args.name+'/gen_cnf/generated'+str(model_no)+'_'+str(step)+'.jpg', nrow=10)


if args.lkh:  
    bpd_array=[] #for each batch, its average bits/dim will be appended here.
    
    for _, (testdatabatch, y) in enumerate(test_loader):
        val_sample=(testdatabatch - trainmean).to(device)#bring the test data to the train data format by removing the mean of the training dataset
        logp_valdet = torch.zeros(val_sample.shape[0]).type(torch.float32).to(device)#initialize the log dets for each sample as zero. 
        for model_no in range(1,args.nblocks+1):
            #print(model_no)
            #load block starting from the one that models the scaled-score of the data distribution
            try:
                func.load_state_dict(torch.load(args.name + '/models/'+str(model_no), map_location=device))
                print('model loaded')
            except:
                print('model does not exist')

            for step in range(args.inter_steps):
                time_arr=np.linspace(0,1,args.nblocks+1)
                ctime=time_arr[model_no]
                b_t=torch.tensor(args.integral*ctime*2).type(torch.float32).to(device)
                integ_b_t=torch.tensor(args.integral*ctime**2).type(torch.float32).to(device)
                
                with torch.set_grad_enabled(True):
                    val_sample.requires_grad_(True)
                    f=-0.5*b_t*(val_sample - (1/torch.sqrt(1-torch.exp(-integ_b_t)))*func(val_sample))
                  
                    e = sample_rademacher_like(val_sample)
                    logdet = divergence_approx(f, val_sample, e=e).squeeze()

                val_sample=val_sample.detach()
                logdet=logdet.detach()
                f=f.detach()
                
                val_sample=val_sample+epsilon*f/args.inter_steps
                logp_valdet=logp_valdet+epsilon*logdet/args.inter_steps

        logp_val=(standard_normal_logprob(val_sample).view(val_sample.shape[0], -1).sum(1) + logp_valdet.view(-1)).mean()# [logp(z(0))]+[integral divergences along the integration path].
        bpd_array.append((8-logp_val/(np.log(2)*im_dim)).detach().cpu())#account for the scaling of data from [0,256] to [0,1]
        logging.info(str(np.array(bpd_array)))  #record the entire current array
        logging.info(str(np.array(bpd_array).mean()))   #record the mean of the current array

