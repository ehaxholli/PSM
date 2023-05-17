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
from denoising_diffusion_pytorch import Unet, GaussianDiffusion


##############################################################################################Set arguments#################################################################################
start_time=time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--dud', type=int, default=0)#trick to quickly run in cluster
parser.add_argument('--name', type=str, default="unet_jho")#Name of the file. Becomes more detailed later
parser.add_argument('--data', type=str, default="cifar") #The dataset. Options: (cifar, celeba, imagenet64)
parser.add_argument('--nblocks', type=int, default=100)#The total number of CNF blocks. In the standard approach this is equal to 1.
parser.add_argument('--integral', type=int, default=5)# b(t)=10t thus integral of b(t) from 0 to 1 is 5*t^2, that is,  args.int*t^2. This means that args.int defines b(t)=2*args.int*t
parser.add_argument('--niters', type=int, default=50001)# number of training iterations
parser.add_argument('--lr', type=float, default=2e-4)#learning rate
parser.add_argument('--tr_batchsz', type=int, default=128)#the batch size used to train the models.
parser.add_argument('--gen_batchsz', type=int, default=100)# number of images to be generated using the reverse CNF approach
parser.add_argument('--gensde_batchsz', type=int, default=100)# number of images to be generated using the reverse SDE approach
parser.add_argument('--te_batchsz', type=int, default=96)#batch size during likelihood estimation
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--netsize', type=int, default=64)#sets the size of the Unet defined in Denoising Score Matching by Jonathan Ho et al. 
parser.add_argument('--device', type=str, default="cuda:0")#set device
parser.add_argument('--gen_sde', type=bool, default=1)#choose 1 to generate pictures using the reverse SDE approach. 0 not to.
parser.add_argument('--gen', type=bool, default=1)#choose 1 to generate pictures using the reverse CNF approach. 0 not to.
parser.add_argument('--lkh', type=bool, default=1)#choose 1 perform likelihood estimation approach. 0 not to.
parser.add_argument('--gensde_steps', type=int, default=1000)#Steps in SDE generation
parser.add_argument('--gen_steps', type=int, default=1500)# Steps in CNF generation
parser.add_argument('--lkh_steps', type=int, default=1000)# #Steps in likelihood estimation

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


############################################################################################Function Wrapper#################################################################################
#Input of the model in Jonathan Ho et al. is not compatible with the torchdiffeq package provided by Chen et al.
#here we define a class whose forward function is compatible with torchdiffeq, and which passes down the argument to the trained model 'func'.
#The class performs three tasks. 1) CNF data generation (mode=generate). 2) SDE data generation (mode=generate_sde). 3) Likelihood estimation (mode=estimate)
class Func_Wrapper(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, func):
        super().__init__()
        self.func=func #the current imported trained network
        self.mode='generate' #the default mode is to generate data using the CNF approach
    def forward(self, t, states):             
        if self.mode=='generate':
            z = states
            #save intermediate image
            save_image((z+trainmean.to(device))[:int(np.sqrt(args.gen_batchsz))**2], './'+args.name+'/gen_cnf/generated'+str(int(t.item()*100000))+'.jpg', nrow=int(np.sqrt(args.gen_batchsz)))
            
            #we deine sigma_t as torch.sqrt(1-torch.exp(-(args.integral*t**2))). the score sand the learned scaled-score f, relate each other as follows: s=-f/sigma_t
            
            col_vec=(args.integral*t**2)*torch.ones(z.shape[0]).to(device)#the integral in sigma_t to unscale the learned scaled-score
            dz_dt=self.func (z, col_vec).to(device)#the learned scaled-score
            f=-0.5*(2*args.integral*t)*(z - (1/torch.sqrt(1-torch.exp(-(args.integral*t**2))))*dz_dt)#the ode in the CNF approach
            return f
            
        elif self.mode=='generate_sde':
            z = states
            #save intermediate image
            #print(int(t.item()*10000))
            save_image((z+trainmean.to(device))[:int(np.sqrt(args.gensde_batchsz))**2], './'+args.name+'/gen_sde/generated'+str(int(t.item()*100000))+'.jpg', nrow=int(np.sqrt(args.gensde_batchsz)))
            
            #we deine sigma_t as torch.sqrt(1-torch.exp(-(args.integral*t**2))). the score sand the learned scaled-score f, relate each other as follows: s=-f/sigma_t
            
            col_vec=(args.integral*t**2)*torch.ones(z.shape[0]).to(device)#the integral in sigma_t to unscale the learned scaled-score
            dz_dt=self.func (z, col_vec).to(device)#the learned scaled-score
            f=-(2*args.integral*t)*(0.5*z - (1/torch.sqrt(1-torch.exp(-(args.integral*t**2))))*dz_dt)#the deterministic part in the SDE approach

            return f

        else:
            z , logp_z= states[0], states[1]

            with torch.set_grad_enabled(True):
                z.requires_grad_(True)
                
                #we deine sigma_t as torch.sqrt(1-torch.exp(-(args.integral*t**2))). the score sand the learned scaled-score f, relate each other as follows: s=-f/sigma_t
                
                col_vec=(args.integral*t**2)*torch.ones(z.shape[0]).to(device)#the integral in sigma_t to unscale the learned scaled-score
                dz_dt=self.func (z, col_vec).to(device)#the learned scaled-score
                f=-0.5*(2*args.integral*t)*(z - (1/torch.sqrt(1-torch.exp(-(args.integral*t**2))))*dz_dt)#the ode in the CNF approach
                
                e=sample_rademacher_like(z)
                dlogp_z_dt = divergence_approx(f, z, e=e).squeeze()#calculate divergence

            return (f, dlogp_z_dt)#the ode and the instantaneus logdet


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


        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.te_batchsz, shuffle=False) 
        
        trainmean=torch.load('imagenet64_mean')


        return 1, test_loader, trainmean, im_size    
    
train_loader, test_loader, trainmean, im_size = import_data()

###################################################################################Create model instance, and set device###########################################################################


device = torch.device(args.device)
func = Unet(dim = args.netsize, dim_mults = (1, 2, 4, 8), channels=3).to(device)  
im_dim=im_size*im_size*3 # the dimensionality of  data sample
gensde_steps=args.gensde_steps#+args.nblocks
gen_steps=args.gen_steps
lkh_steps=args.lkh_steps
###################################################################################SDE/CNF Generation and Likelihood Estimation####################################################################

# Generate SDE evolution of samples
if args.gen_sde:
    batch=torch.randn(args.gensde_batchsz,3,im_size,im_size).to(device)#sample from a standard normal
    #for each trained block do
    epsilon=1/gensde_steps
    for model_no in reversed(range(1,args.nblocks+1)):    
    
        #load block starting from the one that models the scaled-score of the gaussian distribution
        try:
            func.load_state_dict(torch.load(args.name + '/models/'+str(model_no), map_location=device))
            print('model loaded')
        except:
            print('model does not exist')
        #wrap the model to make it compatible with torchdiffeq
        func_wrap=Func_Wrapper(func)

        #set mode to generate via sde
        func_wrap.mode='generate_sde'
        with torch.no_grad():
            for t in torch.tensor(np.linspace(model_no/args.nblocks,(model_no-1)/args.nblocks+1/gensde_steps, gensde_steps//args.nblocks)).to(device):
                batch=batch-epsilon*func_wrap(t, batch)-torch.sqrt(2*args.integral*t*epsilon)*torch.randn(batch.shape).to(device)

# Generate CNF evolution of samples
if args.gen:
    batch=torch.randn(args.gen_batchsz,3,im_size,im_size).to(device)#sample from a standard normal
    
    #for each trained block do
    for model_no in reversed(range(1,args.nblocks+1)):
        #load block starting from the one that models the scaled-score of the gaussian distribution
        try:
            func.load_state_dict(torch.load(args.name + '/models/'+str(model_no), map_location=device))
            print('model loaded')
        except:
            print('model does not exist')
        #wrap the model to make it compatible with torchdiffeq
        func_wrap=Func_Wrapper(func)

        #set mode to generate via cbf
        func_wrap.mode='generate'
        with torch.no_grad():
        
            #torch diffeq. if the intial shape is [a,b,c,d] and the number of integration steps (gen_steps//args.nblocks) is n, then it return a tensor shaped [n,a,b,c,d]
            #[0,a,b,c,d], while [n-1,a,b,c,d] is the output            
            batch = odeint(
                func_wrap,
                batch,
                torch.tensor(np.linspace(model_no/args.nblocks,(model_no-1)/args.nblocks, gen_steps//args.nblocks)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='rk4',
            )
        batch = batch[-1].detach()#take output. if the block is not the last, it will be used as the input of the next block. If it is the last block, then it is discarded.
        
    

 
# Estimate likelihood
if args.lkh:  

    bpd_array=[] #for each batch, its average bits/dim will be appended here.
    
    for _, (testdatabatch, y) in enumerate(test_loader):
        val_sample=(testdatabatch - trainmean).to(device)#bring the test data to the train data format by removing the mean of the training dataset
        logp_valdet = torch.zeros(val_sample.shape[0], 1).type(torch.float32).to(device)#initialize the log dets for each sample as zero. 
        for model_no in range(1,args.nblocks+1):

            #load block starting from the one that models the scaled-score of the data distribution
            try:
                func.load_state_dict(torch.load(args.name + '/models/'+str(model_no), map_location=device))
                print('model loaded')
            except:
                print('model does not exist')
            #wrap the model to make it compatible with torchdiffeq
            func_wrap=Func_Wrapper(func)


                    
            #set mode to estimate
            func_wrap.mode='estimate'
            #torch diffeq. if the intial shape is [a,b,c,d] and the number of integration steps (gen_steps//args.nblocks) is n, then it return a tensor shaped [n,a,b,c,d]
            #[0,a,b,c,d], while [n-1,a,b,c,d] is the output          
            with torch.no_grad():
                val_sample, logp_valdet = odeint(
                    func_wrap,
                    (val_sample, logp_valdet),
                    torch.tensor(np.linspace(0.001+(model_no-1)/args.nblocks if model_no==1 else (model_no-1)/args.nblocks, model_no/args.nblocks, lkh_steps//args.nblocks)).to(device),
                    atol=1e-5,
                    rtol=1e-5,
                    method='rk4',
                )
            val_sample=val_sample[-1].detach()#take output. if the block is not the last, it will be used as the input of the next block. If it is the last block, then it is discarded.
            logp_valdet =logp_valdet[-1].detach()#take output. if the block is not the last, it will be used as the input of the next block. If it is the last block, then it is discarded.
            
            
        logp_val=(standard_normal_logprob(val_sample).view(val_sample.shape[0], -1).sum(1) + logp_valdet.view(-1)).mean()# [logp(z(0))]+[integral divergences along the integration path].
        bpd_array.append((8-logp_val/(np.log(2)*im_dim)).detach().cpu())#account for the scaling of data from [0,256] to [0,1]
        logging.info(str(np.array(bpd_array)))  #record the entire current array
        logging.info(str(np.array(bpd_array).mean()))   #record the mean of the current array


