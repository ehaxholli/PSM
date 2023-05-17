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
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="unet_jho") #Name of the file. Becomes more detailed later
parser.add_argument('--data', type=str, default="cifar") #The dataset. Options: (cifar, celeba, imagenet64)
parser.add_argument('--block_no', type=int, default=1) #The index of the CNF block trained
parser.add_argument('--nblocks', type=int, default=2) #The total number of CNF blocks. In the standard approach this is equal to 1.
parser.add_argument('--integral', type=int, default=5) # b(t)=10t thus integral of b(t) from 0 to 1 is 5*t^2, that is,  args.int*t^2. This means that args.int defines b(t)=2*args.int*t
parser.add_argument('--niters', type=int, default=400001) # number of training iterations
parser.add_argument('--lr', type=float, default=2e-4) #learning rate
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_samples', type=int, default=128) #batch size
parser.add_argument('--savefreq', type=int, default=1000) #save model every this many iterations (turned off by default, that is only saves model after training)
parser.add_argument('--logfreq', type=int, default=10) #update log files every args.logfreq iterations
parser.add_argument('--netsize', type=int, default=64) #sets the size of the Unet defined in Denoising Score Matching by Jonathan Ho et al. 
args = parser.parse_args()


args.name='dataset_'+args.data+'_'+args.name+'_iterations_'+str(args.niters)+'_lr_'+str(args.lr)+'_sampnr_'+str(args.num_samples)+'_nblocks_'+str(args.nblocks)
block_no=args.block_no

############################################################################################Create directories###############################################################################

try: #main directory where all information produced from running this file will be saved
    os.mkdir(args.name) 
except OSError as error: 
    print('Folder exists')  

try: #networks of each CNF block will be saved here
    os.mkdir(args.name+'/models') 
except OSError as error: 
    print('Folder exists')  

try: #log files of each block will be saved here
    os.mkdir(args.name+'/loggs') 
except OSError as error: 
    print('Folder exists')  

############################################################################################ Define log file#################################################################################
logging.basicConfig(filename=args.name+'/loggs/'+str(args.block_no)+".log", level=logging.INFO)
logging.info(args)


#######################################################Averages loss to reduce stochasticity for easier interpretability. Does not affect training###########################################
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


########################################################################Import all data and the precalculated mean of the train data.########################################################

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

        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.num_samples, shuffle=True) 
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.num_samples, shuffle=True) 
        
        trainmean=torch.load('cifar_mean')


        return train_loader, test_loader, trainmean   


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

        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.num_samples, shuffle=True) 
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.num_samples, shuffle=True) 

        
        trainmean=torch.load('celeba_mean')


        return train_loader, test_loader, trainmean   
        
if args.data=='imagenet64':      
     def import_data():

        im_size=64
        train_set = []

        for i in range(6):
            train_set.append(dsets.ImageFolder(root='~/diffusion_data/imagenet/train'+str(i), transform=tforms.Compose([
                    tforms.Resize(im_size),
                    tforms.CenterCrop(im_size),
                    tforms.RandomHorizontalFlip(),
                    tforms.ToTensor(),
                    ])
                ))

        train_set = torch.utils.data.ConcatDataset(train_set)

        test_set = dsets.ImageFolder(root='~/diffusion_data/imagenet/val', transform=tforms.Compose([

                tforms.Resize(im_size),
                tforms.CenterCrop(im_size),
                tforms.ToTensor(),

                ])
        )
        
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.num_samples, shuffle=True) 
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.num_samples, shuffle=True) 

        
        trainmean=torch.load('imagenet64_mean')


        return train_loader, test_loader, trainmean        
        

train_loader, test_loader, trainmean = import_data()

######################################################################## Sets the device, the network, optimizer and loss averager###########################################################
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

func = Unet(dim = args.netsize, dim_mults = (1, 2, 4, 8), channels=3).to(device)  
optimizer = optim.Adam(func.parameters(), lr=args.lr)

loss_meter = RunningAverageMeter()
loss_meter_test = RunningAverageMeter()


######################################################################## Training and Validation###########################################################
itr=0
while itr < ( args.niters + 1):



    for _, (batch, y) in enumerate(train_loader): 
        optimizer.zero_grad()   # set grad to zero
        st_itr_time=time.time() #start measuring time
        itr+=1                  #increase the number of iterations performed
        if itr== (args.niters + 1):
            break               #if the number of iterations performed is higher then bound break this loop. The main loop also breaks and the model is saved. 

        batch=(batch-trainmean).type(torch.float32).to(device)# center the batch of training data and send it to device
        if batch.shape[0]!= args.num_samples:#if the batch is the laston in epoch in epoch and does not save the usual size, then skip iteration
            break

        #samples randomly the score time indices. 
        #after sampling times t1, ... tn, transform them into args.int*( t1)^2, ..., args.int*(tn)^2. Name this new array col_vec
        if block_no==1:
            col_vec=args.integral*np.random.uniform(low=0, high=0.1, size=args.num_samples)**2# (1)
        if block_no==2:
            col_vec=args.integral*np.random.uniform(low=0.1, high=1, size=args.num_samples)**2# (1)
        col_vec=torch.tensor(col_vec).reshape(-1,1,1,1).type(torch.float32).to(device)#these indices will be used to sample from the distributions corresponding to the said scores.
        
        normal_batch=torch.randn(batch.shape).type(torch.float32).to(device) # generate the epsilon as directed by denoising score matching. 
        batch=torch.exp(-0.5*col_vec)*batch + torch.sqrt(1-torch.exp(-col_vec))*normal_batch # sample points from the distribution at the times sampled in (1).
        

        f=func(batch, torch.tensor(col_vec.squeeze()).to(device)).type(torch.float32).to(device)# estimate scaled-score at sampled points corresponding to given times  t1, ... tn. 


        loss=((f-normal_batch)**2).mean() #the scaled-score at sampled points should match the noise used to create those sampled points

        loss.backward() #calculate gradient
        optimizer.step() #update parameters
    
        loss_meter.update(loss.item())


        if itr%args.logfreq==0:
            logging.info('Iter: {}, running avg loss: {:.4f}'.format(itr, loss_meter.avg)+' | ' + 'Iter Time: ' + str(time.time() - st_itr_time))#update log file
            
#        Remove the comments of the two lines below to save the model time every args.savefreq iterations
#        if itr%args.savefreq==0:
#           torch.save(func.state_dict(), args.name+'/models/'+str(block_no))
            

        if itr%1000==0:
            #send the training loss of last batch to log file
            logging.info('Iter: {}, running avg loss TRAIN: {:.4f}'.format(itr, loss.item())+' | ' + 'Iter Time: ' + str(time.time() - st_itr_time))
            
            ###############################################################Validation###############################################################
            for _, (batch, y) in enumerate(test_loader):    
                    
                with torch.no_grad():

                    batch=batch.type(torch.float32).to(device)-trainmean.type(torch.float32).to(device)# center the batch of testing data and send it to device
                    
                    #samples randomly the score time indices. 
                    #after sampling times t1, ... tn, transform them into args.int*( t1)^2, ..., args.int*(tn)^2. Name this new array col_vec
                    col_vec=args.integral*np.random.uniform(low=(block_no-1)/args.nblocks, high=block_no/args.nblocks, size=args.num_samples)**2 #(1)
                    
                    
                    col_vec=torch.tensor(col_vec).reshape(-1,1,1,1).type(torch.float32).to(device)#these indices will be used to sample from the distributions corresponding to the said scores.
                    
                    normal_batch=torch.randn(batch.shape).type(torch.float32).to(device)# generate the epsilon as directed by denoising score matching. 
                    batch=torch.exp(-0.5*col_vec)*batch + torch.sqrt(1-torch.exp(-col_vec))*normal_batch# sample points from the distribution at the times sampled in (1)

                    f=func(batch, torch.tensor(col_vec.squeeze()).to(device)).type(torch.float32).to(device)# estimate score at sampled points corresponding to given times  t1, ... tn. 


                    loss=((f-normal_batch)**2).mean()#the score at sampled points should match the noise used to create those sampled points
                #Update log file
                logging.info('Iter: {}, running avg loss VALIDATION: {:.4f}'.format(itr, loss.item())+' | ' + 'Iter Time: ' + str(time.time() - st_itr_time))
                break

############################################################################SAVE MODEL######################################################################################################
if not torch.isnan(loss):
    torch.save(func.state_dict(), args.name+'/models/'+str(block_no))

