import os
import argparse
import logging
import glob
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import torch
import torch.nn as nn
import torch.optim as optim
import time as timep

parser = argparse.ArgumentParser()
parser.add_argument('--dud', type=float, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--niters', type=int, default=500000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--nblocks', type=int, default=200)
parser.add_argument('--integral', type=float, default=5)
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--results_dir', type=str, default="./results_paral")
args = parser.parse_args()
device=torch.device('cpu')
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
logging.basicConfig(filename="log.log", level=logging.INFO)
        
class CNF(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.l1 = nn.Linear(in_out_dim,100)
        self.l2 = nn.Linear(100,150)
        self.l3 = nn.Linear(150,100)
        self.l4 = nn.Linear(100,in_out_dim//2)
        self.elu=nn.ELU()

    def forward(self, time, states):

        z=states
        batchsize = z.shape[0]
        time=time.to(z).repeat(1,2)

        dz_dt=torch.cat((z,time),1)
        
        dz_dt=self.l1(dz_dt)
        dz_dt=self.elu(dz_dt)
        dz_dt=self.l2(dz_dt)
        dz_dt=self.elu(dz_dt)
        dz_dt=self.l3(dz_dt)
        dz_dt=self.elu(dz_dt)
        dz_dt=self.l4(dz_dt)   
        return dz_dt


class Func_Wrapper(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, func):
        super().__init__()
        self.func=func
        self.train='generate'

    def forward(self, t, states):             
        if self.train=='generate':
            z , logp_z= states[0], states[1]
            batchsize = z.shape[0]
            t=t*torch.ones(batchsize).to(z).reshape(-1,1)
            with torch.set_grad_enabled(True):
                z.requires_grad_(True)
                dz_dt=self.func (t, z)
                dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)
            f=-0.5*(2*args.integral*t)*(z - (1/torch.sqrt(1-torch.exp(-(args.integral*t**2))))*dz_dt)
            return (f,dlogp_z_dt)
        else:
            z , logp_z= states[0], states[1]
            batchsize = z.shape[0]
            t=t*torch.ones(batchsize).to(z).reshape(-1,1)
            with torch.set_grad_enabled(True):
                z.requires_grad_(True)
                dz_dt=self.func (t, z)
                f=-0.5*(2*args.integral*t)*(z - (1/torch.sqrt(1-torch.exp(-(args.integral*t**2))))*dz_dt)
                dlogp_z_dt = -trace_df_dz(f, z).view(batchsize, 1)

            return (f, dlogp_z_dt)


def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()



def get_batch(num_samples, iftrain=True):

    points= np.random.uniform(low=[0.4,-1.25], high=[0.5,1.25], size=(num_samples,2))
    points=np.vstack((points,    np.random.uniform(low=[-0.4,-1.25], high=[-0.5,1.25], size=(num_samples,2))))
    points=np.vstack((points,    np.random.uniform(low=[-1.25,0.4], high=[1.25,0.5], size=(num_samples,2))))
    points=np.vstack((points,    np.random.uniform(low=[-1.25,-0.4], high=[1.25,-0.5], size=(num_samples,2))))

    points=np.vstack((points, np.random.multivariate_normal(np.array([-1.25,-1.25]),0.02*np.array([[1,0],[0,1]]), num_samples)))
    points=np.vstack((points, np.random.multivariate_normal(np.array([-1.25,1.25]),0.02*np.array([[1,0],[0,1]]), num_samples)))
    points=np.vstack((points, np.random.multivariate_normal(np.array([1.25,-1.25]),0.02*np.array([[1,0],[0,1]]), num_samples)))
    points=np.vstack((points, np.random.multivariate_normal(np.array([1.25,1.25]),0.02*np.array([[1,0],[0,1]]), num_samples)))
    points=np.vstack((points, np.random.multivariate_normal(np.array([0,0]),0.02*np.array([[1,0],[0,1]]), num_samples)))

    points= points[np.random.permutation(points.shape[0])]
    points=points[:num_samples]

    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)

    if iftrain:
        return x
    else: return (x,logp_diff_t1)


p_z0 = torch.distributions.MultivariateNormal(
    loc=torch.tensor([0.0, 0.0]).to(device),
    covariance_matrix=torch.tensor([[1, 0.0], [0.0, 1]]).to(device)
)
mean_of_dist=get_batch(100000).mean(0).reshape(1,-1)
viz_samples=30000
viz_timesteps = 1000//args.nblocks
val_samplet = get_batch(viz_samples)-mean_of_dist
val_samplet = torch.tensor(val_samplet).type(torch.float32).to(device)#.reshape([2, -1]).T
l_d_t1 = torch.zeros(val_samplet.shape[0], 1).type(torch.float32).to(device)

for step_no in range(1,args.nblocks+1):
    print(step_no)
    if __name__ == '__main__':

        # model
        func = CNF(in_out_dim=4, hidden_dim=args.hidden_dim, width=args.width).to(device)
        try:
            func.load_state_dict(torch.load('models/model'+str(step_no), map_location=device))
        except:
            print('model does not exist')
        func_wrap=Func_Wrapper(func)


        func_wrap.train='generate'
        with torch.no_grad():
        
        
            #VALIDATION

            
            func_wrap.train='estimate'
            z_t_dens, l_d_t0 = odeint(
                func_wrap,
                (val_samplet, l_d_t1),
                torch.tensor(np.linspace((step_no-1)/args.nblocks+0.0001, step_no/args.nblocks+0.0001, viz_timesteps)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='rk4',
            )
            val_samplet=z_t_dens[-1]
            l_d_t1=l_d_t0[-1]
            print(-(p_z0.log_prob(val_samplet) - l_d_t1.view(-1)).mean())
    
    
    

