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
parser.add_argument('--nblocks', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--integral', type=float, default=5)
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--results_dir', type=str, default="./results_paral_sde")
args = parser.parse_args()
device=torch.device('cpu')
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
logging.basicConfig(filename="log.log", level=logging.INFO)
  
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)
      
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
            f=-(2*args.integral*t)*(0.5*z - (1/torch.sqrt(1-torch.exp(-(args.integral*t**2))))*dz_dt)
            epsilon=1/800
            return (f+torch.randn(f.shape).type(torch.float32).to(device)*torch.sqrt(2*args.integral*t/epsilon),dlogp_z_dt)
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

viz_samples = 30000
z_t0 = p_z0.sample([viz_samples]).to(device)

viz_timesteps = 1000//args.nblocks
target_sample, _ = get_batch(viz_samples, False)

x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-1.5, 1.5, 100)
points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T
z_t1 = torch.tensor(points).type(torch.float32).to(device)

z_t1c=z_t1.clone() - mean_of_dist
logp_diff_t1 = torch.zeros(z_t1c.shape[0], 1).type(torch.float32).to(device)

for step_no in range(1,args.nblocks+1):

    print(step_no)
    if __name__ == '__main__':
        # model
        func = CNF(in_out_dim=4, hidden_dim=args.hidden_dim, width=args.width).to(device)



        
        with torch.no_grad():
            # Generate evolution of samples

            try:
                func.load_state_dict(torch.load('models/model'+str(args.nblocks+1-step_no), map_location=device))
            except:
                print('model does not exist')
            func_wrap=Func_Wrapper(func)
            func_wrap.train='generate'
            logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)
            if step_no==1:
                times=np.linspace(1+0.0001, 0.3+0.0001, viz_timesteps)
            elif step_no==2:     
                times=np.linspace(0.3+0.0001, 0.1+0.0001, viz_timesteps)
            elif step_no==3:     
                times=np.linspace(0.1+0.0001, 0.02+0.0001, viz_timesteps)
            elif step_no==4:     
                times=np.linspace(0.021+0.0001, 0.0001, viz_timesteps)

                
            z_t_samples, _ = odeint(
                func_wrap,
                (z_t0, logp_diff_t0),
                torch.tensor(times).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='euler',
            )
            z_t0 = z_t_samples.clone()
            z_t0 = z_t0[-1].detach()
            z_t_samples=z_t_samples+mean_of_dist.reshape(1,1,-1)
            print(z_t_samples.shape)

            
            # Generate evolution of density
            try:
                func.load_state_dict(torch.load('models/model'+str(step_no), map_location=device))
            except:
                print('model does not exist')
            func_wrap=Func_Wrapper(func)
            
            if step_no==1:
                times=np.linspace(0.0001, 0.02+0.0001, viz_timesteps)
            elif step_no==2:
                times=np.linspace(0.02+0.0001,0.1+0.0001, viz_timesteps)   
            elif step_no==3:
                times=np.linspace(0.1+0.0001,0.3+0.0001, viz_timesteps) 
            elif step_no==4:
                times=np.linspace(0.3+0.0001,1+0.0001, viz_timesteps)     
                            
            func_wrap.train='estimate'
            z_t_density, logp_diff_t = odeint(
                func_wrap,
                (z_t1c, logp_diff_t1),
                torch.tensor(times).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='euler',
            )
            z_t1c = z_t_density[-1].detach()
            logp_diff_t1 = logp_diff_t[-1].detach()
            print(z_t_density.shape)
            print(logp_diff_t.shape)

            # Create plots for each timestep
            for (t, z_sample, z_density, logp_diff) in zip(
                    np.linspace((step_no-1)/args.nblocks+0.0001, step_no/args.nblocks+0.0001, viz_timesteps),
                    z_t_samples, z_t_density, logp_diff_t
            ):
                fig = plt.figure(figsize=(12, 4), dpi=200)
                plt.tight_layout()
                plt.axis('off')
                plt.margins(0, 0)
                fig.suptitle(f'{t:.2f}s')

                ax1 = fig.add_subplot(1, 3, 1)
                ax1.set_title('Target')
                ax1.get_xaxis().set_ticks([])
                ax1.get_yaxis().set_ticks([])
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.set_title('Samples')
                ax2.get_xaxis().set_ticks([])
                ax2.get_yaxis().set_ticks([])
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.set_title('Log Probability')
                ax3.get_xaxis().set_ticks([])
                ax3.get_yaxis().set_ticks([])

                ax1.hist2d(*target_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                ax2.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
                ax3.tricontourf(*z_t1.detach().cpu().numpy().T,
                                np.exp(logp.detach().cpu().numpy()), 200)

                plt.savefig(os.path.join(args.results_dir, f"cnf-viz-{int(t*1000):05d}.jpg"),
                           pad_inches=0.2, bbox_inches='tight')
                plt.close()
            plt.clf()
            plt.close()
            #img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(args.results_dir, f"cnf-viz-*.jpg")))]
#            img.save(fp=os.path.join(args.results_dir, "cnf-viz.gif"), format='GIF', append_images=imgs,
#                     save_all=True, duration=250, loop=0)
            
                     

        print('Saved visualization animation at {}'.format(os.path.join(args.results_dir, "cnf-viz.gif")))
    
    
    

