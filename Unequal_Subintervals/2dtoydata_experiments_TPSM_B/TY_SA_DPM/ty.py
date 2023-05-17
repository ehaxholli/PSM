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
parser.add_argument('--niters', type=int, default=2000000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--integral', type=float, default=6)
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--results_dir', type=str, default="./results")
args = parser.parse_args()

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
                f=-0.5*(2*args.integral*t)*(z - (1/torch.sqrt(1-torch.exp(-(args.integral*t**2))))*dz_dt)
                dlogp_z_dt = -trace_df_dz(f, z).view(batchsize, 1)
            
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
    a=[]
    translation_arr=np.array([0,-0.6])
    for i in range(1):
        a.append(np.random.multivariate_normal(np.array([0,0])-translation_arr, 0.015*np.identity(2),4500))
    
    points=a[0]
    
    b=[]
    for i in range(8):
        b.append(np.random.multivariate_normal(0.45*np.array([np.cos(np.pi*i/4),np.sin(np.pi*i/4)])-translation_arr, 0.003*np.identity(2),1000))
    
    points2=np.vstack((b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]))
    points=np.vstack((points,points2))
    
    points= points[np.random.permutation(points.shape[0])]
    c=[]
    for i in range(16):
        c.append(np.random.multivariate_normal(0.7*np.array([np.cos(np.pi*(i)/8),np.sin(np.pi*(i)/8)])-translation_arr, 0.001*np.identity(2),300))
    
    points2=np.vstack((c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11],c[12],c[13],c[14],c[15]))
    points=np.vstack((points,points2))
    
    ########T
    points=np.vstack((points,    np.random.uniform(low=[-1,-0.8], high=[-0.9,-0.5], size=(500,2))))
    points=np.vstack((points,    np.random.uniform(low=[-1.1,-0.5], high=[-0.8,-0.4], size=(500,2))))
    ########H              
    points=np.vstack((points,    np.random.uniform(low=[-0.6,-0.8], high=[-0.5,-0.4], size=(500,2))))
    points=np.vstack((points,    np.random.uniform(low=[-0.5,-0.65], high=[-0.4,-0.55], size=(125,2))))    
    points=np.vstack((points,    np.random.uniform(low=[-0.4,-0.8], high=[-0.3,-0.4], size=(500,2))))
    ########A
    rot=-70*(np.pi/180)
    temp_points=np.matmul(np.random.uniform(low=[-0.2,-0.05], high=[0.2,0.05], size=(500,2)),np.array([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]]))
    temp_points=temp_points-np.array([0,0.6])
    points=np.vstack((points, temp_points))
    
    rot=-110*(np.pi/180)
    temp_points=np.matmul(np.random.uniform(low=[-0.2,-0.05], high=[0.2,0.05], size=(500,2)),np.array([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]]))
    temp_points=temp_points-np.array([-0.2,0.6])
    points=np.vstack((points, temp_points))  
      
    points=np.vstack((points,    np.random.uniform(low=[-0,-0.7], high=[0.2,-0.6], size=(125,2))))    
    #points=np.vstack((points,    np.random.uniform(low=[-1.25,0.4], high=[1.25,0.5], size=(num_samples,2))))
    #points=np.vstack((points,    np.random.uniform(low=[-1.25,-0.4], high=[1.25,-0.5], size=(num_samples,2))))  
    ########N             
    points=np.vstack((points,    np.random.uniform(low=[0.4,-0.8], high=[0.5,-0.4], size=(500,2))))
    
    rot=-130*(np.pi/180)
    temp_points=np.matmul(np.random.uniform(low=[-0.2,-0.05], high=[0.2,0.05], size=(500,2)),np.array([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]]))
    temp_points=temp_points-np.array([-0.6,0.6])  
    points=np.vstack((points, temp_points))    
    
    points=np.vstack((points,    np.random.uniform(low=[0.7,-0.8], high=[0.8,-0.4], size=(500,2))))
    ########K
    points=np.vstack((points,    np.random.uniform(low=[1,-0.8], high=[1.1,-0.4], size=(500,2))))
    
    rot=-130*(np.pi/180)
    temp_points=np.matmul(np.random.uniform(low=[-0.1,-0.05], high=[0.1,0.05], size=(200,2)),np.array([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]]))
    temp_points=temp_points-np.array([-1.2,0.7])  
    points=np.vstack((points, temp_points))    

    rot=-45*(np.pi/180)
    temp_points=np.matmul(np.random.uniform(low=[-0.1,-0.05], high=[0.1,0.05], size=(200,2)),np.array([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]]))
    temp_points=temp_points-np.array([-1.2,0.5])  
    points=np.vstack((points, temp_points))         
    

    ########Y
    points=np.vstack((points,    np.random.uniform(low=[-0.6,-1.3], high=[-0.5,-1], size=(200,2))))

    rot=-45*(np.pi/180)
    temp_points=np.matmul(np.random.uniform(low=[-0.1,-0.05], high=[0.1,0.05], size=(200,2)),np.array([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]]))
    temp_points=temp_points-np.array([0.45,1])  
    points=np.vstack((points, temp_points))    
    
    rot=-130*(np.pi/180)
    temp_points=np.matmul(np.random.uniform(low=[-0.1,-0.05], high=[0.1,0.05], size=(200,2)),np.array([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]]))
    temp_points=temp_points-np.array([0.65,1])  
    points=np.vstack((points, temp_points))         
    
    ########O
    points=np.vstack((points,    np.random.uniform(low=[-0.25,-1.3], high=[-0.15,-0.9], size=(500,2))))#h
    points=np.vstack((points,    np.random.uniform(low=[0.15,-1.3], high=[0.25,-0.9], size=(500,2))))    #h
    
    points=np.vstack((points,    np.random.uniform(low=[-0.15,-1.3], high=[0.15,-1.2], size=(500,2))))    #v
    points=np.vstack((points,    np.random.uniform(low=[-0.15,-1], high=[0.15,-0.9], size=(500,2))))   #v
    
     
    ########U
    #points=np.vstack((points,    np.random.uniform(low=[-0.8,-1.3], high=[-0.7,-1], size=(500,2))))
    points=np.vstack((points,    np.random.uniform(low=[0.45,-1.3], high=[0.55,-0.9], size=(500,2))))#h
    points=np.vstack((points,    np.random.uniform(low=[0.85,-1.3], high=[0.95,-0.9], size=(500,2))))    #h
    
    points=np.vstack((points,    np.random.uniform(low=[0.55,-1.3], high=[0.85,-1.2], size=(500,2))))    #v   


    
    
    
    
    
    
    
       
    points= points[np.random.permutation(points.shape[0])]
    points=points[:num_samples]

    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)
    if iftrain:
        return x
    else: return (x,logp_diff_t1)



if __name__ == '__main__':
    t0 = 0.01
    t1 = 1
#    device = torch.device('cuda:' + str(args.gpu)
#                          if torch.cuda.is_available() else 'cpu')
    device=torch.device('cpu')
    mean_of_dist=get_batch(100000).mean(0).reshape(1,-1)
    # model
    func = CNF(in_out_dim=4, hidden_dim=args.hidden_dim, width=args.width).to(device)

    try:
        func.load_state_dict(torch.load('model', map_location=device))
    except:
        print('model does not exist')

    optimizer = optim.Adam(func.parameters(), lr=args.lr)
    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.tensor([0.0, 0.0]).to(device),
        covariance_matrix=torch.tensor([[1, 0.0], [0.0, 1]]).to(device)
    )
    loss_meter = RunningAverageMeter()

    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        start_time=timep.time()
        for itr in range(1, args.niters + 1):
           
            optimizer.zero_grad()

            batch = get_batch(args.num_samples)-mean_of_dist
            batch=torch.tensor(batch).type(torch.float32).to(device)
            time=np.random.uniform(low=0, high=1.0, size=args.num_samples)
            time=torch.tensor(time).type(torch.float32).to(device)
            col_vec=args.integral*time**2
            col_vec=torch.tensor(col_vec).reshape(-1,1).type(torch.float32).to(device)

            normal_batch=torch.randn(batch.shape).type(torch.float32).to(device)
            
            batch=torch.exp(-0.5*col_vec)*batch + torch.sqrt(1-torch.exp(-col_vec))*normal_batch

            f=func(torch.tensor(time.reshape(-1,1)).to(device), batch.squeeze()).type(torch.float32).to(device)


            loss=((f-normal_batch)**2).mean()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            if itr%100==0:
                logging.info('Iter: {}, running avg loss: {:.4f}'.format(itr, loss_meter.avg)+' | time:'+str(timep.time()-start_time) )
                start_time=timep.time()


        
    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))
    func_wrap=Func_Wrapper(func)
    torch.save(func.state_dict(), 'model')
    if True:
        
        viz_samples = 30000
        viz_timesteps = 300
        target_sample, _ = get_batch(viz_samples, False)

        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        func_wrap.train='generate'
        with torch.no_grad():

            # Estimate Likelihood
            val_sample, _ = get_batch(viz_samples, False)
            val_sample= val_sample - mean_of_dist
            logp_val = torch.zeros(val_sample.shape[0], 1).type(torch.float32).to(device)
            func_wrap.train='estimate'
            val_samplet, logp_valt = odeint(
                func_wrap,
                (val_sample, logp_val),
                torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='rk4',
            )
            val_sample=val_samplet[-1]
            logp_val =logp_valt[-1]
            print(torch.abs(val_sample).mean())
            print(-p_z0.log_prob(val_sample).mean())
            print(logp_val.view(-1).mean())
            print( (-p_z0.log_prob(val_sample) + logp_val.view(-1)).mean())


            # Generate evolution of samples
            z_t0 = p_z0.sample([viz_samples]).to(device)
            logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)

            z_t_samples, _ = odeint(
                func_wrap,
                (z_t0, logp_diff_t0),
                torch.tensor(np.linspace(t1, t0, viz_timesteps)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='rk4',
            )
            z_t_samples=z_t_samples+mean_of_dist.reshape(1,1,-1)

            # Generate evolution of density
            x = np.linspace(-1.5, 1.5, 100)
            y = np.linspace(-1.5, 1.5, 100)
            points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

            z_t1 = torch.tensor(points).type(torch.float32).to(device)
            z_t1 = z_t1 - mean_of_dist
            logp_diff_t1 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)
            func_wrap.train='estimate'
            z_t_density, logp_diff_t = odeint(
                func_wrap,
                (z_t1, logp_diff_t1),
                torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='rk4',
            )
            
            
            # Create plots for each timestep
            for (t, z_sample, z_density, logp_diff) in zip(
                    np.linspace(t0, t1, viz_timesteps),
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
                ax3.tricontourf(*(z_t1+mean_of_dist).detach().cpu().numpy().T,
                                np.exp(logp.detach().cpu().numpy()), 200)

                plt.savefig(os.path.join(args.results_dir, f"cnf-viz-{int(t*1000):05d}.jpg"),
                           pad_inches=0.2, bbox_inches='tight')
                plt.close()

            img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(args.results_dir, f"cnf-viz-*.jpg")))]
            img.save(fp=os.path.join(args.results_dir, "cnf-viz.gif"), format='GIF', append_images=imgs,
                     save_all=True, duration=250, loop=0)

        print('Saved visualization animation at {}'.format(os.path.join(args.results_dir, "cnf-viz.gif")))
        
        
        

