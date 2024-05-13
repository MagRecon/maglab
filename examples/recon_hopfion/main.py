"""
    This is a demo to reconstruct a hopfion ring using a series magnetic phases.
    Firstly, you need to download the magnetization file from https://github.com/MagRecon/HopfionRing, and copy it to this path.
    
    The tilt angles are from -30 degree to +30 degree, which is a critical situation.
    (This demo requires 12000 epoches)
    
    We are utilizing a relatively small batch size (4) for this reconstruction, which will require more total batches.
    
    For a simpler reproduction, please set the max_tilt_angle to 60 and use a larger batch size (such as 12 or 20).
"""
import torch
import torch.nn as nn

import os
import numpy as np
import matplotlib.pyplot as plt

import maglab
from maglab.dataset import PhaseMap, PhaseSet, PhaseLoader

from maglab.utils import show_list
from maglab.saver import Saver

"""
0.Parameters of the micromagnetic system
"""
Ms0 = 3.84e5
A0 = 4.75e-12
D0 = 0.853e-3
H0 = 150 * maglab.const.mT
cellsize = 3e-9

"""
1.Load magnetization and initialize Micro.
"""

m0 = np.load("HopfionRing.npy")
geometry = maglab.Micro.m2geo(m0)
(nx,ny,nz) = geometry.shape
print("geometry:", geometry.shape)
micro = maglab.Micro(geometry, cellsize).cuda()
micro.init_m0(m0)

"""
2.Prepare phase data.
"""

max_tilt_angle = 30
tilt_angles = np.arange(-1*max_tilt_angle, max_tilt_angle+1, 5)
N = 200 # size of phase
phasemapper = maglab.PhaseMapper(N, cellsize, rotation_padding=N).cuda()
phaseset = PhaseSet()
seed = 0 # use random seed when adding Gaussian noise
for axis in [0, 1]:
    for t in tilt_angles:
        #avoid second load of 0 degree data
        if np.abs(t) < 1e-3 and axis == 1:
            continue
        phi = phasemapper(m0, t, axis, Ms=Ms0)  
        phasemap = PhaseMap(phi, t, axis)
        phasemap = phasemap.add_Gaussian(sigma=0.1, seed=seed)
        phaseset.load(phasemap)
        seed += 1   
loader = PhaseLoader(phaseset, batch_size=4)

# If you want to save the phase data, use 
# ```
# data = phasemap.data.cpu().numpy() 
# ```
# and save it as a numpy array.

"""
3.Start reconstruction.
"""

# 3.1 Initialize the magnetization to uniform state.
micro.init_m0((0,0,1.)) 

# 3.2 Add interactions
micro.add_exch(A0)
micro.add_dmi(D0)
micro.add_zeeman((0,0,H0))
micro.add_demag()
micro.cuda()

# 3.3 Build optimizer and loss function.
optimizable = [x for x in micro.parameters() if x.requires_grad] 
optimizer = torch.optim.Adam(optimizable, lr=1e-2)
mse = nn.MSELoss().cuda()


# 3.4 Start reconstruction.

path_checkpoints = 'checkpoints/'
path_info = 'info/'
for p in [path_checkpoints, path_info]:
    if not os.path.isdir(p):
        os.makedirs(p)
        
weight_phi = 0.2
total_batches = 120000
save_info_epoch = 200
save_m_epoch = 10000

epoch = 0
batch = 0
torch.manual_seed(0) # use random seed for dataloader shuffle

saver = Saver(f"{path_info}/log.txt")
while batch < total_batches:
    for phase, mask, angle, axis in loader:
        optimizer.zero_grad()
        
        # Micromagnetic loss
        loss_m = micro.loss(Ms=Ms0)
        
        # Phase loss
        m = micro.get_m()
        phi_predict = torch.stack([phasemapper(m, angle[i], axis[i], Ms=Ms0) for i in range(len(axis))]).unsqueeze(1)
        mask = mask.cuda()
        # Use mask if the phase data is incomplete.
        loss_phi = mse(phase.cuda() * mask, phi_predict * mask) 

        # Total loss
        loss = loss_m + weight_phi * loss_phi
        loss.backward(retain_graph=True)
        optimizer.step()
        batch += 1
        
    epoch += 1
    if epoch % save_info_epoch == 0:
        info = 'Epoch:{}  Loss M:{:.3e} Loss Phase:{:.3e}'.format(epoch, loss_m, loss_phi, )
        saver.write(info)
        ms = micro.get_m().detach().cpu().numpy()
        show_list([ms[i,:,:,nz//2] for i in range(3)],)
        plt.savefig(f"{path_info}/epoch{epoch}.png")
        plt.close()
        
    if epoch % save_m_epoch == 0:
        micro.save_state(f"{path_checkpoints}/epoch{epoch}.pth", Ms=Ms0)
        
        
micro.save_state(f"{path_checkpoints}/final_epoch{epoch}.pth", Ms = Ms0)
# save as vtk
#maglab.vtk.write_vtk(f"{path_checkpoints}/final", micro.get_m().detach().cpu().numpy(), cellsize=cellsize)