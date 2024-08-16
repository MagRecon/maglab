# this is a demo of employing micromagnetic simulation using steepest descent driver(Labonte's Method)
import torch
import torch.nn as nn
import maglab
import os
import numpy as np
import matplotlib.pyplot as plt


dx = 3e-9
A = 4.75e-12
D = 0.853e-3
H = 150 * maglab.const.mT
Ms = 3.84e5
Ld = 4*np.pi*A/D

def m0_ring(micro):
    nx,ny,nz=micro.shape
    coords = maglab.helper.coords
    dx = micro.dx
    x,y,z = coords(nx)*dx, coords(ny)*dx, coords(nz)*dx
    X,Y,Z = np.meshgrid(x,y,z, indexing='ij')
    R2 = X**2 + Y**2
    m = np.zeros((3,nx,ny,nz))
    m[2,] = 1.
    condition = R2<(Ld/2)**2
    condition = condition | ( (R2>Ld**2) & (R2 < (1.5*Ld)**2) & (Z**2 < (Ld/2)**2) )
    m[2, condition] = -1.
    return m



def init_micro():
    nx,ny,nz = 120,120,60
    geo = maglab.geo.cylider(120, 60)
    micro = maglab.Micro(nx,ny,nz, dx)
    micro.set_Ms(Ms*geo)
    micro.add_exch(A)
    micro.add_dmi(D)
    micro.add_zeeman((0,0,H))
    micro.add_demag()
    micro.cuda()
    return micro

micro = init_micro()
m0 = m0_ring(micro)
micro.init_m0(m0)

sd = maglab.sd.SteepestDescent(micro.shape).cuda()

os.makedirs("results", exist_ok=True)
for i in range(3000):
    spin = micro.get_spin()
    H = micro.get_total_field(spin)
    new_spin = sd(spin, H)
    micro.init_m0(new_spin)
    if i % 100 == 0:
        print(i, sd.tau)
        spin = spin.detach().cpu().numpy()
        maglab.show_list([spin[2,:,:, 30], spin[2,:,60, :]])
        plt.savefig(f"results/sd_{i}.png",dpi=100)
        plt.close()
        maglab.vtk.write_vtk(f"results/sd_{i}", spin, dx=micro.dx)
        
# Note: we now have "get_field_from_loss" function, which will extract the effective field corresponding to the phase loss.
# So you can add these two fields together and use sd to update the micro.
