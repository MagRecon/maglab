"""This reimplements the phase retrieval method using automatic differentiation, as presented in
    `npj Computational Materials (2021) 7:141`.
    First, download the Fresnel images from  http://drive.google.com/uc?id=1S6RFU3eTiHVM6wN5YfS7VBk6NzmATi54, 
    rename it as "dataset", and copy it to the this directory.
"""
import torch
import numpy as np
import maglab


import matplotlib.pyplot as plt
from maglab.utils import show,show_list
import fabio


dataset = fabio.open("dataset")
A_samples = np.zeros((25,512,512))

for i in range(25):
    A_samples[i] =  dataset.getframe(i).data[185-13:697-13,:512]

defocus = np.array([-1440000,-1210000,-1000000,-810000,-640000,-490000,-360000,-250000,-160000,-90000,-40000,-10000,0,\
                      10000,40000,90000,160000,250000,360000,490000,640000,810000,1000000,1210000,1440000]) * 1e-9


ids = [2,5,19,22]
show_list([A_samples[i] for i in ids], titles=["{:.4e}nm".format(x*1e9) for x in defocus[ids]],
           same_colorbar=False)
plt.savefig("FresnelImage.png", dpi=100)
plt.close()

# phase reconstruct using normal TIE
dx = 6.9e-9
tie = maglab.TIE((512,512), dx,  qc=1.5e6)
phi = tie(np.array(A_samples[ids]), defocus[ids])
show(phi)
plt.savefig("NormalTIE.png", dpi=100)
plt.close()

# phase reconstruct using automatic differentiation TIE
E = 200.0e3
Cs=1.0e6 * 1e-9
Cc=5.0e6 * 1e-9
Ca=0.0e6 * 1e-9
phi_a=0
def_spr = 500 * 1e-9

ltem = maglab.LTEM((512,512), dx=dx, C_s=Cs, C_a=Ca,phi_a=phi_a,
              theta_c = 0.01e-3).cuda()

y0 = torch.tensor(A_samples[ids]).cuda()
amp_sqrt_guess = torch.from_numpy(np.sqrt(A_samples[12])).sqrt().cuda() #use sqrt(amp)**2 to make sure amp is positive.
phase_guess = torch.zeros(y0[0].shape).cuda()

amp_sqrt_guess.requires_grad_(True)
phase_guess.requires_grad_(True)

optimizable = [amp_sqrt_guess, phase_guess] 
optimizer = torch.optim.Adam(optimizable, lr=1e-2)
mse = torch.nn.MSELoss().cuda()

total_epoches = 1000

for i in range(total_epoches):
    optimizer.zero_grad()
    intensity_pred = torch.stack([ltem(amp_sqrt_guess**2, phase_guess, df=df, spread=def_spr) for df in defocus[ids]])
    loss = mse(y0, intensity_pred)
    loss.backward(retain_graph=True)
    optimizer.step()
    
    if i % 500 == 0:
        print("epoch:{} loss:{:.2e}".format(i, loss.item()))

data = [x.detach().cpu().numpy() for x in [amp_sqrt_guess**2, phase_guess]]
data[1] -= data[1].mean()
show_list(data, titles=["amp", "phase"], same_colorbar=False)
plt.savefig("AD_TIE.png", dpi=100)
plt.close()

y1 = torch.stack([ltem(amp_sqrt_guess**2, phase_guess, df=df, spread=def_spr) for df in defocus[ids]]).detach().cpu().numpy()
show_list([y1[i] for i in range(len(ids))], titles=["{:.4e}nm".format(x*1e9) for x in defocus[ids]],
           same_colorbar=False)
plt.savefig("predFresnelImage.png", dpi=100)
