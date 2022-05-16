## Lexi Jones
## Last edited: 05/16/22

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model_functions_project_v2 import krylov_solve_2D_elliptic_streamfunction,velocity_from_streamfunction,upwind_fluxes_2D,diffusive_fluxes_2D,reformat_1D_to_2D

Lx,Ly = 50,50 # Lx & Ly are the dimensions of psi, so the tracer has dimension (Lx-1,Ly-1)
del_x,del_y,del_t = 2000,2000,3000
num_steps = 2419 ######## EDDY PEAKS @ 6 WEEKS = 1209 timesteps, EDDY DIES @ 12 WEEKS = 2419 timesteps

P,N,psi,N_star,P_star = FE_upwind_2D_spinup_adv_diff_2tracers(Lx,Ly,del_x,del_y,del_t,num_steps,1)
P_2D = reformat_1D_to_2D(P,Lx-1,Ly-1)
N_2D = reformat_1D_to_2D(N,Lx-1,Ly-1)
psi_2D = reformat_1D_to_2D(psi,Lx,Ly)

#################### ANIMATION ####################
fig, ax = plt.subplots(1,3,figsize=(21,5))

stream = ax[0].pcolormesh(psi_2D[0],vmin=-20000,vmax=0)
ax[0].set_title('Streamfunction ($\psi$)',fontsize=18)
ax[0].set_xlabel('x index (%sm)'%(del_x),fontsize=18)
ax[0].set_ylabel('y index (%sm)'%(del_y),fontsize=18)
cbar0 = fig.colorbar(stream, ax=ax[0])
cbar0.set_label('$\psi$', rotation=270)
cbar0.ax.get_yaxis().labelpad = 15

phyto = ax[1].pcolormesh(P_2D[0,1:-1,1:-1]/1000,vmin=0,vmax=(P_star/1000)*3)
ax[1].set_title('Phytoplankton Concentration',fontsize=18)
ax[1].set_xlabel('x index (%sm)'%(del_x),fontsize=18)
ax[1].set_ylabel('y index (%sm)'%(del_y),fontsize=18)
cbar1 = fig.colorbar(phyto, ax=ax[1])
cbar1.set_label('$P (\mu M N)$', rotation=270)
cbar1.ax.get_yaxis().labelpad = 15

nut = ax[2].pcolormesh(N_2D[0,1:-1,1:-1]/1000,vmin=0.0,vmax=(N_star/1000)*3)
ax[2].set_title('Nutrient Concentration',fontsize=18)
ax[2].set_xlabel('x index (%sm)'%(del_x),fontsize=18)
ax[2].set_ylabel('y index (%sm)'%(del_y),fontsize=18)
cbar2 = fig.colorbar(phyto, ax=ax[2])
cbar2.set_label('$N (\mu M N)$', rotation=270)
cbar2.ax.get_yaxis().labelpad = 15

skip = 20 ########## SKIP EVERY X TIME STEPS

def animate(i):
    stream.set_array(psi_2D[i*skip].ravel())
    phyto.set_array((P_2D[i*skip,1:-1,1:-1]/1000).ravel())
    nut.set_array((N_2D[i*skip,1:-1,1:-1]/1000).ravel())

    ax[1].text(0.5, 1.100, "Timestep: %s (%s s)"%(i*skip,del_t),fontsize=18,
            bbox={'facecolor': 'white'}, #,'alpha': 0.5, 'pad': 5
            transform=ax[1].transAxes, ha="center")

    return stream,phyto,nut

anim = FuncAnimation(fig, animate, interval=10, frames=int(len(P_2D)/skip))

anim.save('phytoplankton_nutrients_spinup_vort_dev.gif')

#################### FIGURES ####################

phyto_sum,nut_sum = [],[]
for i in np.arange(0,len(P_2D)):
    phyto_sum.append(np.sum(P_2D[i])/1000)
    nut_sum.append(np.sum(N_2D[i])/1000)
    #phyto_sum.append(np.sum(P_2D[i])/(1000))
    #nut_sum.append(np.sum(N_2D[i])/(1000))

fig, ax = plt.subplots(1,2,figsize=(12,5))

ax[0].plot(phyto_sum)
ax[0].set_title('Total Phytoplankton Concentration')
ax[0].set_xlabel('Time Step (%s s)'%(del_t))
ax[0].set_ylabel('$\sum_{i,j} P^n (\mu M N)$')

ax[1].plot(nut_sum)
ax[1].set_title('Total Nitrate Concentration')
ax[1].set_xlabel('Time Step (%s s)'%(del_t))
ax[1].set_ylabel('$\sum_{i,j} N^n (\mu M N)$')

plt.savefig('NP_concentrations_vort_dev.png')
