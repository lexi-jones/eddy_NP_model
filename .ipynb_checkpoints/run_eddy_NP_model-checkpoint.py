## Lexi Jones
## Last edited: 05/17/22

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from eddy_NP_model_functions import reformat_1D_to_2D,FE_upwind_2D_adv_diff_eddy_NP_model,FE_upwind_2D_adv_diff_eddy_NPP_model

# Experimental variables
P_combo = 'S' # 'S', 'L', 'SLu', or 'SLe'
alter_vort = 0 # 0 (coherent) or 1 (leaky)
death_rate = 0.5 #d^-1

# Set up grid and time stepping
Lx,Ly = 50,50 # Lx & Ly are the dimensions of psi, so the tracer has dimension (Lx-1,Ly-1)
del_x,del_y,del_t = 2000,2000,3000
num_steps = 100#2419 ######## EDDY PEAKS @ 6 WEEKS = 1209 timesteps, EDDY DIES @ 12 WEEKS = 2419 timesteps

all_P = [] # will hold the data for each of the phytoplankton
if (P_combo == 'S') or (P_combo == 'L'): # one phytoplankton
    numP = 1
    P_mat,N_mat,psi_mat,N_star,P_star = FE_upwind_2D_adv_diff_eddy_NP_model(Lx,Ly,del_x,del_y,del_t,num_steps,alter_vort,P_combo,death_rate)
    P_2D = reformat_1D_to_2D(P_mat,Lx-1,Ly-1)
    all_P.append(P_2D)
    P_stars = [P_star] #array used for setting max value in figures
    if (P_combo == 'S'):
        size = ['Small'] #array used for figure-making
    elif (P_combo == 'L'):
        size = ['Large']

elif (P_combo == 'SLu') or (P_combo == 'SLe'): # two phytoplankton
    numP = 2
    PL_mat,PS_mat,N_mat,psi_mat,N_star,P_star_S,P_star_L = FE_upwind_2D_adv_diff_eddy_NPP_model(Lx,Ly,del_x,del_y,del_t,num_steps,alter_vort,P_combo,death_rate)
    PS_2D = reformat_1D_to_2D(PS_mat,Lx-1,Ly-1)
    PL_2D = reformat_1D_to_2D(PL_mat,Lx-1,Ly-1)
    all_P.append(PS_2D)
    all_P.append(PL_2D)
    P_stars = [P_star_S,P_star_L] #array used for setting max value in figures
    size = ['Small','Large'] #array used for figure-making

N_2D = reformat_1D_to_2D(N_mat,Lx-1,Ly-1)
psi_2D = reformat_1D_to_2D(psi_mat,Lx,Ly)

#################### ANIMATION ####################
fig, ax = plt.subplots(1,numP+2,figsize=(7*(numP+2),5))

stream = ax[0].pcolormesh(psi_2D[0],vmin=-20000,vmax=0)
ax[0].set_title('Streamfunction ($\psi$)',fontsize=18)
ax[0].set_xlabel('x index (%sm)'%(del_x),fontsize=18)
ax[0].set_ylabel('y index (%sm)'%(del_y),fontsize=18)
cbar0 = fig.colorbar(stream, ax=ax[0])
cbar0.set_label('$\psi$', rotation=270)
cbar0.ax.get_yaxis().labelpad = 15

for n in np.arange(0,numP):
    if n == 0:
        phyto1 = ax[1+n].pcolormesh(all_P[0][0,1:-1,1:-1]/1000,vmin=0,vmax=(P_stars[n]/1000)*3)
        cbar = fig.colorbar(phyto1, ax=ax[1+n])
    else:
        phyto2 = ax[1+n].pcolormesh(all_P[0][0,1:-1,1:-1]/1000,vmin=0,vmax=(P_stars[n]/1000)*3)
        cbar = fig.colorbar(phyto2, ax=ax[1+n])

    ax[1+n].set_title('%s Phytoplankton\nConcentration'%(size[n]),fontsize=18)
    ax[1+n].set_xlabel('x index (%sm)'%(del_x),fontsize=18)
    ax[1+n].set_ylabel('y index (%sm)'%(del_y),fontsize=18)
    cbar.set_label('$P (\mu M N)$', rotation=270)
    cbar.ax.get_yaxis().labelpad = 15

nut = ax[-1].pcolormesh(N_2D[0,1:-1,1:-1]/1000,vmin=0.0,vmax=(N_star/1000)*3)
ax[-1].set_title('Nutrient Concentration',fontsize=18)
ax[-1].set_xlabel('x index (%sm)'%(del_x),fontsize=18)
ax[-1].set_ylabel('y index (%sm)'%(del_y),fontsize=18)
cbar2 = fig.colorbar(nut, ax=ax[-1])
cbar2.set_label('$N (\mu M N)$', rotation=270)
cbar2.ax.get_yaxis().labelpad = 15

skip = 20 ########## SKIP EVERY X TIME STEPS FOR ANIMATION

def animate_NP(i):
    stream.set_array(psi_2D[i*skip].ravel())
    phyto1.set_array((all_P[0][i*skip,1:-1,1:-1]/1000).ravel())
    nut.set_array((N_2D[i*skip,1:-1,1:-1]/1000).ravel())
    ax[0].text(0.5, 1.100, "Timestep: %s (%s s)"%(i*skip,del_t),fontsize=18,
            bbox={'facecolor': 'white'}, #,'alpha': 0.5, 'pad': 5
            transform=ax[1].transAxes, ha="center")
    return stream,phyto1,nut

def animate_NPP(i):
    stream.set_array(psi_2D[i*skip].ravel())
    phyto1.set_array((all_P[0][i*skip,1:-1,1:-1]/1000).ravel())
    phyto2.set_array((all_P[1][i*skip,1:-1,1:-1]/1000).ravel())
    nut.set_array((N_2D[i*skip,1:-1,1:-1]/1000).ravel())
    ax[0].text(0.5, 1.100, "Timestep: %s (%s s)"%(i*skip,del_t),fontsize=18,
            bbox={'facecolor': 'white'}, #,'alpha': 0.5, 'pad': 5
            transform=ax[1].transAxes, ha="center")
    return stream,phyto1,phyto2,nut

# Set up labels
if alter_vort == 0:
    eddy_type = 'coherent'
elif alter_vort == 1:
    eddy_type = 'leaky'

# Run & save the animation
if numP == 1:
    anim = FuncAnimation(fig, animate_NP, interval=10, frames=int(len(all_P[0])/skip))
    anim.save('./animations/NP_%s_%s_eddy.gif'%(P_combo,eddy_type))

elif numP == 2:
    anim = FuncAnimation(fig, animate_NPP, interval=10, frames=int(len(all_P[0])/skip))
    anim.save('./animations/NPP_%s_%s_eddy.gif'%(P_combo,eddy_type))

#################### FIGURES ####################

## Total concentration figure
if numP == 1:
    phyto_sum,nut_sum = [],[]
    for i in np.arange(0,len(all_P[0])):
        nut_sum.append(np.sum(N_2D[i])/1000)
        phyto_sum.append(np.sum(all_P[0][i])/1000)
elif numP == 2:
    phytoS_sum,phytoL_sum,nut_sum = [],[],[]
    for i in np.arange(0,len(all_P[0])):
        nut_sum.append(np.sum(N_2D[i])/1000)
        phytoS_sum.append(np.sum(all_P[0][i])/1000)
        phytoL_sum.append(np.sum(all_P[1][i])/1000)

fig, ax = plt.subplots(1,2,figsize=(12,5))

if numP == 1:
    ax[0].plot(phyto_sum,label='%s Phytoplankton'%(size[0]))

elif numP == 2:
    ax[0].plot(phytoS_sum,label='Small Phytoplankton')
    ax[0].plot(phytoL_sum,label='Large Phytoplankton')

ax[0].set_title('Total Phytoplankton Concentration')
ax[0].set_xlabel('Time Step (%s s)'%(del_t))
ax[0].set_ylabel('$\sum_{i,j} P^n (\mu M N)$')
ax[0].legend()

ax[1].plot(nut_sum)
ax[1].set_title('Total Nitrate Concentration')
ax[1].set_xlabel('Time Step (%s s)'%(del_t))
ax[1].set_ylabel('$\sum_{i,j} N^n (\mu M N)$')

if numP == 1:
    plt.savefig('./figs/NP_conc_sum_%s_%s_eddy.png'%(P_combo,eddy_type))

elif numP == 2:
    plt.savefig('./figs/NPP_conc_sum_%s_%s_eddy.png'%(P_combo,eddy_type))
