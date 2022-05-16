import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model_functions_project_v2 import krylov_solve_2D_elliptic_streamfunction,velocity_from_streamfunction,upwind_fluxes_2D,diffusive_fluxes_2D,reformat_1D_to_2D


def FE_upwind_2D_spinup_adv_diff_3tracers(Lx,Ly,del_x,del_y,del_t,num_steps,alter_vort):
    """
    u: velocity function
    phi_0: western boundary condition
    phi_m: easter boundary condition
    del_x: spatial step
    del_t: time step
    num_steps: number of time steps to take
    alter_vort: [0,1] if 0, vorticity will NOT vary; if 1, vorticity will vary
    """

    mu_max_L = 2.5/(24*60*60) #max growth rate of large pp; d^-1, convert to s^-1
    mu_max_S = 1.4/(24*60*60) #max growth rate of small pp;

    kN_L = 0.56*1000 #half saturation
    kN_S = 0.24*1000

    m = 0.5/(24*60*60) #death rate (keeping constant for both rn)

    N_star_S = (m*kN_S)/(mu_max_S - m)
    N_star_L = (m*kN_L)/(mu_max_L - m)
    N_star = N_star_S + N_star_L

    P_star_S = 0.1*1000
    P_star_L = 0.01*1000

    SN_star = (mu_max_S*N_star*P_star_S)/(N_star + kN_S) + (mu_max_L*N_star*P_star_L)/(N_star + kN_L)

    #Initialize tracers
    PL_n = np.full(((Lx-1)*(Ly-1)),P_star_L) #large phytoplankton
    PS_n = np.full(((Lx-1)*(Ly-1)),P_star_S) #small phytoplankton
    N_n = np.full(((Lx-1)*(Ly-1)),N_star) #nutrients

    # Set up matrix to save the data
    PL_mat,PS_mat,N_mat = [],[],[]
    PL_mat.append(PL_n)
    PS_mat.append(PS_n)
    N_mat.append(N_n)

    # Set up the streamfunction and velocity field
    time = del_t # initialize with 1 time step to avoid errors with velocity being 0
    psi = krylov_solve_2D_elliptic_streamfunction(Lx,Ly,del_x,del_y,time,1,0)
    u,v = velocity_from_streamfunction(psi,Lx,Ly,del_x,del_y)
    psi_mat = []
    psi_mat.append(psi)

    # Time step forward delta t to find phi_n+1
    for t in np.arange(0,num_steps): # number of time steps
        time = time + del_t

        if t%100 == 0:
            print(t)

        PL_n1 = PL_n.copy()
        PS_n1 = PS_n.copy()
        N_n1 = N_n.copy()

        psi = krylov_solve_2D_elliptic_streamfunction(Lx,Ly,del_x,del_y,time,1,alter_vort)
        psi_mat.append(psi)
        u,v = velocity_from_streamfunction(psi,Lx,Ly,del_x,del_y)

        for j in np.arange(1,Ly-2):
            for i in np.arange(1,Lx-2): # only iterate after the first cell and before the last

                # Indeces for phi matrix
                ij_ind = j*(Lx-1) + i # Index for tracers
                ij_psi_ind = j*Lx + i #Index for streamfunction

                # Solve the advective tracer fluxes
                PL_Aw,PL_Ae,PL_As,PL_An = upwind_fluxes_2D(PL_n,Lx,i,j,u,v)
                PS_Aw,PS_Ae,PS_As,PS_An = upwind_fluxes_2D(PS_n,Lx,i,j,u,v)
                N_Aw,N_Ae,N_As,N_An = upwind_fluxes_2D(N_n,Lx,i,j,u,v)

                # Solve the diffusive
                PL_Dw,PL_De,PL_Ds,PL_Dn = diffusive_fluxes_2D(PL_n,Lx,i,j,del_x,del_y)
                PS_Dw,PS_De,PS_Ds,PS_Dn = diffusive_fluxes_2D(PS_n,Lx,i,j,del_x,del_y)
                N_Dw,N_De,N_Ds,N_Dn = diffusive_fluxes_2D(N_n,Lx,i,j,del_x,del_y)

                # Solve tracers at the next time step
                PL_n1[ij_ind] = PL_n[ij_ind] + del_t*(((PL_Aw-PL_Ae)/del_x) + ((PL_As-PL_An)/del_y) + ((PL_De-PL_Dw)/del_x) + ((PL_Dn-PL_Ds)/del_y) + ((mu_max_L*N_n[ij_ind])/(N_n[ij_ind] + kN_L) - m)*PL_n[ij_ind])
                PS_n1[ij_ind] = PS_n[ij_ind] + del_t*(((PS_Aw-PS_Ae)/del_x) + ((PS_As-PS_An)/del_y) + ((PS_De-PS_Dw)/del_x) + ((PS_Dn-PS_Ds)/del_y) + ((mu_max_S*N_n[ij_ind])/(N_n[ij_ind] + kN_S) - m)*PS_n[ij_ind])

                if psi[ij_psi_ind] < -10000: #when streamfunction is high add nutrients
                    SN = 3*SN_star
                else:
                    SN = SN_star

                N_n1[ij_ind] = N_n[ij_ind] + del_t*(((N_Aw-N_Ae)/del_x) + ((N_As-N_An)/del_y) + ((N_De-N_Dw)/del_x) + ((N_Dn-N_Ds)/del_y) + SN - ((mu_max_L*N_n[ij_ind])/(N_n[ij_ind] + kN_L))*PL_n[ij_ind] - ((mu_max_S*N_n[ij_ind])/(N_n[ij_ind] + kN_S))*PS_n[ij_ind])

        PL_mat.append(PL_n1)
        PS_mat.append(PS_n1)
        N_mat.append(N_n1)

        PL_n = PL_n1.copy()
        PS_n = PS_n1.copy()
        N_n = N_n1.copy()

    return PL_mat,PS_mat,N_mat,psi_mat,N_star,P_star_S,P_star_L

Lx,Ly = 50,50 # Lx & Ly are the dimensions of psi, so the tracer has dimension (Lx-1,Ly-1)
del_x,del_y,del_t = 2000,2000,3000
num_steps = 2419 ######## EDDY PEAKS @ 6 WEEKS = 1209 timesteps, EDDY DIES @ 12 WEEKS = 2419 timesteps

PL,PS,N,psi,N_star,P_star_S,P_star_L = FE_upwind_2D_spinup_adv_diff_3tracers(Lx,Ly,del_x,del_y,del_t,num_steps,1)

PS_2D = reformat_1D_to_2D(PS,Lx-1,Ly-1)
PL_2D = reformat_1D_to_2D(PL,Lx-1,Ly-1)
N_2D = reformat_1D_to_2D(N,Lx-1,Ly-1)
psi_2D = reformat_1D_to_2D(psi,Lx,Ly)

#################### ANIMATION ####################
fig, ax = plt.subplots(1,4,figsize=(28,5))

stream = ax[0].pcolormesh(psi_2D[0],vmin=-20000,vmax=0)
ax[0].set_title('Stream Function ($\psi$)',fontsize=18)
ax[0].set_xlabel('x index (%sm)'%(del_x),fontsize=18)
ax[0].set_ylabel('y index (%sm)'%(del_y),fontsize=18)
cbar = fig.colorbar(stream, ax=ax[0])

phytoL = ax[1].pcolormesh(PL_2D[0,1:-1,1:-1]/1000,vmin=0,vmax=(P_star_L/1000)*5)
ax[1].set_title('Large PP Concentration',fontsize=18)
ax[1].set_xlabel('x index (%sm)'%(del_x),fontsize=18)
ax[1].set_ylabel('y index (%sm)'%(del_y),fontsize=18)
cbar = fig.colorbar(phytoL, ax=ax[1])

phytoS = ax[2].pcolormesh(PS_2D[0,1:-1,1:-1]/1000,vmin=0,vmax=(P_star_S/1000)*5)
ax[2].set_title('Small PP Concentration',fontsize=18)
ax[2].set_xlabel('x index (%sm)'%(del_x),fontsize=18)
ax[2].set_ylabel('y index (%sm)'%(del_y),fontsize=18)
cbar = fig.colorbar(phytoS, ax=ax[2])

nut = ax[3].pcolormesh(N_2D[0,1:-1,1:-1]/1000,vmin=0.0,vmax=(N_star/1000)*3)
ax[3].set_title('Nutrient Concentration',fontsize=18)
ax[3].set_xlabel('x index (%sm)'%(del_x),fontsize=18)
ax[3].set_ylabel('y index (%sm)'%(del_y),fontsize=18)
cbar = fig.colorbar(nut, ax=ax[3])

skip = 20

def animate(i):
    stream.set_array(psi_2D[i*skip].ravel())
    phytoL.set_array((PL_2D[i*skip,1:-1,1:-1]/1000).ravel())
    phytoS.set_array((PS_2D[i*skip,1:-1,1:-1]/1000).ravel())
    nut.set_array((N_2D[i*skip,1:-1,1:-1]/1000).ravel())

    ax[0].text(0.5, 1.100, "Timestep: %s; $\Delta t = %s s$"%(i*skip,del_t),fontsize=18,
            bbox={'facecolor': 'white'}, #,'alpha': 0.5, 'pad': 5
            transform=ax[1].transAxes, ha="center")

    return stream,phytoL,phytoS,nut

anim = FuncAnimation(fig, animate, interval=10, frames=int(len(PL_2D)/skip))
anim.save('two_phytoplankton_nutrients_spinup_vort_dev.gif')

#################### FIGURES ####################

phytoL_sum,phytoS_sum,nut_sum = [],[],[]
for i in np.arange(0,len(PL_2D)):
    phytoL_sum.append(np.sum(PL_2D[i])/1000)
    phytoS_sum.append(np.sum(PS_2D[i])/1000)
    nut_sum.append(np.sum(N_2D[i])/1000)

fig, ax = plt.subplots(1,2,figsize=(12,5))

ax[0].plot(phytoS_sum,label='Small Phytoplankton')
ax[0].plot(phytoL_sum,label='Large Phytoplankton')
ax[0].set_title('Total Phytoplankton Concentrations')
ax[0].set_xlabel('Time Step (%s s)'%(del_t))
ax[0].set_ylabel('$\sum_{i,j} P^n (\mu M N)$')
ax[0].legend()

ax[1].plot(nut_sum)
ax[1].set_title('Total Nitrate Concentration')
ax[1].set_xlabel('Time Step (%s s)'%(del_t))
ax[1].set_ylabel('$\sum_{i,j} N^n (\mu M N)$')

plt.savefig('NPP_concentrations_vort_dev.png')

#np.save('phyto_sum_vort_const.npy',phyto_sum)
#np.save('nut_sum_vort_const.npy',nut_sum)
