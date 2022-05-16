import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model_functions_project_v2 import krylov_solve_2D_elliptic_streamfunction,velocity_from_streamfunction,upwind_fluxes_2D,diffusive_fluxes_2D,reformat_1D_to_2D

def FE_upwind_2D_spinup_adv_diff_2tracers(Lx,Ly,del_x,del_y,del_t,num_steps,alter_vort):
    """
    u: velocity function
    phi_0: western boundary condition
    phi_m: easter boundary condition
    del_x: spatial step
    del_t: time step
    num_steps: number of time steps to take
    alter_vort: [0,1] if 0, vorticity will NOT vary; if 1, vorticity will vary
    """

    #Initialize center tracer
    mu_max = 1.4/(24*60*60) #max growth rate d^-1, converted to s^-1
    kN = 0.24*1000#*del_x*del_y #half saturation, converted mu M to mu mol N m^-2
    m = 0.5/(24*60*60) #death rate; s^-1
    N_star = (m*kN)/(mu_max - m) # mu mol N m^-2
    P_star = 0.1*1000#*del_x*del_y # mu mol N m^-2; 0.1 was an arbitrary choice
    SN_star = (mu_max*N_star*P_star)/(N_star + kN) # m mol N m^-2 s^-1

    P_n = np.full(((Lx-1)*(Ly-1)),P_star) #phytoplankton initial & boundary conditions
    N_n = np.full(((Lx-1)*(Ly-1)),N_star) #nutrients initial & boundary conditions

    # Set up matrix to save the data
    P_mat,N_mat = [],[]
    P_mat.append(P_n)
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

        P_n1 = P_n.copy()
        N_n1 = N_n.copy()

        psi = krylov_solve_2D_elliptic_streamfunction(Lx,Ly,del_x,del_y,time,1,alter_vort)
        psi_mat.append(psi)
        u,v = velocity_from_streamfunction(psi,Lx,Ly,del_x,del_y)

        for j in np.arange(1,Ly-2):
            for i in np.arange(1,Lx-2): # only iterate after the first cell and before the last

                # Indeces for phi matrix
                ij_ind = j*(Lx-1) + i # Index for tracers
                ij_psi_ind = j*Lx + i #Index for streamfunction

                # Solve the tracer fluxes
                P_Aw,P_Ae,P_As,P_An = upwind_fluxes_2D(P_n,Lx,i,j,u,v)
                N_Aw,N_Ae,N_As,N_An = upwind_fluxes_2D(N_n,Lx,i,j,u,v)
                P_Dw,P_De,P_Ds,P_Dn = diffusive_fluxes_2D(P_n,Lx,i,j,del_x,del_y)
                N_Dw,N_De,N_Ds,N_Dn = diffusive_fluxes_2D(N_n,Lx,i,j,del_x,del_y)

                # Solve tracers at the next time step
                P_n1[ij_ind] = P_n[ij_ind] + del_t*(((P_Aw-P_Ae)/del_x) + ((P_As-P_An)/del_y) + ((P_De-P_Dw)/del_x) + ((P_Dn-P_Ds)/del_y) + ((mu_max*N_n[ij_ind])/(N_n[ij_ind] + kN) - m)*P_n[ij_ind])

                if psi[ij_psi_ind] < -10000: #when streamfunction is high increase supply rate
                    SN = 3*SN_star
                else:
                    SN = SN_star

                N_n1[ij_ind] = N_n[ij_ind] + del_t*(((N_Aw-N_Ae)/del_x) + ((N_As-N_An)/del_y) + ((N_De-N_Dw)/del_x) + ((N_Dn-N_Ds)/del_y) + SN - ((mu_max*N_n[ij_ind])/(N_n[ij_ind] + kN))*P_n[ij_ind])

        P_mat.append(P_n1)
        N_mat.append(N_n1)
        P_n = P_n1.copy()
        N_n = N_n1.copy()

    return P_mat,N_mat,psi_mat,N_star,P_star

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

#np.save('phyto_sum_vort_const.npy',phyto_sum)
#np.save('nut_sum_vort_const.npy',nut_sum)
