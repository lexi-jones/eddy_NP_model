# Version 2 gives the option to spin the vorticity up and down
## Lexi Jones
## Last edited: 05/16/22

import numpy as np

####################### UNIVERSAL FUNCTIONS #############################
def reformat_1D_to_2D(array,Lx,Ly):
    array = np.array(array)
    reformatted_array = []
    for m in np.arange(0,len(array)):
        temp = []
        for n in np.arange(0,Ly):
            temp.append(array[m][n*Lx:n*Lx+Lx])
        reformatted_array.append(temp)
    reformatted_array = np.array(reformatted_array)
    return reformatted_array

####################### FUNCTIONS FOR FLOW FIELD #############################
def grid_cell_edges_in_meters(L,del_m):
    """
    L: grid cell length of the dimension
    del_m: size of grid cell in meters

    returns the distance of the cell edges (grid_edges) in meters from the first grid cell
    and the location in meters of center of the grid (grid_center)
    """
    grid_edges = [0.0]
    for c in np.arange(1,L):
        grid_edges.append(grid_edges[c-1]+del_m)
    grid_edges = np.array(grid_edges)
    grid_center = grid_edges[-1]/2

    return grid_edges,grid_center

def Acoefficients_elliptic_streamfunction(Lx,Ly,del_x,del_y,time,spinup,alter_vort):
    """
    del_x: size of grid cell in x direction (m)
    del_y: size of grid cell in y direction (m)
    time: time past since start of sim
    spinup: [0,1] if 0, NO spin-up/spin-down; if 1, spin-up will happen until peak at 6 weeks and then spin-down
    alter_vort: [0,1] if 0, vorticity will NOT vary; if 1, vorticity will vary

    returns arrays of size (Lx * Ly)
    """
    # Set up A matrix coefficients
    A_ijminus1, A_ijplus1, A_iminus1j, A_iplus1j, A_ij, vort = [],[],[],[],[],[]

    # Get center cells for vorticity
    x_grid,xc = grid_cell_edges_in_meters(Lx,del_x) #center x-coord of grid
    y_grid,yc = grid_cell_edges_in_meters(Ly,del_y) #center x-coord of grid

    # Set up vorticity equation
    if spinup == 0:
        A = del_x/2
    else:
        frac_of_peak = time/3628800
        if frac_of_peak < 1:
            A = (del_x/2)*frac_of_peak
        else:
            A = (del_x/2)*(1 + (1-frac_of_peak))

    if alter_vort == 0: # No vorticity deviations
        B,C = 1,1
    else: # Add random vorticity deviations
        B_mag,C_mag = np.random.rand(2)#/10 # get random magnitude of change for coefficients; NOTE this may need to change when del_x / del_z change
        binary = np.random.randint(2, size=(2))
        B_sign,C_sign = np.where(binary<1,-1,binary) # get random sign of change for coefficients
        B = (1 + B_sign*B_mag)*2
        C = (1 + C_sign*C_mag)*2

    delx_over_dely2 = del_x/(del_y**2)
    dely_over_delx2 = del_y/(del_x**2)

    # Solve the A coefficients
    for j in np.arange(0,Ly):
        for i in np.arange(0,Lx):
            A_ijminus1.append(delx_over_dely2)
            A_ijplus1.append(delx_over_dely2)
            A_iminus1j.append(dely_over_delx2)
            A_iplus1j.append(dely_over_delx2)
            A_ij.append((-2*delx_over_dely2) + (-2*dely_over_delx2))
            try:
                value = A/(np.sqrt(B*((x_grid[i]-xc)**2) + C*((y_grid[j]-yc)**2)))
                if np.isinf(value):
                    vort.append(A)
                else:
                    vort.append(value)
            except: # exact center of circle will be 0, cannot divide by 0
                vort.append(A)

    return A_ijminus1, A_ijplus1, A_iminus1j, A_iplus1j, A_ij, vort

def Atimesvector_2D_nonperiodic(Lx, Ly, Aijminus1, Aijplus1, Aiminus1j, Aiplus1j, Aij, di):
    """
    Return A*vector, where A is a pentadiagnoal matrix (Lx * Ly) and a vector is being muliplied to the matrix.

    Lx: number of grid cells in the x dir
    Ly: number of grid cells in the y dir
    Aijminus1, Aijplus1, Aiminus1j, Aiplus1j, Aij: Coefficients for the A matrix, vectors of length Lx * Ly
    di: vector of length Lx * Ly

    """

    Axvector = []
    for j in np.arange(0,Ly):
        for i in np.arange(0,Lx):
            ij_ind = j*Lx + i
            ijminus1_ind = (j-1)*Lx + i
            ijplus1_ind = (j+1)*Lx + i
            iminus1j_ind = j*Lx + i - 1
            iplus1j_ind = j*Lx + i + 1

            # south to north index j
            if j == 0: #south side; j-1 term gone
                ijminus1_term = 0
                ijplus1_term = Aijplus1[ij_ind]*di[ijplus1_ind]
            elif j == (Ly - 1): #north side; j+1 term gone
                ijminus1_term = Aijminus1[ij_ind]*di[ijminus1_ind]
                ijplus1_term = 0
            else: #middle
                ijminus1_term = Aijminus1[ij_ind]*di[ijminus1_ind]
                ijplus1_term = Aijplus1[ij_ind]*di[ijplus1_ind]

            # west to east index i
            if i == 0: # west side; i-1 term gone
                iminus1j_term = 0
                iplus1j_term = Aiplus1j[ij_ind]*di[iplus1j_ind]
            elif: # east side; i+1 term gone
                iminus1j_term = Aiminus1j[ij_ind]*di[iminus1j_ind]
                iplus1j_term = 0
            else: #middle
                iminus1j_term = Aiminus1j[ij_ind]*di[iminus1j_ind]
                iplus1j_term = Aiplus1j[ij_ind]*di[iplus1j_ind]

            Axvector.append(ijminus1_term + ijplus1_term + iminus1j_term + iplus1j_term + Aij[ij_ind]*di[ij_ind])
    return Axvector

def krylov_solve(ri,di,Adi,Pm):
    """
    Conjugate gradient Krylov method

    ri: list of residuals
    di: list of search directions
    Adi: list of A matrix times di
    Pm: solution from previous iteration
    """

    riT = np.array([ri]) # horizontal vector
    ri_vertical = np.transpose(riT) # vertical vector
    diT = np.array([di]) # horizontal vector
    Adi_vertical = np.transpose(np.array([Adi])) # vertical vector

    alpha = float(np.dot(riT,ri_vertical)/np.dot(diT,Adi_vertical)) #new search direction

    # Now we go back to using the standard format array di (a list)
    Pm1 = Pm + [alpha*i for i in di]

    # Set up the next di & ri
    ri1 = ri - [alpha*i for i in Adi]
    ri1T = np.array([ri1]) # horizontal vector
    ri1_vertical = np.transpose(ri1T) # vertical vector

    # Beta ensures di+1 is orthogonal to d1,d2,..
    betai1 = float(np.dot(ri1T,ri1_vertical)/np.dot(riT,ri_vertical))
    di1 = ri1 + [betai1*i for i in di]

    return ri1,di1,Pm1

def krylov_solve_2D_elliptic_streamfunction(Lx,Ly,del_x,del_y,time,spinup,alter_vort):
    """
    alter_vort: [0,1] if 0, vorticity will NOT vary; if 1, vorticity will vary

    returns array of size (Lx * Ly) streamfunction psi
    """
    A_ijminus1, A_ijplus1, A_iminus1j, A_iplus1j, A_ij, vort = Acoefficients_elliptic_streamfunction(Lx,Ly,del_x,del_y,time,spinup,alter_vort) # A matrix coefficients
    psi = np.full((Lx*Ly),0.0) #First guess

    # Initialize di & ri vectors with RHS - AP0
    di = np.subtract(vort,psi)
    ri = np.subtract(vort,psi)

    RMS = 1
    tol = 10**(-8)
    count = 0
    while RMS > tol:
        #Conjugate Gradient - Krylov method
        # First calculate A*di (di is assumed to be a vertical vector)
        Adi = Atimesvector_2D_nonperiodic(Lx, Ly, A_ijminus1, A_ijplus1, A_iminus1j, A_iplus1j, A_ij, di)
        ri1,di1,psi1 = krylov_solve(ri,di,Adi,psi)

        # Calculate the residuals
        residuals = [(psi1[i]-psi[i])**2 for i in np.arange(0,len(psi))]
        RMS = ((1/len(residuals))*np.sum(residuals))**(1/2)

        count += 1
        if count == 20000: #safety
            break

        di = di1.copy()
        ri = ri1.copy()
        psi = psi1.copy()
    return psi

def velocity_from_streamfunction(psi,Lx,Ly,del_x,del_y):
    u,v = [],[]
    for j in np.arange(0,Ly):
        for i in np.arange(0,Lx):

            ij_ind = j*Lx + i
            ijplus1_ind = (j+1)*Lx + i
            iplus1j_ind = j*Lx + i + 1

            if j == Ly-1:
                u.append(np.nan)
            else:
                u.append((psi[ij_ind]-psi[ijplus1_ind])/del_y)

            if i == Lx-1:
                v.append(np.nan)
            else:
                v.append((psi[iplus1j_ind]-psi[ij_ind])/del_x)
    return u,v

####################### FUNCTIONS FOR ADVECTION AND DIFFUSION #############################
def abs_value_avg(vel):
    """
    vel: velocity
    """
    vel_plus = (vel + np.abs(vel))/2
    vel_minus = (vel - np.abs(vel))/2

    return vel_plus,vel_minus

def upwind_fluxes_2D(phi_n,Lx,i,j,u,v):
    """
    Solve the 1st order upwind fluxes for advection
    """

    #phi-specific indeces, grid length Lx-1
    ij_ind = j*(Lx-1) + i
    iminus1j_ind = j*(Lx-1) + i - 1
    iplus1j_ind = j*(Lx-1) + i + 1
    ijminus1_ind = (j-1)*(Lx-1) + i
    ijplus1_ind = (j+1)*(Lx-1) + i

    # Indeces for u, w vel
    ij_vel_ind = j*Lx + i
    ijplus1_vel_ind = (j+1)*Lx + i
    iplus1j_vel_ind = j*Lx + i + 1

    uw = u[ij_vel_ind]
    ue = u[iplus1j_vel_ind]
    vs = v[ij_vel_ind]
    vn = v[ijplus1_vel_ind]

    uw_plus,uw_minus = abs_value_avg(uw)
    ue_plus,ue_minus = abs_value_avg(ue)
    vs_plus,vs_minus = abs_value_avg(vs)
    vn_plus,vn_minus = abs_value_avg(vn)

    Fw = uw_plus*phi_n[iminus1j_ind] + uw_minus*phi_n[ij_ind]
    Fe = ue_plus*phi_n[ij_ind] + ue_minus*phi_n[iplus1j_ind]
    Fs = vs_plus*phi_n[ijminus1_ind] + vs_minus*phi_n[ij_ind]
    Fn = vn_plus*phi_n[ij_ind] + vn_minus*phi_n[ijplus1_ind]

    return Fw,Fe,Fs,Fn

def diffusive_fluxes_2D(phi_n,Lx,i,j,del_x,del_y):

    kappa = 10**(-2) #m2/s

    #phi-specific indeces, grid length Lx-1
    ij_ind = j*(Lx-1) + i
    iminus1j_ind = j*(Lx-1) + i - 1
    iplus1j_ind = j*(Lx-1) + i + 1
    ijminus1_ind = (j-1)*(Lx-1) + i
    ijplus1_ind = (j+1)*(Lx-1) + i

    Fe = kappa*del_y*((phi_n[iplus1j_ind] - phi_n[ij_ind])/del_x)
    Fw = kappa*del_y*((phi_n[ij_ind] - phi_n[iminus1j_ind])/del_x)
    Fn = kappa*del_x*((phi_n[ijplus1_ind] - phi_n[ij_ind])/del_y)
    Fs = kappa*del_x*((phi_n[ij_ind] - phi_n[ijminus1_ind])/del_y)

    return Fw,Fe,Fs,Fn

####################### MODEL SIMULATION #############################

#Phytoplankton constants
mu_max_S = 1.4/(24*60*60) #max growth rate of small pp; d^-1, converted to s^-1
mu_max_L = 2.5/(24*60*60) #max growth rate of large pp
kN_S = 0.24*1000#half saturation, converted mu M to mu mol N m^-2 (assuming grid cell depth of 1 m)
kN_L = 0.24*1000
N_star_S = (m*kN_S)/(mu_max_S - m) #equilibrium resource requirement
N_star_L = (m*kN_L)/(mu_max_L - m)

def FE_upwind_2D_adv_diff_eddy_NP_model(Lx,Ly,del_x,del_y,del_t,num_steps,alter_vort,P_combo,death_rate):
    """
    Lx,Ly: Number of grid cells in the x and y directions
    del_x,del_y: spatial step size (in meters) in the x and y directions
    del_t: time step
    num_steps: number of time steps to take
    alter_vort: [0,1] if 0, vorticity will NOT vary; if 1, vorticity will vary
    death_rate: constant; d^-1
    P_combo: Combination of phytoplankton in model:
            - 'L': 1 large phytoplankton cell type
            - 'S': 1 small phytoplankton cell type
    """

    # Set up background nutrient concentration and supply rate
    m = death_rate/(24*60*60)
    P_star = 0.1*1000 # mu mol N m^-2
    if P_combo == 'S':
        N_star = N_star_S
        SN_star = (mu_max_S*N_star_S*P_star)/(N_star_S + kN_S)
    elif P_combo == 'L':
        N_star = N_star_L
        SN_star = (mu_max_L*N_star*P_star)/(N_star + kN_L)

    # Initialize the tracers & set up the matrices to save the data
    N_n = np.full(((Lx-1)*(Ly-1)),N_star) #nutrients initial & boundary conditions
    P_n = np.full(((Lx-1)*(Ly-1)),P_star)
    N_mat,P_mat = [],[]
    N_mat.append(N_n)
    P_mat.append(P_n)

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

def FE_upwind_2D_adv_diff_eddy_NPP_model(Lx,Ly,del_x,del_y,del_t,num_steps,alter_vort,P_combo,death_rate):
    """
    Lx,Ly: Number of grid cells in the x and y directions
    del_x,del_y: spatial step size (in meters) in the x and y directions
    del_t: time step
    num_steps: number of time steps to take
    alter_vort: [0,1] if 0, vorticity will NOT vary; if 1, vorticity will vary
    death_rate: constant; d^-1
    P_combo: Combination of phytoplankton in model:
            - 'SLe': 1 large + 1 small; even initial concentrations
            - 'SLu': 1 large + 1 small; uneven initial concentrations
    """

    # Set up background nutrient concentration and supply rate
    m = death_rate/(24*60*60)
    N_star = N_star_S + N_star_L
    P_star_S = 0.1*1000
    if P_combo == 'SLe': #even initial concentrations of small & large
        P_star_L = 0.1*1000
    elif P_combo == 'SLu': # large population is 1/5th the size of the small
        P_star_L = (0.1/5)*1000
    SN_star = (mu_max_S*N_star*P_star_S)/(N_star + kN_S) + (mu_max_L*N_star*P_star_L)/(N_star + kN_L)

    # Initialize the tracers & set up the matrices to save the data
    PL_n = np.full(((Lx-1)*(Ly-1)),P_star_L) #large phytoplankton
    PS_n = np.full(((Lx-1)*(Ly-1)),P_star_S) #small phytoplankton
    N_n = np.full(((Lx-1)*(Ly-1)),N_star) #nutrients
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
