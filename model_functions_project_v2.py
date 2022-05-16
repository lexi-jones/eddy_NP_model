# Version 2 gives the option to spin the vorticity up and down

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
    x_grid = [0.0]
    for x in np.arange(1,Lx):
        x_grid.append(x_grid[x-1]+del_x)
    x_grid = np.array(x_grid)
    xc = x_grid[-1]/2 #center x-coord of grid

    y_grid = [0.0]
    for y in np.arange(1,Ly):
        y_grid.append(y_grid[y-1]+del_y)
    y_grid = np.array(y_grid)
    yc = y_grid[-1]/2 #center z-coord of grid

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

            if j == 0: #bottom; j-1 term gone
                if i == 0: # bottom west corner
                    Axvector.append(Aijplus1[ij_ind]*di[ijplus1_ind] + Aiplus1j[ij_ind]*di[iplus1j_ind] + Aij[ij_ind]*di[ij_ind])
                elif i == (Lx - 1): # bottom east corner
                    Axvector.append(Aijplus1[ij_ind]*di[ijplus1_ind] + Aiminus1j[ij_ind]*di[iminus1j_ind] + Aij[ij_ind]*di[ij_ind])
                else: # bottom edge
                    Axvector.append(Aijplus1[ij_ind]*di[ijplus1_ind] + Aiminus1j[ij_ind]*di[iminus1j_ind] + Aiplus1j[ij_ind]*di[iplus1j_ind] + Aij[ij_ind]*di[ij_ind])

            elif j == (Ly - 1): #top; j+1 term gone
                if i == 0: # top west corner
                    Axvector.append(Aijminus1[ij_ind]*di[ijminus1_ind] + Aiplus1j[ij_ind]*di[iplus1j_ind] + Aij[ij_ind]*di[ij_ind])
                elif i == (Lx - 1): # top east corner
                    Axvector.append(Aijminus1[ij_ind]*di[ijminus1_ind] + Aiminus1j[ij_ind]*di[iminus1j_ind] + Aij[ij_ind]*di[ij_ind])
                else: # top edge
                    Axvector.append(Aijminus1[ij_ind]*di[ijminus1_ind] + Aiminus1j[ij_ind]*di[iminus1j_ind] + Aiplus1j[ij_ind]*di[iplus1j_ind] + Aij[ij_ind]*di[ij_ind])

            else:
                if i == 0: #west edge
                    Axvector.append(Aijminus1[ij_ind]*di[ijminus1_ind] + Aijplus1[ij_ind]*di[ijplus1_ind] + Aiplus1j[ij_ind]*di[iplus1j_ind] + Aij[ij_ind]*di[ij_ind])
                elif i == (Lx - 1): #east edge
                    Axvector.append(Aijminus1[ij_ind]*di[ijminus1_ind] + Aijplus1[ij_ind]*di[ijplus1_ind] + Aiminus1j[ij_ind]*di[iminus1j_ind] + Aij[ij_ind]*di[ij_ind])
                else: #middle cells
                    Axvector.append(Aijminus1[ij_ind]*di[ijminus1_ind] + Aijplus1[ij_ind]*di[ijplus1_ind] + Aiminus1j[ij_ind]*di[iminus1j_ind] + Aiplus1j[ij_ind]*di[iplus1j_ind] + Aij[ij_ind]*di[ij_ind])

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
    Solve the 1st order upwind fluxes
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
