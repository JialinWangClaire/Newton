import scipy as sc
import scipy.sparse as sparse
import scipy.sparse.linalg
import scipy.interpolate as interpolate
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import time


"""# Parameters and useful functions"""
""" ************************************************** """
""" ************** USEFUL FUNCTIONS ****************** """
""" ************************************************** """
# Gaussian density
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
def gaussianN(x):
    return gaussian(x, 0, 1)
# positive part
def ppart(x):
    return np.max(x,0)
# negative part (with minus)
def npart(x):
    return -np.min(x,0)


""" ************************************************** """
""" ***************** PARAMETERS ********************* """
""" ************************************************** """
coefCT = 1./2.
xmin =  0.0
xmax =  1.0
Lx = xmax - xmin
Nx =    50 # number of subintervals
kx =    2 # subsampling points for plots
Dx =    (xmax-xmin)/Nx
T =     0.5
Nt =    50 # number of subintervals
kt =    5 # subsampling of points for plots
Dt =    T/Nt
xSpace =    np.linspace(xmin, xmax, Nx, endpoint=True)
tSpace =    np.linspace(0, T, Nt+1, endpoint=True)
xSpacecut = np.linspace(xmin, xmax, Nx//kx, endpoint=True)
tSpacecut = np.linspace(0, T, Nt//kt+1, endpoint=True)
sigma = 1.0 # diffusion


""" ************************************************** """
""" ***************** LOSS FUNCTION ********************* """
""" ************************************************** """

f =     np.zeros(Nx) # running cost
h =     np.zeros(Nx) # final cost
rho0 =  np.zeros(Nx) # initial density
# COST
for i in range(Nx):
    xi = xmin + i*Dx
    def potential(x):
        return 50* (0.1*np.cos(x*2*np.pi) + np.cos(x*4*np.pi) + 0.1*np.sin((x-np.pi/8)*2*np.pi))
    f[i] = potential(xi)
# INITIAL DENSITY
for i in range(Nx):
    xi = xmin + i*Dx
    rho0[i] = gaussian(xi, Lx/2., (xmax-xmin)/10.)
    rho0[i] = max(rho0[i]-0.05,0) # truncate to have Dirichlet BC = 0
rho0 /= sum(rho0*Dx)


# Total mass as function of time
def getmasstot(m):
    return np.sum(m*Dx, axis=1)

def getuFinal_simple():
    return h


""" ************************************************** """
""" ***************** NEWTON'S METHOD ********************* """
""" ************************************************** """

# AUU
def getAUU(sigma, Uk_n, Ukd1_n, Ukp1_n):
   AUU = np.zeros((Nt**2, Nx**2))
   # Thinking about terminal and boundary conditions
   p1 = npart((Ukp1_n - Uk_n) / Dx)
   p2 = ppart((Uk_n - Ukd1_n) / Dx)
   for i in range(Nt**2):
       for j in range(Nx**2):
           if i == j:
               if p1 < 0 and p2 < 0:
                   AUU[i][j] = 1 / Dt + (sigma ** 2)/(Dx ** 2) + (p1 + p2)(-2 / Dx)
               if p1*p2 < 0:
                   AUU[i][j] = 1 / Dt + (sigma ** 2) / (Dx ** 2)
               if p1 > 0 and p2 > 0:
                   AUU[i][j] = 1 / Dt + (sigma ** 2) / (Dx ** 2) + (p1 + p2)(2 / Dx)
           if j - i == 1 and i % Nx != Nx - 1:
               if p2 < 0:
                   AUU[i][j] = -(sigma ** 2)/(2*Dx ** 2) + (p1 + p2)(1 / Dx)
               if p2 > 0:
                   AUU[i][j] = -(sigma ** 2) / (2*Dx ** 2) + (p1 + p2)(-1 / Dx)
           if i - j == 1 and i % Nx != 0:
               if p1 < 0:
                   AUU[i][j] = -(sigma ** 2)/(2*Dx ** 2) + (p1 + p2)(1 / Dx)
               if p1 > 0:
                   AUU[i][j] = -(sigma ** 2) / (2*Dx ** 2) + (p1 + p2)(-1 / Dx)
           if i - j == Nx:
               AUU[i][j] = -1/Dt
   AUU_sparse = sparse.csr_matrix(AUU)
   return AUU_sparse


# AMM
def getAMM(sigma, Uk_n, Ukd1_n, Ukd2_n, Ukp1_n, Ukp2_n):
    AMM = np.zeros((Nt ** 2, Nx ** 2))
    # Thinking about terminal and boundary conditions
    p1 = npart((Ukp1_n - Uk_n) / Dx)
    p2 = ppart((Uk_n - Ukd1_n) / Dx)
    p3 = npart((Uk_n - Ukd1_n) / Dx)
    p4 = ppart((Ukd1_n - Ukd2_n) / Dx)
    p5 = npart((Ukp2_n - Ukp1_n) / Dx)
    p6 = ppart((Ukp1_n - Uk_n) / Dx)
    for i in range(Nt ** 2):
        for j in range(Nx ** 2):
            if i == j:
                AMM[i][j] = -1/Dt
            if i - j == Nx:
                if p1*p2 > 0:
                    AMM[i][j] = 1/Dt - (sigma**2)/(Dx**2)
                if p1 < 0 and p2 > 0:
                    AMM[i][j] = 1 / Dt - (sigma ** 2) / (Dx ** 2) - (1/Dx)(2/Dx)(p1 + p2)
                if p1 > 0 and p2 < 0:
                    AMM[i][j] = 1 / Dt - (sigma ** 2) / (Dx ** 2) + (1 / Dx)(2 / Dx)(p1 + p2)
            if i - j == Nx - 1 and i % Nx != Nx - 1:
                if p3 < 0:
                    AMM[i][j] = (sigma ** 2) / (2*Dx ** 2) - (1/Dx)(p3 + p4)(1/Dx)
                if p3 > 0:
                    AMM[i][j] = (sigma ** 2) / (2 * Dx ** 2) - (1 / Dx)(p3 + p4)(-1 / Dx)
            if i - j == Nx + 1 and i % Nx != 0:
                if p4 < 0:
                    AMM[i][j] = (sigma ** 2) / (2*Dx ** 2) - (1/Dx)(p5 + p6)(1/Dx)
                if p4 > 0:
                    AMM[i][j] = (sigma ** 2) / (2 * Dx ** 2) - (1 / Dx)(p5 + p6)(-1 / Dx)
    AMM_sparse = sparse.csr_matrix(AMM)
    return AMM_sparse


# AMU
def getAMU(Uk_n, Ukd1_n, Ukd2_n, Ukp1_n, Ukp2_n, Mk_np1, Mkp1_np1, Mkd1_np1):
    AMU = np.zeros((Nt ** 2, Nx ** 2))
    # Thinking about terminal and boundary conditions
    p1 = npart((Ukp1_n - Uk_n) / Dx)
    p2 = ppart((Uk_n - Ukd1_n) / Dx)
    p3 = npart((Uk_n - Ukd1_n) / Dx)
    p4 = ppart((Ukd1_n - Ukd2_n) / Dx)
    p5 = npart((Ukp2_n - Ukp1_n) / Dx)
    p6 = ppart((Ukp1_n - Uk_n) / Dx)
    for i in range(Nt ** 2):
        for j in range(Nx ** 2):
            if i == j:
                if p3 < 0 and p6 < 0:
                    AMU[i][j] = (-1 / Dx)(- Mkd1_np1 * (1 / Dx) + Mkp1_np1 * (1 / Dx))
                if p3 < 0 and p6 > 0:
                    AMU[i][j] = (-1 / Dx)(- Mkd1_np1 * (1 / Dx) + Mkp1_np1 * (-1 / Dx))
                if p3 > 0 and p6 < 0:
                    AMU[i][j] = (-1 / Dx)(- Mkd1_np1 * (-1 / Dx) + Mkp1_np1 * (1 / Dx))
                if p3 > 0 and p6 > 0:
                    AMU[i][j] = (-1 / Dx)(- Mkd1_np1 * (-1 / Dx) + Mkp1_np1 * (-1 / Dx))
            if j - i  == 2 and i % Nx != Nx - 1 and i % Nx != Nx - 2:
                if p4 < 0:
                    AMU[i][j] = (-1 / Dx)(- Mkd1_np1 * (1 / Dx))
                if p4 > 0:
                    AMU[i][j] = (-1 / Dx)(- Mkd1_np1 * (-1 / Dx))
            if j - i  == 1 and i % Nx != Nx - 1:
                if p3 * p4 < 0:
                    AMU[i][j] = 0
                if p3 < 0 and p4 < 0:
                    AMU[i][j] = (-1 / Dx)(- Mkd1_np1 * (-2 / Dx))
                if p3 > 0 and p4 > 0:
                    AMU[i][j] = (-1 / Dx)(- Mkd1_np1 * (2 / Dx))
            if i - j  == 1 and i % Nx != 0:
                if p5 * p6 < 0:
                    AMU[i][j] = 0
                if p5 < 0 and p6 < 0:
                    AMU[i][j] = (-1 / Dx)(Mkp1_np1 * (-2 / Dx))
                if p5 > 0 and p6 > 0:
                    AMU[i][j] = (-1 / Dx)(Mkp1_np1 * (2 / Dx))
            if i - j  == 1 and i % Nx != 0 and i % Nx != 1:
                if p5 < 0:
                    AMU[i][j] = (-1 / Dx)(Mkp1_np1 * (1 / Dx))
                if p5 > 0:
                    AMU[i][j] = (-1 / Dx)(Mkp1_np1 * (-1 / Dx))
    AMU_sparse = sparse.csr_matrix(AMU)
    return AMU_sparse

# AUM
def getAUM():
    AUM = np.zeros((Nt ** 2, Nx ** 2))
    AUM_sparse = sparse.csr_matrix(AUM)
    return AUM_sparse

# getFnU
def getFnU_withoutM(sigma, Uk_n, Uk_np1, Ukp1_n, Ukd1_n):
    FnU = np.zeros(Nx)
# getFnM
def getFnM_withU(sigma, Mk_n, Mk_np1, Mkp1_np1, Mkd1_np1, Uk_n, Ukp1_n, Ukd1_n, Ukd2_n, Ukp2_n):


""" ************************************************** """
""" **************** Newton Iteration ******************* """
""" ************************************************** """



def newton_solve(iter_num, sigma, Uinit, Minit):
    # initial settings
    U = Uinit
    M = Minit
    for i in range(iter_num):
        AUU = getAUU(sigma, Uk_n, Ukd1_n, Ukp1_n)
        AUM = getAUM()
        AMU = getAMU(Uk_n, Ukd1_n, Ukd2_n, Ukp1_n, Ukp2_n, Mk_np1, Mkp1_np1, Mkd1_np1)
        AMM = getAMM(sigma, Uk_n, Ukd1_n, Ukd2_n, Ukp1_n, Ukp2_n)
        # combine matrices
        horizontal1 = sparse.hstack([AUU, AUM])
        horizontal2 = sparse.hstack([AMU, AMM])
        whole_A = sparse.vstack([horizontal1, horizontal2])
        # solve
        B =
        sol = sparse.linalg.spsolve(whole_A, B)

        # update
        U_new = sol + U
        M_new = sol + M
        U = U_new
        M = M_new


    return U, M

# INITIALIZATION
Minit = np.zeros((Nx**2+))
for n in range(Nt+1):
    Minit[n] = rho0
Minit_tmp = Minit
Uinit = np.zeros((Nt+1,Nx))
for n in range(Nt+1):
    Uinit[n] = getuFinal_simple()

print(newton_solve(1000, 0.1, Uinit, Minit))

