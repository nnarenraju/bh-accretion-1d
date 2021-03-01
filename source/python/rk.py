# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
rk.py
    Numerical simulation of 1D Hydrodynamic Spherical Accretion
    Reference from Bondi Accretion

Created on Sun Jan 31 20:18:01 2021

__author__      = nnarenraju
__copyright__   = Copyright 2021, bh-accretion
__credits__     = nnarenraju
__license__     = Apache License 2.0
__version__     = 1.0.0
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = inProgress


Github Repository: https://github.com/nnarenraju/bh-accretion.git

Documentation: NULL

"""

import numpy as np

# Plotting routines
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 16})

class BHAccretion:
    
    """
    
    
    """
    
    def __init__(self):
        # CONSTANTS
        # Polytropic EOS Constant (K)
        self.K = 1.0
        # Gamma value for entire run
        self.gamma = 1.0
        # Gravitational Constant
        self.G = 6.67259 * 10**-12 # cm^3 g^-1 s^-2
        # Boltzmann Constant
        self.k = 1.3807 * 10**-16 # cm^2 g s^-2 K^-1
        # Mass of the object
        self.M = 1.0 * 1.989e+33 # g
        
        # Number of iterations in RK4
        self.niter = 20000
        
        # Values at infinity
        # Assuming the temperature to be 10000 K
        self.a_inf = 10.0 * 1.0e5 # cm/s
        # Density of ISM 1 particle/cm^3
        self.rho_inf = 1.0e-24 # g/cm^3
        # Temperature 
        self.T_inf = 1.0e4 # K
        # Transonic radius setting
        self.rs = self._get_rs()
        # Lower limit of the simulation
        self.r_zero = 0.01 * self.rs
        # Upper limit of the simulation
        self.r_inf = 10.0 * self.rs
        # Pressure
        self.P_inf = self._get_Pinf()
        # Mass loss rate (Mdot)
        self.Mdot = self._get_Mdot()
        # Initial Velocity
        self.u_inf = self._get_Uinf()
        
        # Storage values
        self.RHO = []
        self.VEL = []
    
    def _get_Mdot(self):
        # Calculate the mass loss rate based on analytical solution
        term_1 = (0.5)**((self.gamma+1)/2.0*(self.gamma-1.0))
        term_2 = ((5.-3.*self.gamma)/4.0)**(-1*(5.-3.*self.gamma)/2.0*(self.gamma-1.0))
        lambda_s = term_1 * term_2
        lambda_s = 1.120
        print("Lambda_s value = {}".format(lambda_s))
        # Using lambda_s to calculate the mass loss rate from 14.3.4
        return 4.0*np.pi*lambda_s*self.rho_inf*self.a_inf*(self.G*self.M/self.a_inf**2)**2
    
    def _get_rs(self):
        # Calculate the transonic radius with given input params
        return (5.-3.*self.gamma)*(self.G*self.M/self.a_inf**2)
    
    def _get_Uinf(self):
        # Calculates the initial velocity given the mass loss rate
        # this calculation is done at infinity
        return self.Mdot/(4.0*np.pi* self.r_inf**2 * self.rho_inf)
    
    def _get_Pinf(self):
        # Calculate the pressure at infinity/ initial pressure in ISM
        return (self.rho_inf * self.a_inf**2)/self.gamma
        
    def outshot(self):
        # Calculating the outshot solution of Spherical Accretion
        # Set upper limit in __init__
        llimit = self.r_inf
        ulimit = self.rs
        # Setting initial values
        rho_init = self.rho_inf
        u_init = self.u_inf
        # Verbosity
        print("Running outshot with settings:")
        print("Initial radius = {}".format(llimit))
        print("Final radius = {}".format(ulimit))
        print("Initial density = {}".format(rho_init))
        print("Initial velocity = {}".format(u_init))
        # Calling RK4 with required initial values
        self.rk4(rho_init, u_init, llimit, ulimit)
        print("Outshot solution completed.")
    
    def inshot(self):
        # Calculating the inshot solution of Spherical Accretion
        # Set lower limit in __init__
        ulimit = 0.01*self.rs
        llimit = self.rs
        # Setting initial values
        rho_init = self.RHO[-1]
        u_init = self.VEL[-1]
        # Calling RK4 with required initial values
        self.rk4(rho_init, u_init, llimit, ulimit)
    
    def rk4(self, rho_init, u_init, llimit, ulimit):
        # 4th Order Runge-Kutta method
        print("\nRunning the 4th Order Runge-Kutta Numerical Solution")
        # Calculating the required step-size
        h = (ulimit - llimit)/self.niter
        # Required functions f and g
        P = lambda rho: self.K*rho**self.gamma
        a = lambda rho: np.sqrt(self.gamma*P(rho)/rho)
        f = lambda rho, u, r: u*(2.0*a(rho)**2/r - self.G*self.M/r**2)/(u**2-a(rho)**2)
        g = lambda rho, u, r: -1*rho*(2.0*u**2/r - self.G*self.M/r**2)/(u**2-a(rho)**2)
        # Setting initial values into variables
        rho = rho_init
        u = u_init
        # Storing results
        URHO = [rho]
        UVEL = [u]
        # Implementing RK4
        for r in np.arange(llimit, ulimit, h):
            k0 = h * f(rho, u, r)
            l0 = h * g(rho, u, r)
            k1 = h * f(rho+0.5*k0, u+0.5*l0, r+0.5*h)
            l1 = h * g(rho+0.5*k0, u+0.5*l0, r+0.5*h)
            k2 = h * f(rho+0.5*k1, u+0.5*l1, r+0.5*h)
            l2 = h * g(rho+0.5*k1, u+0.5*l1, r+0.5*h)
            k3 = h * f(rho+k2, u+l2, r+h)
            l3 = h * g(rho+k2, u+l2, r+h)
            rho = rho + (1./6.)*(k0+2.0*k1+2.0*k2+k3)
            u = u + (1./6.)*(l0+2.0*l1+2.0*l2+l3)
            # Updating storage variables
            URHO.append(rho)
            UVEL.append(u)
        
        # Storing required params
        self.RHO.extend(URHO)
        self.VEL.extend(UVEL)
        # Plotting
        X = np.arange(llimit, ulimit, h)[:self.niter]/self.rs
        Y = (np.array(UVEL)/a(np.array(URHO)))[:self.niter]
        print(Y)
        self._plot(X, Y)
    
    def _plot(self, X, Y):
        # Reproducing results
        print("Plotting the solution")
        plt.figure(figsize=(9.0, 9.0))
        plt.title('1D Hydrodynamical Accretion')
        plt.xlabel('r/rs')
        plt.ylabel('u/a')
        plt.plot(X, Y)
        plt.grid(True)
        plt.savefig('accretion.png')
        plt.close()


if __name__ == "__main__":
    
    bh = BHAccretion()
    bh.outshot()
    print("Done")



