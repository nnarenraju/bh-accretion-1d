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
        # Number of iterations in RK4
        self.niter = 20000
        # Transonic radius setting
        self.rs = 1.0
        
        # Lower limit of the simulation
        self.surface = 0.01 * self.rs
        # Upper limit of the simulation
        self.infty = 10.0 * self.rs
        
        # Values at infinity
        # Assuming the temperature to be 10 K
        self.a_inf = 0.3 * 1.0e5 # cm/s
        # Density of ISM 1 particle/cm^3
        self.rho_inf = 1.0e-24 # g/cm^3
        
        # CONSTANTS
        # Polytropic EOS Constant (K)
        self.K = 1.0
        # Gamma value for entire run
        self.gamma = 5./3.
        # Gravitational Constant
        self.G = 6.67259 * 10**-12 # cm^3 g^-1 s^-2
        # Boltzmann Constant
        self.k = 1.3807 * 10**-16 # cm^2 g s^-2 K^-1
        # Mass of the object
        self.M = 5 * 1.989e+33 # g
        
        # Storage values
        self.RHO = []
        self.VEL = []
        
    def outshot(self):
        # Calculating the outshot solution of Spherical Accretion
        # Set upper limit in __init__
        ulimit = self.rs
        llimit = self.infty
        # Setting initial values
        rho_init = self.rho_inf
        u_init = self.a_inf**2
        
        rho_init = 10.0
        u_init = 5.0
        
        # Calling RK4 with required initial values
        self.rk4(rho_init, u_init, llimit, ulimit)
    
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
        X = np.arange(llimit, ulimit, h)[:self.niter]
        Y = (np.array(UVEL)/a(np.array(URHO)))[:self.niter]
        self._plot(X, Y)
    
    def _plot(self, X, Y):
        # Reproducing results
        plt.figure(figsize=(9.0, 9.0))
        plt.title('1D Hydrodynamical Accretion')
        plt.xlabel('r/rs')
        plt.ylabel('u/a')
        plt.plot(X, Y)
        plt.grid(True)
        plt.savefig('accretion.png')
        plt.close()
        
    
    
    
    
    
    