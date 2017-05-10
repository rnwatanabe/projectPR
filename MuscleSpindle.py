'''
    Neuromuscular simulator in Python.
    Copyright (C) 2016  Renato Naville Watanabe

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Contact: renato.watanabe@usp.br
'''



import numpy as np
import math
from scipy.sparse import lil_matrix
import time




def runge_kutta(derivativeFunction, t, x, timeStep, timeStepByTwo,  timeStepBySix):
    '''
    Function to implement the fourth order Runge-Kutta Method to solve numerically a 
    differential equation.

    - Inputs: 
        + **derivativeFunction**: function that corresponds to the derivative of the differential equation.

        + **t**: current instant.

        + **x**:  current state value.

        + **timeStep**: time step of the solution of the differential equation, in the same unit of t.

        + **timeStepByTwo**:  timeStep divided by two, for computational efficiency.

        + **timeStepBySix**: timeStep divided by six, for computational efficiency.

    This method is intended to solve the following differential equation:

    \f{equation}{
        \frac{dx(t)}{dt} = f(t, x(t))
    \f}
    First, four derivatives are computed:

    \f{align}{
        k_1 &= f(t,x(t))\\
        k_2 &= f(t+\frac{\Delta t}{2}, x(t) + \frac{\Delta t}{2}.k_1)\\
        k_3 &= f(t+\frac{\Delta t}{2}, x(t) + \frac{\Delta t}{2}.k_2)\\
        k_4 &= f(t+\Delta t, x(t) + \Delta t.k_3)
    \f}
    where \f$\Delta t\f$ is the time step of the numerical solution of the
    differential equation.

    Then the value of \f$x(t+\Delta t)\f$ is computed with:

    \f{equation}{
        x(t+\Delta t) = x(t) + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3+k_4)
    \f}
    '''       
    k1 = derivativeFunction(t, x)
    k2 = derivativeFunction(t + timeStepByTwo, x + timeStepByTwo * k1)
    k3 = derivativeFunction(t + timeStepByTwo, x + timeStepByTwo * k2)
    k4 = derivativeFunction(t + timeStep, x + timeStep * k3)
    
    return x + timeStepBySix * (k1 + k2 + k2 + k3 + k3 + k4)



class MuscleSpindle(object):
    '''
    Class that implements a muscle spindle model. 
    '''

    def __init__(self, conf, muscle):
        '''
        Constructor

        - Inputs:
            + **conf**: Configuration object with the simulation parameters.

            + **muscle**: string with the muscle to which the motor
            unit belongs.              
        '''

        ## Configuration object with the simulation parameters.
        self.conf = conf

        self.muscle = muscle

        
        self.beta0Bag1 = float(conf.parameterSet('beta0Bag1', muscle, 0))
        self.beta0Bag2 = float(conf.parameterSet('beta0Bag2', muscle, 0))
        self.beta0Chain = float(conf.parameterSet('beta0Chain', muscle, 0))
        self.beta1Bag1 = float(conf.parameterSet('beta1Bag1', muscle, 0))
        self.beta2Bag2 = float(conf.parameterSet('beta2Bag2', muscle, 0))
        self.beta2Chain = float(conf.parameterSet('beta2Chain', muscle, 0))

        
        self.GAMMA1Bag1 = float(conf.parameterSet('GAMMA1Bag1', muscle, 0))
        self.GAMMA2Bag2 = float(conf.parameterSet('GAMMA2Bag2', muscle, 0))
        self.GAMMA2Chain = float(conf.parameterSet('GAMMA2Chain', muscle, 0))

        self.freq_bag1_Hz = float(conf.parameterSet('freqBag1', muscle, 0))
        self.freq_bag2_Hz = float(conf.parameterSet('freqBag2', muscle, 0))
        self.freq_Chain_Hz = float(conf.parameterSet('freqChain', muscle, 0))

        self.tauBag1_ms = float(conf.parameterSet('tauBag1', muscle, 0))
        self.tauBag2_ms = float(conf.parameterSet('tauBag2', muscle, 0))

        self.betaBag1 = 0
        self.betaBag2 = 0
        self.betaChain = 0

        self.GAMMABag1 = 0
        self.GAMMABag2 = 0
        self.GAMMAChain = 0

        ## Vector with the activation of each fusimotor fiber. The first
        # element is the frequency of Bag1, the second of Bag2 and the
        # third of the Chain.  
        self.fusimotorActivation = np.zeros((3), dtype=np.float64)

        ## Vector with the tensions and tensions derivatives of each 
        # fusimotor fiber. The first two elements correspond to the Bag1,
        #  the third and fourth to Bag2 and the last ones to the Chain. 
        self.fiberTension = np.zeros((6), dtype=np.float64)

        self.IaFR_Hz = 0.0
        self.IIFR_Hz = 0.0


    def atualizeMuscleSpindle(self, t, fascicleLength, fascicleVelocity, fascicleAcceleration, gammaMNDynamicFR, gammaMNStaticFR):
        '''
        Atualize the dynamical and nondynamical (delay) parts of the motor unit.

        - Inputs:
            + **t**: current instant, in ms.
        '''

        self.fusimotorActivation = self.computeFusimotorActivation(t, gammaMNDynamicFR, gammaMNStaticFR)
        
        self.betaBag1 = self.beta0Bag1 + self.beta1Bag1 * self.fusimotorActivation[0]
        self.betaBag2 = self.beta0Bag2 + self.beta2Bag2 * self.fusimotorActivation[1]
        self.betaChain = self.beta0Chain + self.beta2Chain * self.fusimotorActivation[2]
        
        self.GAMMABag1 = self.GAMMA1Bag1 * self.fusimotorActivation[0]
        self.GAMMABag2 = self.GAMMA2Bag2 * self.fusimotorActivation[1]
        self.GAMMAChain = self.GAMMA2Chain * self.fusimotorActivation[2]
        
            

        self.computeFiberTension(t, fascicleLength, fascicleVelocity, fascicleAcceleration)


        self.IaFR = self.computeIa(t)
        self.IIFR = self.computeII(t)

    def computeFusimotorActivation(self, t, gammaMNDynamicFR, gammaMNStaticFR):
        '''

        '''
        df = self.dfdt(t,gammaMNDynamicFR, gammaMNStaticFR)
        self.fusimotorActivation[0] += self.conf.timeStep * df[0] 
        self.fusimotorActivation[1] += self.conf.timeStep * df[1]
        self.fusimotorActivation[2] = gammaMNStaticFR**2/(gammaMNStaticFR**2 + self.freq_Chain_Hz**2)


    def dfdt(self, t, gammaMNDynamicFR, gammaMNStaticFR):
        '''
        '''
        df = np.zeros((2))

        df[0] = (gammaMNDynamicFR**2/(gammaMNDynamicFR**2 + self.freq_bag1_Hz**2) - self.fusimotorActivation[0]) / self.tauBag1_ms
        df[1] = (gammaMNStaticFR**2/(gammaMNStaticFR**2 + self.freq_bag2_Hz**2) - self.fusimotorActivation[1]) / self.tauBag2_ms

        return df

    def computeFiberTension(self, t, fascicleLength, fascicleVelocity, fascicleAcceleration):
        '''
        '''
        dT = self.dTdt(t, fascicleLength, fascicleVelocity, fascicleAcceleration)

    def dTdt(self, fascicleLength, fascicleVelocity, fascicleAcceleration):
        '''

        '''
        dT = np.zeros((6))
        
        dT[0] = self.fiberTension[1]
        dT[1] = self.KsrBag1 / MBag1 * ()
        dT[2] = self.fiberTension[3]
        dT[3] = 
        dT[4] = self.fiberTension[5]
        dT[5] = 
        return dT

    def computeIa(self, t):
        '''

        '''

    def computeII(self, t):
        '''

        '''








    
        