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


from Compartment import Compartment
import numpy as np
from AxonDelay import AxonDelay
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

        self.pool = pool
        
        self.kind = ''
        
        
    
    def atualizeMuscleSpindle(self, t, fascicleLength, fascicleVelocity, fascicleAcceleration, gammaMNDynamicFR, gammaMNStaticFR):
        '''
        Atualize the dynamical and nondynamical (delay) parts of the motor unit.

        - Inputs:
            + **t**: current instant, in ms.
        '''
        self.atualizeCompartments(t)

    
        