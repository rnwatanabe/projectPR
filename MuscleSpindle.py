'''
    Neuromuscular simulator in Python.
    Copyright (C) 2017  Renato Naville Watanabe

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

            + **muscle**: string with the muscle to which the muscle spindle belongs.              
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

        self.tauBag1_s = float(conf.parameterSet('tauBag1', muscle, 0))
        self.tauBag2_s = float(conf.parameterSet('tauBag2', muscle, 0))

        self.KsrBag1 = float(conf.parameterSet('KsrBag1', muscle, 0))
        self.KsrBag2 = float(conf.parameterSet('KsrBag2', muscle, 0))
        self.KsrChain = float(conf.parameterSet('KsrChain', muscle, 0))

        self.MBag1 = float(conf.parameterSet('MBag1', muscle, 0))
        self.MBag2 = float(conf.parameterSet('MBag2', muscle, 0))
        self.MChain = float(conf.parameterSet('MChain', muscle, 0))

        self.L0SrBag1 = float(conf.parameterSet('L0SrBag1', muscle, 0))
        self.L0SrBag2 = float(conf.parameterSet('L0SrBag2', muscle, 0))
        self.L0SrChain = float(conf.parameterSet('L0SrChain', muscle, 0))

        self.L0PrBag1 = float(conf.parameterSet('L0PrBag1', muscle, 0))
        self.L0PrBag2 = float(conf.parameterSet('L0PrBag2', muscle, 0))
        self.L0PrChain = float(conf.parameterSet('L0PrChain', muscle, 0))

        self.LNSrBag1 = float(conf.parameterSet('LNSrBag1', muscle, 0))
        self.LNSrBag2 = float(conf.parameterSet('LNSrBag2', muscle, 0))
        self.LNSrChain = float(conf.parameterSet('LNSrChain', muscle, 0))

        
        self.LNPrBag2 = float(conf.parameterSet('LNPrBag2', muscle, 0))
        self.LNPrChain = float(conf.parameterSet('LNPrChain', muscle, 0))

        self.RBag1 = float(conf.parameterSet('RBag1', muscle, 0))
        self.RBag2 = float(conf.parameterSet('RBag2', muscle, 0))
        self.RChain = float(conf.parameterSet('RChain', muscle, 0))

        self.KPrBag1 = float(conf.parameterSet('KPrBag1', muscle, 0))
        self.KPrBag2 = float(conf.parameterSet('KPrBag2', muscle, 0))
        self.KPrChain = float(conf.parameterSet('KPrChain', muscle, 0))

        self.GPrimaryBag1 = float(conf.parameterSet('GPrimaryBag1', muscle, 0))
        self.GPrimaryBag2 = float(conf.parameterSet('GPrimaryBag2', muscle, 0))
        self.GPrimaryChain = float(conf.parameterSet('GPrimaryChain', muscle, 0))

        self.SOcclusionFactor = float(conf.parameterSet('SOcclusionFactor', muscle, 0))

        self.betaBag1 = 0
        self.betaBag2 = 0
        self.betaChain = 0

        self.GAMMABag1 = 0
        self.GAMMABag2 = 0
        self.GAMMAChain = 0

        self.primaryPotentialBag1 = 0
        self.primaryPotentialBag2 = 0
        self.primaryPotentialChain = 0

        self.secondaryPotentialBag1 = 0
        self.secondaryPotentialBag2 = 0
        self.secondaryPotentialChain = 0

        

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

        print 'Muscle spindle from muscle ' + self.muscle + ' built.'


    def atualizeMuscleSpindle(self, t, fascicleLength, fascicleVelocity, 
                              fascicleAcceleration, gammaMNDynamicFR, gammaMNStaticFR):
        '''
        Atualize the dynamical and nondynamical (delay) parts of the motor unit.

        - Inputs:
            + **t**: current instant, in ms.
        '''

        self.computeFusimotorActivation(t, gammaMNDynamicFR, gammaMNStaticFR)
        
        
        self.betaBag1 = self.beta0Bag1 + self.beta1Bag1 * self.fusimotorActivation[0]
        self.betaBag2 = self.beta0Bag2 + self.beta2Bag2 * self.fusimotorActivation[1]
        self.betaChain = self.beta0Chain + self.beta2Chain * self.fusimotorActivation[2]
        
        self.GAMMABag1 = self.GAMMA1Bag1 * self.fusimotorActivation[0]
        self.GAMMABag2 = self.GAMMA2Bag2 * self.fusimotorActivation[1]
        self.GAMMAChain = self.GAMMA2Chain * self.fusimotorActivation[2]
        
        

        self.computeFiberTension(t, fascicleLength, fascicleVelocity, fascicleAcceleration)


        self.IaFR_Hz = self.computeIa(t)
        #self.IIFR = self.computeII(t)

    def computeC(self, fascicleVelocity, dT, Ksr):
        if (fascicleVelocity - dT/Ksr)  > 0:
            C = 1.0
        else:
            C = 0.42
        return C

    def computeFusimotorActivation(self, t, gammaMNDynamicFR, gammaMNStaticFR):
        '''

        '''
        df = self.dfdt(t,gammaMNDynamicFR, gammaMNStaticFR)
        self.fusimotorActivation[0] += self.conf.timeStep_ms / 1000.0 * df[0] 
        self.fusimotorActivation[1] += self.conf.timeStep_ms / 1000.0 * df[1]
        self.fusimotorActivation[2] = gammaMNStaticFR**2/(gammaMNStaticFR**2 + self.freq_Chain_Hz**2)


    def dfdt(self, t, gammaMNDynamicFR, gammaMNStaticFR):
        '''
        '''
        df = np.zeros((2))

        df[0] = ((gammaMNDynamicFR**2)/(gammaMNDynamicFR**2 + self.freq_bag1_Hz**2) - self.fusimotorActivation[0]) / self.tauBag1_s
        df[1] = ((gammaMNStaticFR**2)/(gammaMNStaticFR**2 + self.freq_bag2_Hz**2) - self.fusimotorActivation[1]) / self.tauBag2_s

        return df

    def computeFiberTension(self, t, fascicleLength, fascicleVelocity, fascicleAcceleration):
        '''
        '''
        dT = self.dTdt(t, fascicleLength, fascicleVelocity, fascicleAcceleration)
        self.fiberTension += self.conf.timeStep_ms / 1000 * dT
        

    def dTdt(self, t, fascicleLength, fascicleVelocity, fascicleAcceleration):
        '''

        '''
        dT = np.zeros((6))
        
        dT[0] = self.fiberTension[1]
        dT[1] = self.KsrBag1 / self.MBag1 * (self.computeC(fascicleVelocity, self.fiberTension[1], self.KsrBag1) * self.betaBag1 * np.sign(fascicleVelocity - self.fiberTension[1]/self.KsrBag1)
                                             * (math.fabs(fascicleVelocity - self.fiberTension[1]/self.KsrBag1)**0.3) * 
                                             (fascicleLength - self.L0SrBag1 - self.fiberTension[0]/self.KsrBag1 - self.RBag1)
                                             + self.KPrBag1 * (fascicleLength - self.L0SrBag1 - self.fiberTension[0]/self.KsrBag1 - self.L0PrBag1)
                                             + self.MBag1 * fascicleAcceleration + self.GAMMABag1 - self.fiberTension[0]
                                            )
        dT[2] = self.fiberTension[3]
        dT[3] = self.KsrBag2 / self.MBag2 * (self.computeC(fascicleVelocity, self.fiberTension[3], self.KsrBag2) * self.betaBag2 * np.sign(fascicleVelocity - self.fiberTension[3]/self.KsrBag2)
                                        * (math.fabs(fascicleVelocity - self.fiberTension[3]/self.KsrBag2)**0.3) * 
                                        (fascicleLength - self.L0SrBag2 - self.fiberTension[2]/self.KsrBag2 - self.RBag2)
                                        + self.KPrBag2 * (fascicleLength - self.L0SrBag2 - self.fiberTension[2]/self.KsrBag2 - self.L0PrBag2)
                                        + self.MBag2 * fascicleAcceleration + self.GAMMABag2 - self.fiberTension[2]
                                       )
        dT[4] = self.fiberTension[5]
        dT[5] = self.KsrChain / self.MChain * (self.computeC(fascicleVelocity, self.fiberTension[5], self.KsrChain) * self.betaChain * np.sign(fascicleVelocity - self.fiberTension[5]/self.KsrChain)
                                        * (math.fabs(fascicleVelocity - self.fiberTension[5]/self.KsrChain)**0.3) * 
                                        (fascicleLength - self.L0SrChain - self.fiberTension[4]/self.KsrChain - self.RChain)
                                        + self.KPrChain * (fascicleLength - self.L0SrChain - self.fiberTension[4]/self.KsrChain - self.L0PrChain)
                                        + self.MChain * fascicleAcceleration + self.GAMMAChain - self.fiberTension[4]
                                       )
        return dT

    def computeIa(self, t):
        '''

        '''
        self.computePrimaryActivity(t)

        if (self.primaryPotentialBag1 >= (self.primaryPotentialBag2 + self.primaryPotentialChain)):
            larger = self.primaryPotentialBag1
            smaller = self.primaryPotentialBag2 + self.primaryPotentialChain
        else:
            smaller = self.primaryPotentialBag1
            larger = self.primaryPotentialBag2 + self.primaryPotentialChain

        return self.SOcclusionFactor * smaller + larger

    def computeII(self, t):
        '''

        '''

    def computePrimaryActivity(self, t):
        '''

        '''
        self.primaryPotentialBag1 = self.GPrimaryBag1 * (self.fiberTension[0] / self.KsrBag1 - 
                                                         self.LNSrBag1 + self.L0SrBag1
                                                        )

        self.primaryPotentialBag2 = self.GPrimaryBag2 * (self.fiberTension[2] / self.KsrBag2 - 
                                                         self.LNSrBag2 + self.L0SrBag2
                                                        )
                                                    
        self.primaryPotentialChain = self.GPrimaryChain * (self.fiberTension[4] / self.KsrChain - 
                                                           self.LNSrChain + self.L0SrChain
                                                          )

    def reset(self, t):
        '''
        '''
        self.betaBag1 = 0
        self.betaBag2 = 0
        self.betaChain = 0

        self.GAMMABag1 = 0
        self.GAMMABag2 = 0
        self.GAMMAChain = 0

        self.primaryPotentialBag1 = 0
        self.primaryPotentialBag2 = 0
        self.primaryPotentialChain = 0

        self.secondaryPotentialBag1 = 0
        self.secondaryPotentialBag2 = 0
        self.secondaryPotentialChain = 0
      
        self.fusimotorActivation = np.zeros((3), dtype=np.float64)

        self.fiberTension = np.zeros((6), dtype=np.float64)

        self.IaFR_Hz = 0.0
        self.IIFR_Hz = 0.0








    
        