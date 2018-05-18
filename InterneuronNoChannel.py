'''
    Neuromuscular simulator in Python.
    Copyright (C) 2018  Renato Naville Watanabe

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

    Contact: renato.watanabe@ufabc.edu.br
'''


from CompartmentNoChannel import CompartmentNoChannel
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



class InterneuronNoChannel(object):
    '''
    Class that implements a motor unit model. Encompasses a motoneuron
    and a muscle unit.
    '''

    def __init__(self, conf, pool, index):
        '''
        Constructor

        - Inputs:
            + **conf**: Configuration object with the simulation parameters.

            + **pool**: string with Interneuron pool to which the motor
            unit belongs.  It can
            be *RC* (Renshaw cell), *IaIn* (Ia Interneuron), *IbIn* (Ib Interneuron) and 
            *gII*.

            + **index**: integer corresponding to the motor unit order in
            the pool, according to the Henneman's principle (size principle).
        '''

        ## Configuration object with the simulation parameters.
        self.conf = conf

        self.pool = pool
        
        self.kind = ''
        # Neural compartments
        ## The instant of the last spike of the Motor unit
        ## at the Soma compartment.
        self.tSomaSpike = float("-inf")
        compartmentsList = ['soma']
        ## Vector with the instants of spikes at the soma.
        self.somaSpikeTrain = []
        ## Integer corresponding to the Interneuron order in the pool.
        self.index = int(index)
        ## Vector of Compartment of the Motor Unit.
        self.compartment = dict()
        ## Value of the membrane potential, in mV, that is considered a spike.
        self.threshold_mV = conf.parameterSet('threshold', pool, index)

        ## Anatomical position of the neuron, in mm.
        self.position_mm = conf.parameterSet('position', pool, index)
        

        for i in xrange(len(compartmentsList)): 
            self.compartment[i] = CompartmentNoChannel(compartmentsList[i], self.conf, self.pool, self.index, self.kind)

        ## Number of compartments.
        self.compNumber = len(self.compartment)
        ## Vector with membrane potential,in mV, of all compartments.
        self.v_mV = np.zeros((self.compNumber), dtype = np.float64)

        gLeak = np.zeros_like(self.v_mV, dtype = 'd')

        capacitance_nF = np.zeros_like(self.v_mV, dtype = 'd')
        EqPot = np.zeros_like(self.v_mV, dtype = 'd') 

        for i in xrange(len(self.compartment)):
            capacitance_nF[i] = self.compartment[i].capacitance_nF
            gLeak[i] = self.compartment[i].gLeak_muS
            EqPot[i] = self.compartment[i].EqPot_mV


        ## Vector with  the inverse of the capacitance of all compartments.
        self.capacitanceInv = 1 / capacitance_nF

        ## Vector with current, in nA,  of each compartment coming from other elements of the model. For example
        ## from ionic channels and synapses.
        self.iIonic = np.full_like(self.v_mV, 0.0)
        ## Vector with the current, in nA, injected in each compartment.
        self.iInjected = np.zeros_like(self.v_mV, dtype = 'd')
        #self.iInjected = np.array([0, 10.0])


        GL = -np.diag(gLeak)

        ## Matrix of the conductance of the motoneuron. Multiplied by the vector self.v_mV,
        ## results in the passive currents of each compartment.
        self.G = np.float64(GL)

        self.EqCurrent_nA = np.dot(-GL, EqPot)

        ## index of the soma compartment.
        self.somaIndex = compartmentsList.index('soma')
        
        ## Refractory period, in ms, of the motoneuron.
        self.RefPer_ms = float(conf.parameterSet(self.pool + 'SomaRefPer', pool, index))
       
        ## Vector with the instants of spikes at the terminal.
        self.terminalSpikeTrain = []
                
        
        ## Build synapses       
         
        self.SynapsesOut = []
        self.transmitSpikesThroughSynapses = []
        self.indicesOfSynapsesOnTarget = []
        
    
    def atualizeInterneuron(self, t, v_mV):
        '''
        Atualize the dynamical and nondynamical (delay) parts of the motor unit.

        - Inputs:
            + **t**: current instant, in ms.
        '''
        self.atualizeCompartments(t, v_mV)

    def atualizeCompartments(self, t, v_mV):
        '''
        Atualize all neural compartments.

        - Inputs:
            + **t**: current instant, in ms.
        '''
        self.v_mV[:] = v_mV

        if self.v_mV[self.somaIndex] > self.threshold_mV and t-self.tSomaSpike > self.RefPer_ms:
            self.addSomaSpike(t)
            self.v_mV[self.somaIndex] = -10


    def addSomaSpike(self, t):
        '''
        When the soma potential is above the threshold a spike is added to the soma.

        - Inputs:
            + **t**: current instant, in ms.
        '''
        self.tSomaSpike = t
        self.somaSpikeTrain.append([t, int(self.index)])
        self.transmitSpikes(t)

       
    def transmitSpikes(self, t):
        '''
        - Inputs:
            + **t**: current instant, in ms.
        '''
        
        for i in xrange(len(self.indicesOfSynapsesOnTarget)):
            self.transmitSpikesThroughSynapses[i].receiveSpike(t, self.indicesOfSynapsesOnTarget[i])

        
