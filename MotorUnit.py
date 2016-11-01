'''
Created on Oct 6, 2015

@author: root
'''




from Compartment import Compartment
import numpy as np
from AxonDelay import AxonDelay
import math
from scipy.sparse import lil_matrix
import time


def calcGCoupling(cytR, lComp1, lComp2, dComp1, dComp2):
    '''
    Calculates the coupling conductance between two compartments.


    - Inputs: 
         + **cytR**: Cytoplasmatic resistivity in \f$\Omega\f$.cm.

         + **lComp1, lComp2**: length of the compartments in \f$\mu\f$m.

         + **dComp1, dComp2**: diameter of the compartments in \f$\mu\f$m.

    - Output:
         + coupling conductance in MS.

    The coupling conductance between compartment 1 and 2 is
    computed by the following equation:

    \f{equation}{
        g_c = \frac{2.10^2}{\frac{R_{cyt}l_1}{\pi r_1^2}+\frac{R_{cyt}l_2}{\pi r_2^2}}
    \f}
    where \f$g_c\f$ is the coupling conductance [MS], \f$R_{cyt}\f$ is the
    cytoplasmatic resistivity [\f$\Omega\f$.cm], \f$l_1\f$ and \f$l_2\f$
    are the lengths [\f$\mu\f$m] of compartments 1 and 2, respectively and
    \f$r_1\f$ and \f$r_2\f$ are the radius [\f$\mu\f$m] of compartments 1 and
    2, respectively.
    '''
    rAxis1 = (cytR * lComp1) / (math.pi * math.pow(dComp1/2, 2))
    rAxis2 = (cytR * lComp2) / (math.pi * math.pow(dComp2/2, 2))
    
    return (1e2 * 2) / (rAxis1 + rAxis2)




def compGCouplingMatrix(gc):
    '''
    Computes the Coupling Matrix to be used in the dVdt function of the N compartments of the motor unit. 
    The Matrix uses the values obtained with the function calcGcoupling.
 
    - Inputs: 
        + **gc**: the vector with N elements, with the coupling conductance of each compartment of the Motor Unit.

    - Output:
        + the GC matrix


    \f{equation}{
       GC = \left[\begin{array}{cccccccc}
       -g_c[0]&g_c[0]&0&...&...&0&0&0\\
       g_c[0]&-g_c[0]-g_c[1]&g_c[1]&0&...&...&0&0\\
       \vdots&&\ddots&&...&&0&0 \\
       0&...&g_c[i-1]&-g_c[i-1]-g_c[i]&g_c[i]&0&...&0\\
       0&0&0&...&...&&&0\\
       0&&...&&g_c[N-2]&-g_c[N-2]-g_c[N-1]&g_c[N-1]&0\\
       0&...&0&&&0&g_c[N-1]&-g_c[N-1]\end{array}\right] 
    \f} 
    '''
    
    GC = np.zeros((len(gc),len(gc)))
    
    for i in xrange(0, len(gc)):
        if i == 0:
            GC[i,i:i+2] = [-gc[i], gc[i]] 
        elif i == len(gc) - 1:
            GC[i,i-1:i+1] = [gc[i-1], -gc[i-1]]  
        else:
            GC[i,i-1:i+2] = [gc[i-1], -gc[i-1]-gc[i], gc[i]]
                  
            
    return GC

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



class MotorUnit(object):
    '''
    Class that implements a motor unit model. Encompasses a motoneuron
    and a muscle unit.
    '''

    def __init__(self, conf, pool, index, kind):
        '''
        Constructor

        - Inputs:
            + **conf**: Configuration object with the simulation parameters.

            + **pool**: string with Motor unit pool to which the motor
            unit belongs.

            + **index**: integer corresponding to the motor unit order in
            the pool, according to the Henneman's principle (size principle).

            + **kind**: string with the type of the motor unit. It can
            be *S* (slow), *FR* (fast and resistant), and
            *FF* (fast and fatigable).
        '''

        ## Configuration object with the simulation parameters.
        self.conf = conf

        ## String with the type of the motor unit. It can be
        ## *S* (slow), *FR* (fast and resistant) and
        ## *FF** (fast and fatigable).
        self.kind = kind
        
        # Neural compartments
        ## The instant of the last spike of the Motor unit
        ## at the Soma compartment.
        self.tSomaSpike = float("-inf")
        compartmentsList = ['dendrite', 'soma']
        ## Vector with the instants of spikes at the soma.
        self.somaSpikeTrain = []
        ## Integer corresponding to the motor unit order in the pool, according to the Henneman's principle (size principle).
        self.index = int(index)
        ## Vector of Compartment of the Motor Unit.
        self.compartment = []
        ## Value of the membrane potential, in mV, that is considered a spike.
        self.threshold_mV = conf.parameterSet('threshold', pool, index)
                
        ## Anatomical position of the neuron, in mm.
        self.position_mm = conf.parameterSet('position', pool, index)
        
        for i in compartmentsList: self.compartment.append(Compartment(i, conf, pool, index, self.kind))        
        
        ## Number of compartments.
        self.compNumber = len(self.compartment)
        ## Vector with membrane potential,in mV, of all compartments. 
        self.v_mV = np.zeros((self.compNumber), dtype = np.float64)
        
        
        gCoupling_MS = np.zeros_like(self.v_mV, dtype = 'd')
        gLeak = np.zeros_like(self.v_mV, dtype = 'd')        
        for i in self.compartment[0:-1]: gCoupling_MS[self.compartment.index(i)] = calcGCoupling(float(conf.parameterSet('cytR',pool, index)), 
                          self.compartment[self.compartment.index(i)].length_mum,
                          self.compartment[self.compartment.index(i) + 1].length_mum,
                          self.compartment[self.compartment.index(i)].diameter_mum,
                          self.compartment[self.compartment.index(i) + 1].diameter_mum)
        
        
        capacitance_nF = np.zeros_like(self.v_mV, dtype = 'd')  
        
        for i in self.compartment:                                                              
            capacitance_nF[self.compartment.index(i)] = i.capacitance_nF
            gLeak[self.compartment.index(i)] = i.gLeak
            

        ## Vector with  the inverse of the capacitance of all compartments. 
        self.capacitanceInv = 1 / capacitance_nF

        ## Vector with current, in nA,  of each compartment coming from other elements of the model. For example 
        ## from ionic channels and synapses.       
        self.iIonic = np.full_like(self.v_mV, 0.0)
        ## Vector with the current, in nA, injected in each compartment.
        self.iInjected = np.zeros_like(self.v_mV, dtype = 'd')
        #self.iInjected = np.array([0, 10.0])
        
        GC = compGCouplingMatrix(gCoupling_MS)
        
        GL = -np.diag(gLeak)
        
        ## Matrix of the conductance of the motoneuron. Multiplied by the vector self.v_mV,
        ## results in the passive currents of each compartment.
        self.G = np.float64(GC + GL)


        ## index of the soma compartment.
        self.somaIndex = compartmentsList.index('soma')
        
        ## Refractory period, in ms, of the motoneuron.
        self.MNRefPer_ms = float(conf.parameterSet('MNSomaRefPer', pool, index))
        
        # delay
        ## String with type of the nerve. It can be PTN (posterior tibial nerve) or CPN
        ## (common peroneal nerve).
        if (pool == 'SOL' or pool == 'MG' or pool == 'LG'):
            self.nerve = 'PTN'
        else:
            self.nerve = 'CPN'
            
        ## AxonDelay object of the motor unit.
        self.Delay = AxonDelay(conf, self.nerve, pool, index)


        ## Vector with the instants of spikes at the terminal.
        self.terminalSpikeTrain = []
                
        
        # contraction DataMUnumber_S = int(conf.parameterSet('MUnumber_S_' + pool, pool, 0))
        activationModel = conf.parameterSet('activationModel', pool, 0)
        
        ## Contraction time of the twitch muscle unit, in ms.
        self.TwitchTc_ms = conf.parameterSet('twitchTimePeak', pool, index)
        ## Amplitude of the muscle unit twitch, in N.
        self.TwitchAmp_N = conf.parameterSet('twitchPeak', pool, index)
        ## Parameter of the saturation.
        self.bSat = conf.parameterSet('bSat'+ activationModel,pool,index)
        ## Twitch- tetanus relationship
        self.twTet = conf.parameterSet('twTet' + activationModel,pool,index)
        
        ## EMG data
        
        ## Build synapses       
         
        self.SynapsesOut = []
        self.transmitSpikesThroughSynapses = []
        self.indicesOfSynapsesOnTarget = []
    
    def atualizeMotorUnit(self, t):
        '''
        Atualize the dynamical and nondynamical (delay) parts of the motor unit.

        - Inputs:
            + **t**: current instant, in ms.
        ''' 
        self.atualizeCompartments(t)
        self.atualizeDelay(t)
        
    def atualizeCompartments(self, t):
        '''
        Atualize all neural compartments.

        - Inputs:
            + **t**: current instant, in ms.

        '''
        
        np.clip(runge_kutta(self.dVdt, t, self.v_mV, self.conf.timeStep_ms, self.conf.timeStepByTwo_ms, self.conf.timeStepBySix_ms), -16.0, 120.0, self.v_mV)
        if (self.v_mV[self.somaIndex] > self.threshold_mV and t-self.tSomaSpike > self.MNRefPer_ms): self.addSomaSpike(t)    
     
       
    def dVdt(self, t, V): 
        '''
        Compute the potential derivative of all compartments of the motor unit.

        - Inputs:
            + **t**: current instant, in ms.

            + **V**: Vector with the current potential value of all neural
            compartments of the motor unit.
        
        \f{equation}{
            \frac{dV}{dt} = (I_{active} + GV+ I_{inj})C_inv   
        }
        where all the variables are vectors with the number of elements equal
        to the number of compartments and \f$G\f$ is the conductance matrix built
        in the compGCouplingMatrix function.
        '''
        for compartment in xrange(0, self.compNumber):  
            self.iIonic.itemset(compartment, self.compartment[compartment].computeCurrent(t, V.item(compartment)))
              
        return (self.iIonic + np.dot(self.G, V)  + self.iInjected) * self.capacitanceInv
    
    
    def addSomaSpike(self, t):
        '''
        When the soma potential is above the threshold a spike is added tom the soma.

        - Inputs:
            + **t**: current instant, in ms.
        '''
        self.tSomaSpike = t
        self.somaSpikeTrain.append([t, int(self.index)])
        self.Delay.addSpinalSpike(t)
        self.transmitSpikes(t)
        
        for channel in self.compartment[self.somaIndex].Channels:
            for channelState in channel.condState: channelState.changeState(t)    
              
              
    def atualizeDelay(self, t):
        '''
        Atualize the terminal spike train, by considering the Delay of the nerve.

        - Inputs:
            + **t**: current instant, in ms.
        '''
        if abs(t - self.Delay.terminalSpikeTrain) < 1e-3: 
            self.terminalSpikeTrain.append([t, self.index])

    def transmitSpikes(self, t):
        '''

        - Inputs:
            + **t**: current instant, in ms.
        '''
        for i in xrange(len(self.indicesOfSynapsesOnTarget)):
            self.transmitSpikesThroughSynapses[i].receiveSpike(t, self.indicesOfSynapsesOnTarget[i])
        
    
        