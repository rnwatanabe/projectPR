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


def calcGCoupling(cytR, lComp1, lComp2, dComp1, dComp2):
    '''
    Calculates the coupling conductance between two compartments.

    - Inputs: 
         + **cytR**: Cytoplasmatic resistivity in \f$\Omega\f$.cm.

         + **lComp1, lComp2**: length of the compartments in \f$\mu\f$m.

         + **dComp1, dComp2**: diameter of the compartments in \f$\mu\f$m.

    - Output:
         + coupling conductance in \f$\mu\f$S.

    The coupling conductance between compartment 1 and 2 is
    computed by the following equation:

    \f{equation}{
        g_c = \frac{2.10^2}{\frac{R_{cyt}l_1}{\pi r_1^2}+\frac{R_{cyt}l_2}{\pi r_2^2}}
    \f}
    where \f$g_c\f$ is the coupling conductance [\f$\mu\f$S], \f$R_{cyt}\f$ is the
    cytoplasmatic resistivity [\f$\Omega\f$.cm], \f$l_1\f$ and \f$l_2\f$
    are the lengths [\f$\mu\f$m] of compartments 1 and 2, respectively and
    \f$r_1\f$ and \f$r_2\f$ are the radius [\f$\mu\f$m] of compartments 1 and
    2, respectively.
    '''
    rAxis1 = (cytR * lComp1) / (math.pi * math.pow(dComp1/2.0, 2))
    rAxis2 = (cytR * lComp2) / (math.pi * math.pow(dComp2/2.0, 2))
    
    return 200 / (rAxis1 + rAxis2)




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

#@profile
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
    #return x + timeStep * (k1)


class MotorUnit(object):
    '''
    Class that implements a motor unit model. Encompasses a motoneuron
    and a muscle unit.
    '''

    def __init__(self, conf, pool, index, kind, muscleThickness, skinThickness):
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


        self.pool = pool
        
        # Neural compartments
        ## The instant of the last spike of the Motor unit
        ## at the Soma compartment.
        self.tSomaSpike = float("-inf")

        NumberOfAxonNodes = int(conf.parameterSet('NumberAxonNodes', pool, index))
        

        compartmentsList = ['dendrite', 'soma']
        for i in xrange(0, NumberOfAxonNodes):
              compartmentsList.append('internode')
              compartmentsList.append('node')
              


        
        ## Integer corresponding to the motor unit order in the pool, according to the Henneman's principle (size principle).
        self.index = int(index)
        ## Dictionary of Compartment of the Motor Unit.
        self.compartment = dict()
        ## Value of the membrane potential, in mV, that is considered a spike.
        self.threshold_mV = conf.parameterSet('threshold', pool, index)

        

        ## Anatomical position of the neuron, in mm.
        self.position_mm = conf.parameterSet('position', pool, index)
        
        # EMG data
        self.MUSpatialDistribution = conf.parameterSet('MUSpatialDistribution',pool, 0)       
        if self.MUSpatialDistribution == 'random':
            radius = (muscleThickness/2) * np.random.uniform(0.0, 1.0)
            angle = 2.0 * math.pi * np.random.uniform(0.0, 1.0)
		
        x = radius * math.sin(angle)
        y = radius * math.cos(angle)
        ## Anatomical coordinate of the muscle unit in a muscle section, in (mm,mm).
        self.muSectionPosition_mm = [x,y]

        ## Distance of the MU to the EMG elctrode, in mm.
        self.distance_mm = math.sqrt((x + muscleThickness/2.0 +
                                      skinThickness)**2 + y**2)

        ## Attenuation of the MUAP amplitude, as measured in the electrode.
        self.attenuationToSkin = math.exp(-self.distance_mm / conf.EMGAttenuation_mm1)

        ## Widening of the MUAP duration, as measured in the electrode.
        self.timeWidening = 1 + conf.EMGWidening_mm1 * self.distance_mm
        
        ## Type of the Hermitez-Rodiguez curve. It can be 1 or 2.
        self.hrType = np.random.random_integers(1,2)
        
        ## MUAP amplitude in mV.
        self.ampEMG_mV = conf.parameterSet('EMGAmplitude', pool, index)
        self.ampEMG_mV = self.ampEMG_mV * self.attenuationToSkin

        ## MUAP time constant, in ms.
        self.timeCteEMG_ms = conf.parameterSet('EMGDuration', pool, index)
        self.timeCteEMG_ms = self.timeCteEMG_ms * self.timeWidening
        
        

        for i in xrange(len(compartmentsList)):
            self.compartment[i] = Compartment(compartmentsList[i], conf, pool, index, self.kind)

        ## Number of compartments.
        self.compNumber = len(self.compartment)
        ## Vector with membrane potential,in mV, of all compartments. 
        self.v_mV = np.zeros((self.compNumber), dtype = np.float64)
        ## Vector with the last instant of spike of all compartments. 
        self.tSpikes = np.zeros((self.compNumber), dtype = np.float64)
        
        gCoupling_muS = np.zeros_like(self.v_mV, dtype = 'd')
        
            
        for i in xrange(len(self.compartment)-1): 
            gCoupling_muS[i] = calcGCoupling(float(conf.parameterSet('cytR',pool, index)), 
                                             self.compartment[i].length_mum,
                                             self.compartment[i + 1].length_mum,
                                             self.compartment[i].diameter_mum,
                                             self.compartment[i + 1].diameter_mum)
        
        
        gLeak = np.zeros_like(self.v_mV, dtype = 'd')    
        capacitance_nF = np.zeros_like(self.v_mV, dtype = 'd')
        EqPot = np.zeros_like(self.v_mV, dtype = 'd')
        IPump = np.zeros_like(self.v_mV, dtype = 'd')
        compLength = np.zeros_like(self.v_mV, dtype = 'd')        
        
        for i in xrange(len(self.compartment)):                                                              
            capacitance_nF[i] = self.compartment[i].capacitance_nF
            gLeak[i] = self.compartment[i].gLeak_muS
            EqPot[i] = self.compartment[i].EqPot_mV
            IPump[i] = self.compartment[i].IPump_nA
            compLength[i] = self.compartment[i].length_mum
            self.v_mV[i] = self.compartment[i].EqPot_mV
        
        
        ## Vector with  the inverse of the capacitance of all compartments. 
        self.capacitanceInv = 1.0 / capacitance_nF

        
        ## Vector with current, in nA,  of each compartment coming from other elements of the model. For example 
        ## from ionic channels and synapses.       
        self.iIonic = np.full_like(self.v_mV, 0.0)
        ## Vector with the current, in nA, injected in each compartment.
        self.iInjected = np.zeros_like(self.v_mV, dtype = 'd')
        #self.iInjected = np.array([0, 10.0])
        
        GC = compGCouplingMatrix(gCoupling_muS)
        
        GL = -np.diag(gLeak)
        
        ## Matrix of the conductance of the motoneuron. Multiplied by the vector self.v_mV,
        ## results in the passive currents of each compartment.
        self.G = np.float64(GC + GL)

        self.EqCurrent_nA = np.dot(-GL, EqPot) + IPump 

        
        ## index of the soma compartment.
        self.somaIndex = compartmentsList.index('soma')

        ## index of the last compartment.
        self.lastCompIndex = self.compNumber - 1
        
        ## Refractory period, in ms, of the motoneuron.
        self.MNRefPer_ms = float(conf.parameterSet('MNSomaRefPer', pool, index))
        
        # delay
        ## String with type of the nerve. It can be PTN (posterior tibial nerve) or CPN
        ## (common peroneal nerve).
        if pool == 'SOL' or pool == 'MG' or pool == 'LG':
            self.nerve = 'PTN'
        elif pool == 'TA':
            self.nerve = 'CPN'

        ## Distance, in m, of the stimulus position to the terminal. 
        self.stimulusPositiontoTerminal = float(conf.parameterSet('stimDistToTerm_' + self.nerve, pool, index))   
        ## AxonDelay object of the motor unit.
        if NumberOfAxonNodes == 0:
            dynamicNerveLength = 0
        else:
            dynamicNerveLength = np.sum(compLength[2:-1]) * 1e-6
        
        self.nerveLength = float(conf.parameterSet('nerveLength_' + self.nerve, pool, index))    

        delayLength =  self.nerveLength - dynamicNerveLength

        if self.stimulusPositiontoTerminal < delayLength:
            self.Delay = AxonDelay(conf, self.nerve, pool, delayLength, self.stimulusPositiontoTerminal, index)
            self.stimulusCompartment = 'delay'
        else:
            self.Delay = AxonDelay(conf, self.nerve, pool, delayLength, -1, index)
            self.stimulusCompartment = -1    
        # Nerve stimulus function    
        self.stimulusMeanFrequency_Hz = float(conf.parameterSet('stimFrequency_' + self.nerve, pool, 0))
        self.stimulusPulseDuration_ms = float(conf.parameterSet('stimPulseDuration_' + self.nerve, pool, 0))
        self.stimulusIntensity_mA = float(conf.parameterSet('stimIntensity_' + self.nerve, pool, 0))
        self.stimulusStart_ms = float(conf.parameterSet('stimStart_' + self.nerve, pool, 0))
        self.stimulusStop_ms = float(conf.parameterSet('stimStop_' + self.nerve, pool, 0))
        self.stimulusModulationStart_ms = float(conf.parameterSet('stimModulationStart_' + self.nerve, pool, 0))
        self.stimulusModulationStop_ms = float(conf.parameterSet('stimModulationStop_' + self.nerve, pool, 0))

        exec 'def axonStimModulation(t): return '   +  conf.parameterSet('stimModulation_' + self.nerve, pool, 0)
        
        startStep = int(np.rint(self.stimulusStart_ms / self.conf.timeStep_ms))
        self.axonStimModulation = axonStimModulation
        ## Vector with the nerve stimulus, in mA.
        self.nerveStimulus_mA = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        for i in xrange(len(self.nerveStimulus_mA)):
                if (i * self.conf.timeStep_ms >= self.stimulusStart_ms and  i * self.conf.timeStep_ms <= self.stimulusStop_ms):
                    if (i * self.conf.timeStep_ms > self.stimulusModulationStart_ms and  i * self.conf.timeStep_ms < self.stimulusModulationStop_ms):
                        stimulusFrequency_Hz = self.stimulusMeanFrequency_Hz + axonStimModulation(i * self.conf.timeStep_ms)
                    else:
                        stimulusFrequency_Hz = self.stimulusMeanFrequency_Hz
                    if stimulusFrequency_Hz > 0:
                        stimulusPeriod_ms = 1000.0 / stimulusFrequency_Hz
                        numberOfSteps = int(np.rint(stimulusPeriod_ms / self.conf.timeStep_ms))
                        if ((i - startStep) % numberOfSteps == 0):
                            self.nerveStimulus_mA[i:int(np.rint(i+self.stimulusPulseDuration_ms / self.conf.timeStep_ms))] = self.stimulusIntensity_mA
        
        # 
        ## Vector with the instants of spikes at the soma.
        self.somaSpikeTrain = []
        ## Vector with the instants of spikes at the last compartment.
        self.lastCompSpikeTrain = []
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
    
    def atualizeMotorUnit(self, t, v_mV):
        '''
        Atualize the dynamical and nondynamical (delay) parts of the motor unit.

        - Inputs:
            + **t**: current instant, in ms.
        ''' 
        self.atualizeCompartments(t, v_mV)
        self.atualizeDelay(t)
        

    #@profile        
    def atualizeCompartments(self, t, v_mV):
        '''
        Atualize all neural compartments.

        - Inputs:
            + **t**: current instant, in ms.

        '''        
        self.v_mV[:] = v_mV

        for i in xrange(self.somaIndex, self.compNumber):
            if self.v_mV[i] > self.threshold_mV and t-self.tSpikes[i] > self.MNRefPer_ms: 
                self.addCompartmentSpike(t, i)    
     
    
    #@profile
    def addCompartmentSpike(self, t, comp):
        '''
        When the soma potential is above the threshold a spike is added tom the soma.

        - Inputs:
            + **t**: current instant, in ms.

            + **comp**: integer with the compartment index.
        '''
        self.tSpikes[comp] = t
        if comp == self.somaIndex:
            self.somaSpikeTrain.append([t, int(self.index)])
            self.transmitSpikes(t)
        if comp == self.lastCompIndex:     
            self.lastCompSpikeTrain.append([t, int(self.index)])
            self.Delay.addSpinalSpike(t)
        
        for channel in self.compartment[comp].Channels:
            for channelState in channel.condState: channelState.changeState(t)    
              
              
    def atualizeDelay(self, t):
        '''
        Atualize the terminal spike train, by considering the Delay of the nerve.

        - Inputs:
            + **t**: current instant, in ms.
        '''

        if -1e-3 < (t - self.Delay.terminalSpikeTrain) < 1e-3: 
            self.terminalSpikeTrain.append([t, self.index])
                   
        
        # Check whether there is antidromic impulse reaching soma or RC
        if self.Delay.indexAntidromicSpike < len(self.Delay.antidromicSpikeTrain) and -1e-2 < (t - self.Delay.antidromicSpikeTrain[self.Delay.indexAntidromicSpike]) < 1e-2: 

            # Considers only MN-RC connections
            self.transmitSpikes(t)
            
            # Refractory period of MN soma
            if t-self.tSpikes[self.somaIndex] > self.MNRefPer_ms:
                self.tSpikes[self.somaIndex] = t
                self.somaSpikeTrain.append([t, int(self.index)])
                self.Delay.indexAntidromicSpike += 1
                for channel in self.compartment[self.somaIndex].Channels:
                    for channelState in channel.condState: channelState.changeState(t)    
           
        
        if self.stimulusCompartment == 'delay':
            self.Delay.atualizeStimulus(t, self.nerveStimulus_mA[int(np.rint(t/self.conf.timeStep_ms))])

    def transmitSpikes(self, t):
        '''
        - Inputs:
            + **t**: current instant, in ms.
        '''
        for i in xrange(len(self.indicesOfSynapsesOnTarget)):
            self.transmitSpikesThroughSynapses[i].receiveSpike(t, self.indicesOfSynapsesOnTarget[i])

    def getEMG(self, t):
        '''

        '''
        emg = 0
        numberOfSpikesUntilt = []
        ta = 0        

        if (len(self.terminalSpikeTrain) == 0):
            emg = 0
        else:
            for spike in self.terminalSpikeTrain:
                if spike[0] < t:
                    numberOfSpikesUntilt.append(spike[0])

        for spikeInstant in numberOfSpikesUntilt:
            ta = t - spikeInstant - 3 * self.timeCteEMG_ms
            if (ta <= 6 * self.timeCteEMG_ms):
                
                if (self.hrType == 1):
                    emg += 1.19 * self.ampEMG_mV * ta * math.exp(-(ta/self.timeCteEMG_ms)**2) / self.timeCteEMG_ms
                elif (self.hrType == 2):
                    emg += 0.69 * self.ampEMG_mV * (1 - 2*((ta / self.timeCteEMG_ms)**2)) * math.exp(-(ta/self.timeCteEMG_ms)**2)
               
        return emg

    def createStimulus(self):
        '''

        '''
        self.stimulusMeanFrequency_Hz = float(self.conf.parameterSet('stimFrequency_' + self.nerve, self.pool, 0))
        self.stimulusPulseDuration_ms = float(self.conf.parameterSet('stimPulseDuration_' + self.nerve, self.pool, 0))
        self.stimulusIntensity_mA = float(self.conf.parameterSet('stimIntensity_' + self.nerve, self.pool, 0))
        self.stimulusStart_ms = float(self.conf.parameterSet('stimStart_' + self.nerve, self.pool, 0))
        self.stimulusStop_ms = float(self.conf.parameterSet('stimStop_' + self.nerve, self.pool, 0))
        self.stimulusModulationStart_ms = float(self.conf.parameterSet('stimModulationStart_' + self.nerve, self.pool, 0))
        self.stimulusModulationStop_ms = float(self.conf.parameterSet('stimModulationStop_' + self.nerve, self.pool, 0))

        startStep = int(np.rint(self.stimulusStart_ms / self.conf.timeStep_ms))
        for i in xrange(len(self.nerveStimulus_mA)):
            if (i * self.conf.timeStep_ms >= self.stimulusStart_ms and  i * self.conf.timeStep_ms <= self.stimulusStop_ms):
                if (i * self.conf.timeStep_ms > self.stimulusModulationStart_ms and  i * self.conf.timeStep_ms < self.stimulusModulationStop_ms):
                    stimulusFrequency_Hz = self.stimulusMeanFrequency_Hz + self.axonStimModulation(i * self.conf.timeStep_ms)
                else:
                    stimulusFrequency_Hz = self.stimulusMeanFrequency_Hz
                if stimulusFrequency_Hz > 0:
                    stimulusPeriod_ms = 1000.0 / stimulusFrequency_Hz
                    numberOfSteps = int(np.rint(stimulusPeriod_ms / self.conf.timeStep_ms))
                    if ((i - startStep) % numberOfSteps == 0):
                        self.nerveStimulus_mA[i:int(np.rint(i+1.0 / self.conf.timeStep_ms))] = self.stimulusIntensity_mA


    def reset(self):
        '''

        '''
        self.tSomaSpike = float("-inf")
        self.v_mV = np.zeros((self.compNumber), dtype = np.float64)
        for i in xrange(len(self.compartment)): 
            self.v_mV[i] = self.compartment[i].EqPot_mV
            self.compartment[i].reset()
        self.Delay.reset()
        self.tSpikes = np.zeros((self.compNumber), dtype = np.float64)
        self.iIonic = np.full_like(self.v_mV, 0.0)
        self.iInjected = np.zeros_like(self.v_mV, dtype = 'd')

        self.somaSpikeTrain = []
        ## Vector with the instants of spikes at the last compartment.
        self.lastCompSpikeTrain = []
        ## Vector with the instants of spikes at the terminal.
        self.terminalSpikeTrain = []



