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
    calculates the coupling conductance between two compartments
    Inputs: cytR. Cytoplasmatic resistance in Ohm.cm
                lComp1, lComp2: length of the compartments in mum
                dComp1, dComp2 diameter of the compartments in mum
    Output: coupling conductance in MS
    '''
    rAxis1 = (cytR * lComp1) / (math.pi * math.pow(dComp1/2, 2))
    rAxis2 = (cytR * lComp2) / (math.pi * math.pow(dComp2/2, 2))
    
    return (1e2 * 2) / (rAxis1 + rAxis2)




def compGCouplingMatrix(gc):
    '''
    computes the Coupling Matrix to be used in the dVdt function of the N compartments of the motor unit. 
    The Matrix uses the values obtained with the function calcGcoupling.
                _______________________________________________________
                |-gc[0]           gc[0]              0      ....                    .....          0        0          0|
                |gc[0]    -gc[0]-gc[1]    gc[1]    0  .....                     .....                           0|
                |  .      .                ...                        ...                                .....                      0|  
    GC =   |  :        . .  . ........................................                                                           :|
                |  0  .......   0         gc[i-1]    -gc[i-1]-gc[i]    gc[i]    0     .....            0        :|
                |  0 ...    ...............................                                ....        ..............................|
                |  0  .............................................gc[N-2]    -gc[N-2]-gc[N-1]    gc[N-1]|
                |  0 ..........................................................................  gc[N-1]        -gc[N-1]|
                |---------------------------------------------------------------------------------------------------|
    Inputs: the vector with N elements, with the coupling conductance of each compartment of the Motor Unit.
    Output: the GC matrix
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
    
    '''       
    k1 = derivativeFunction(t, x)
    k2 = derivativeFunction(t + timeStepByTwo, x + timeStepByTwo * k1)
    k3 = derivativeFunction(t + timeStepByTwo, x + timeStepByTwo * k2)
    k4 = derivativeFunction(t + timeStep, x + timeStep * k3)
    
    return x + timeStepBySix * (k1 + k2 + k2 + k3 + k3 + k4)



class MotorUnit(object):
    '''
    classdocs
    '''
   

    def __init__(self, conf, pool, index, kind):
        '''
        Constructor
        '''
        self.conf = conf
        self.kind = kind
        ## Neural compartments
        self.tSomaSpike = float("-inf")
        compartmentsList = ['dendrite', 'soma']
        self.somaSpikeTrain = []
        self.index = int(index)
        self.compartment = []
        self.threshold_mV = conf.parameterSet('threshold', pool, index)
                
        for i in compartmentsList: self.compartment.append(Compartment(i, conf, pool, index, self.kind))        
        
        self.compNumber = len(self.compartment)
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
            
         
        self.capacitanceInv = 1 / capacitance_nF
       
        
        self.iIonic = np.full_like(self.v_mV, 0.0)  
        self.iSynaptic = np.full_like(self.v_mV, 0)
        self.iInjected = np.zeros_like(self.v_mV, dtype = 'd')
        #self.iInjected = np.array([0, 10.0])
        
        GC = compGCouplingMatrix(gCoupling_MS)
        
        GL = -np.diag(gLeak)
        self.G = np.zeros_like(GC, dtype = float)
        self.G = np.float64(GC + GL)
       
        
        
           
        
        self.somaIndex = compartmentsList.index('soma')
        
        self.MNRefPer_ms = float(conf.parameterSet('MNSomaRefPer', pool, index))
        
        ## delay
        if (pool == 'SOL' or pool == 'MG' or pool == 'LG'):
            self.nerve = 'PTN'
        else:
            self.nerve = 'CPN'
            
        self.Delay = AxonDelay(conf, self.nerve, pool, index)
        self.terminalSpikeTrain = []
                
        
        ## contraction Data
        self.activationModel = conf.parameterSet('activationModel', pool, 0)
        
        self.TwitchTc_ms = conf.parameterSet('twitchTimePeak', pool, index)
        self.TwitchAmp_N = conf.parameterSet('twitchPeak', pool, index)
        self.bSat = conf.parameterSet('bSat'+self.activationModel,pool,index)
        self.twTet = conf.parameterSet('twTet'+self.activationModel,pool,index)
        
        ## EMG data
        
        
    
    def atualizeMotorUnit(self, t): 
        self.atualizeCompartments(t)
        self.atualizeDelay(t)
        
    def atualizeCompartments(self, t):
        '''
        atualize all neural compartments
        '''
        
        np.clip(runge_kutta(self.dVdt, t, self.v_mV, self.conf.timeStep_ms, self.conf.timeStepByTwo_ms, self.conf.timeStepBySix_ms), -16.0, 120.0, self.v_mV)
        if (self.v_mV[self.somaIndex] > self.threshold_mV and t-self.tSomaSpike > self.MNRefPer_ms): self.addSomaSpike(t)    
     
       
    def dVdt(self, t, V): 
        '''
        compute the potential derivative of all compartments of
        the motor unit
        '''
        for compartment in xrange(0, self.compNumber):  
            self.iIonic.itemset(compartment, self.compartment[compartment].computeCurrent(t, V.item(compartment)))
        
        
        return (self.iIonic + np.dot(self.G, V)  + self.iInjected) * self.capacitanceInv
    
    
    def addSomaSpike(self, t):
        '''
        when the soma potential is above the threshold
        a spike is added tom the soma
        '''
        self.tSomaSpike = t
        self.somaSpikeTrain.append([t, int(self.index)])
        self.Delay.addSpinalSpike(t)
        
        for channel in self.compartment[self.somaIndex].Channels:
            for channelState in channel.condState: channelState.changeState(t)    
              
              
    def atualizeDelay(self, t):
        '''
        '''
        if abs(t - self.Delay.terminalSpikeTrain) < 1e-3: 
            self.terminalSpikeTrain.append([t, self.index])
        
    
        