'''
Created on Oct 26, 2015

@author: root
'''

import math
import numpy as np




    

def compSynapCond(Gmax, Ron, Roff):
    return Gmax * (Ron + Roff)


def compRon(Non, rInf, ron, t0, t,  tauOn):
    return Non * rInf + (ron - Non * rInf) * math.exp((t0 - t) / tauOn) 


def compRoff(roff, t0, t, tauOff):
    return roff  * math.exp((t0 - t) / tauOff) 


def compRiStart(ri, t, ti, tPeak, tauOff):
    return ri * math.exp(-(t - ti- tPeak) / tauOff)


def compRiStop(rInf, ri, expFinish):
    return rInf + (ri - rInf) * expFinish
 

def compRonStart(Ron, ri, synContrib):
    return Ron + ri * synContrib

def compRoffStart(Roff, ri, synContrib):
    return Roff - ri * synContrib


def compRonStop(Ron, ri, synContrib):
    return Ron - ri * synContrib


def compRoffStop(Roff, ri, synContrib):
    return Roff + ri * synContrib


class Synapse(object):
    '''
    classdocs
    '''

    
    def __init__(self, conf, pool, index, compartment, kind, neuronKind):
        '''
        Constructor
        '''
        self.pool = pool
        self.kind = kind
        self.neuronKind = neuronKind
        
        self.EqPot_mV = float(conf.parameterSet('EqPotSyn_' + pool + '_'  + self.neuronKind + '_' + self.kind, pool, index))
        self.alpha_ms1 = float(conf.parameterSet('alphaSyn_' + self.kind + '_' + pool + '_'  + self.neuronKind, pool, index))
        self.beta_ms1 = float(conf.parameterSet('betaSyn_' + self.kind + '_' + pool + '_'  + self.neuronKind, pool, index))
        self.Tmax_mM = float(conf.parameterSet('TmaxSyn_' + self.kind + '_' + pool + '_'  + self.neuronKind, pool, index))
        self.tPeak_ms = float(conf.parameterSet('tPeakSyn_' + self.kind + '_' + pool + '_'  + self.neuronKind, pool, index))
        
        
        self.gmax_muS = np.array([])
        self.delay_ms = np.array([])
        self.dynamics = []
        

        
        
        self.gMaxTot_muS = 0
        self.numberOfIncomingSynapses = 0
        
        
        self.rInf = (self.alpha_ms1 * self.Tmax_mM) / (self.alpha_ms1 * self.Tmax_mM + self.beta_ms1)
        self.tauOn = 1.0 / (self.alpha_ms1 * self.Tmax_mM + self.beta_ms1)
        self.tauOff = 1.0 / self.beta_ms1
        self.expFinish = math.exp(- self.tPeak_ms/self.tauOn)
        
             
        self.Non = 0
        self.Ron = 0.0
        self.ron = 0.0
        self.Roff = 0.0
        self.roff = 0.0
        self.t0 = 0.0
        
        self.spikesReceived =[]
        
        self.conductanceState = np.array([])        
        self.tBeginOfPulse = np.array([])
        self.tEndOfPulse = np.array([])
        self.ri =  np.array([])
        self.ti =  np.array([])
        self.synContrib =  np.array([])
        self.startDynamicFunction = []
        self.stopDynamicFunction = [] 
       
        self.startEntrance = 0
        self.stopEntrance = 0
    
    
    def computeCurrent(self, t, V_mV):
        '''
        '''
        if len(self.tEndOfPulse)==0:
            self.tBeginOfPulse = np.ones_like(self.gmax_muS, dtype = float) * float("-inf")
            self.tEndOfPulse = np.ones_like(self.gmax_muS, dtype = float) * float("-inf")
            self.conductanceState = np.zeros_like(self.gmax_muS, dtype = int)
            self.ri = np.zeros_like(self.gmax_muS, dtype = float)
            self.ti = np.zeros_like(self.gmax_muS, dtype = float)
            self.synContrib = self.gmax_muS/self.gMaxTot_muS
            for dyn in xrange(len(self.dynamics)):
                if self.dynamics[dyn] == 'None': 
                    self.startDynamicFunction.append(self.startConductanceNone)
                    self.stopDynamicFunction.append(self.stopConductanceNone)
                else: self.startDynamicFunction.append(self.startConductanceDynamics) 
            self.computeCurrent = self.computeCurrent2
        
            
        return self.computeConductance(t) * (self.EqPot_mV - V_mV)
    
    
    def computeCurrent2(self, t, V_mV):
        
                        
        return self.computeConductance(t)  * (self.EqPot_mV - V_mV)
    
    
    def computeConductance(self, t):
        '''
        '''
        self.Ron, self.Roff = compRon(self.Non, self.rInf, self.ron, self.t0, t, self.tauOn), compRoff(self.roff, self.t0, t, self.tauOff) 
              
        self.startConductanceNone(t, np.where(np.abs(t-self.tBeginOfPulse < 1e-3))[0])
        self.stopConductanceNone(t, np.where(np.abs(t-self.tEndOfPulse) < 1e-3)[0])
        
        
        return compSynapCond(self.gMaxTot_muS, self.Ron, self.Roff)
            
        
        
   
    def startConductanceNone(self, t, idxBeginPulse):
        '''
        ''' 
        for synapseNumber in idxBeginPulse:      
            if self.conductanceState[synapseNumber] == 0:
                self.ri.itemset(synapseNumber, compRiStart(self.ri.item(synapseNumber), t,  self.ti.item(synapseNumber),  self.tPeak_ms, self.tauOff))
                self.ti.itemset(synapseNumber, t)                    
                self.ron = compRonStart(self.Ron, self.ri.item(synapseNumber), self.synContrib.item(synapseNumber))
                self.roff = compRoffStart(self.Roff, self.ri.item(synapseNumber), self.synContrib.item(synapseNumber))
                self.Non += self.synContrib.item(synapseNumber)
                self.t0 = t
                self.conductanceState.itemset(synapseNumber, 1)
            
            self.tEndOfPulse.itemset(synapseNumber, t + self.tPeak_ms)
            self.tBeginOfPulse.itemset(synapseNumber, -1000000)
    
    
    def startConductanceDynamics(self, t, synapsesNumber):
        '''
        '''
        
        
        
    
    def stopConductanceNone(self, t, idxEndPulse):
        
        for synapseNumber in idxEndPulse:      
            self.ri.itemset(synapseNumber, compRiStop(self.rInf, self.ri.item(synapseNumber), self.expFinish))
            self.t0 = t
            self.ron = compRonStop(self.Ron, self.ri.item(synapseNumber), self.synContrib.item(synapseNumber))
            self.roff = compRoffStop(self.Roff, self.ri.item(synapseNumber), self.synContrib.item(synapseNumber))
            self.Non -= self.synContrib.item(synapseNumber)
            self.tEndOfPulse.itemset(synapseNumber, -10000)
            self.conductanceState.itemset(synapseNumber, 0)  
    
    
    
    def stopConductanceDynamics(self, t, synapseNumber):
        '''
        '''
        
    
    def receiveSpike(self, t, synapseNumber):
        '''
        '''
        
        self.tBeginOfPulse[synapseNumber] = t+self.delay_ms[synapseNumber]
        
      
    def addConductance(self, gmax, delay, dynamics, weight):
        self.gMaxTot_muS += gmax
        self.numberOfIncomingSynapses += 1
        self.gmax_muS = np.append(self.gmax_muS, gmax)
        self.delay_ms = np.append(self.delay_ms, delay)
        self.dynamics.append(dynamics)
        
        
    
        