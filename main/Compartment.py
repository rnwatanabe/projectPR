'''
Author: Renato Naville Watanabe
'''
from ChannelConductance import ChannelConductance

from Synapse import Synapse
import math


def calcGLeak(area, specificRes):
    '''
    computes the leak conductance of the compartment
    input: area: area of the compartment in cm2
                specificRes: specific resistance of the compartment in Ohm.cm2
    output: gLeak in MS 
    '''    
    return (1e6 * area) / specificRes

class Compartment(object):
    '''
    classdocs
    '''


    def __init__(self, kind, conf, pool, index, neuronKind):
        '''
        Constructor
        '''
        
        self.Channels = []
        self.neuronKind = neuronKind
        self.SynapsesOut = []
        
        self.SynapsesIn = [] 
        self.SynapsesIn.append(Synapse(conf, pool, index, kind, 'excitatory', neuronKind))
        self.SynapsesIn.append(Synapse(conf, pool, index, kind, 'inhibitory', neuronKind))
        
        self.kind = kind
        
        
        self.index = index
        
        self.length_mum = float(conf.parameterSet('l_' + kind, pool, index))
        self.diameter_mum = float(conf.parameterSet('d_' + kind, pool, index))    
        self.area_cm2 = float(self.length_mum * math.pi * self.diameter_mum * 1e-8)
        self.specifRes_Ohmcm2 = float(conf.parameterSet('res_' + kind, pool, index))
        self.capacitance_nF = float(float(conf.parameterSet('membCapac',pool, index)) * self.area_cm2 * 1e3)
        self.gLeak = calcGLeak(self.area_cm2, self.specifRes_Ohmcm2)
        
        
        if (kind == 'soma'):              
            self.Channels.append(ChannelConductance('Kf', conf, self.area_cm2, pool, index))
            self.Channels.append(ChannelConductance('Ks', conf, self.area_cm2, pool, index))
            self.Channels.append(ChannelConductance('Na', conf, self.area_cm2, pool, index))            
        elif (kind == 'dendrite'):
            pass      
        
        self.numberChannels = len(self.Channels)
        self.numberofMultiSynapses = len(self.SynapsesIn)
      
         
    def computeCurrent(self, t, V_mV):
        
        I = 0
        if (self.numberChannels != 0 ):
            for i in xrange(0,self.numberChannels): I += self.Channels[i].computeCurrent(t, V_mV)
        if self.SynapsesIn[0].numberOfIncomingSynapses : I += self.SynapsesIn[0].computeCurrent(t, V_mV)
        if self.SynapsesIn[1].numberOfIncomingSynapses : I += self.SynapsesIn[1].computeCurrent(t, V_mV)
         
        return I             