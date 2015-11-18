'''
Created on Oct 26, 2015

@author: root
'''

from PointProcessGenerator import PointProcessGenerator
import math
import numpy as np
from multiprocessing import Pool
import functools



class NeuralTractUnit(object):
    '''
    classdocs
    '''

    
    def __init__(self, conf, pool, index):
        '''
        Constructor
        '''     
        #   point process generator data    
        self.GammaOrder = int(conf.parameterSet('GammaOrder_' + pool, pool, index))
        
          
        self.spikesGenerator = PointProcessGenerator(self.GammaOrder, index)        
        self.terminalSpikeTrain = self.spikesGenerator.points
        
        
        
         
        # Build synapses       
        self.SynapsesOut = []
        self.transmitSpikesThroughSynapses = []
        self.indicesOfSynapsesOnTarget = []
        
       
    
      
    def atualizeNeuralTractUnit(self, t, FR):
        '''
        '''        
        
        self.spikesGenerator.atualizeGenerator(t, FR)
        if self.terminalSpikeTrain and abs(t - self.terminalSpikeTrain[-1][0]) < 1e-3: self.transmitSpikes(t)
        
        
    
    def transmitSpikes(self, t):
        '''
        '''
        for i in xrange(len(self.indicesOfSynapsesOnTarget)):
            self.transmitSpikesThroughSynapses[i].receiveSpike(t, self.indicesOfSynapsesOnTarget[i])
        