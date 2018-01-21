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

    Contact: renato.watanabe@usp.br
'''

from PointProcessGenerator import PointProcessGenerator
import math
import numpy as np
from multiprocessing import Pool
import functools



class NeuralTractUnit(object):
    '''
    Class that implements a neural tract unit. 
    It consists of a point process generator.
    '''

    
    def __init__(self, conf, pool, index):
        '''
        Constructor

        - Inputs:
            + **conf**: Configuration object with the simulation parameters.

            + **pool**: string with the name of the Neural tract.

            + **index**: integer corresponding to the neural tract unit identification.

        '''     
        
        
        ## A PointProcessGenerator object, corresponding the generator of
        ## spikes of the neural tract unit.   
        self.spikesGenerator = PointProcessGenerator(index)  
        ## List of the spikes of the neural tract unit.       
        self.terminalSpikeTrain = self.spikesGenerator.points
        
        self.kind = ''
        
         
        # Build synapses       
        ## 
        self.SynapsesOut = []
        self.transmitSpikesThroughSynapses = []
        self.indicesOfSynapsesOnTarget = []
        
        ## Integer corresponding to the neural tract unit identification.
        self.index = index
    
      
    def atualizeNeuralTractUnit(self, t, FR, GammaOrder):
        '''

        - Inputs:
            + **t**: current instant, in ms.

            + **FR**:
        '''

        self.spikesGenerator.atualizeGenerator(t, FR, GammaOrder)
        if self.terminalSpikeTrain and -1e-3 < (t - self.terminalSpikeTrain[-1][0]) < 1e-3:
            self.transmitSpikes(t)

    def transmitSpikes(self, t):
        '''
        - Inputs:
            + **t**: current instant, in ms.
        '''
        for i in xrange(len(self.indicesOfSynapsesOnTarget)):
            self.transmitSpikesThroughSynapses[i].receiveSpike(t, self.indicesOfSynapsesOnTarget[i])

    def reset(self):
        self.spikesGenerator.reset()
        self.terminalSpikeTrain = self.spikesGenerator.points
    