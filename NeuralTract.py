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
from NeuralTractUnit import NeuralTractUnit
import numpy as np

class NeuralTract(object):
    '''
    Class that implements a a neural tract, composed by the descending
    commands from the motor cortex.
    '''
    
    def __init__(self, conf, pool):
        '''
        Constructor

        - Inputs:
            + **conf**: Configuration object with the simulation parameters.

            + **pool**: string with the name of the Neural tract.
        '''
        ## Indicates that is a neural tract.
        self.kind = 'NT'
        ## String with the name of the Neural tract.
        self.pool = pool
        ## The number of neural tract units.
        self.Number = int(conf.parameterSet('Number_' + pool, pool, 0))
        
        ## List of NeuralTRactUnit objects.
        self.unit = dict() 
        
        self.GammaOrder = int(conf.parameterSet('GammaOrder_' + pool, pool, 0))

        for i in xrange(0, self.Number): 
            self.unit[i] = NeuralTractUnit(conf, pool, self.GammaOrder, i)
        ## Vector with the instants of spikes in the terminal, in ms.
        self.poolTerminalSpikes = np.array([]) 
        ## Indicates the measure that the TargetFunction of the
        ## spikes follows. For now it can be *ISI* (interspike
        ## interval) or *FR* (firing rate).
        self.target = conf.parameterSet('DriveTarget_' + pool, pool, 0)
        if self.target == 'ISI' :       
            exec 'def DriveFunction(t): return 1000.0/('  +  conf.parameterSet('DriveFunction_' + pool, pool, 0) + ')'
        else:
            exec 'def DriveFunction(t): return '   +  conf.parameterSet('DriveFunction_' + pool, pool, 0)
        
        ## The  mean firing rate of the neural tract units. 
        self.FR = conf.inputFunctionGet(DriveFunction) * conf.timeStep_ms/1000.0
        
        ## 
        self.timeIndex = 0
        ##
        print 'Descending Command ' + pool + ' built'
    
    def atualizePool(self, t):
        '''
        Update all neural tract units from the neural tract.
        
        - Inputs:
            + **t**: cuurent instant, in ms.
        '''    
        for i in xrange(self.Number): self.unit[i].atualizeNeuralTractUnit(t, self.FR[self.timeIndex])
        self.timeIndex +=1        
        
    def listSpikes(self):
        '''
        List the spikes that occurred in neural tract units.
        '''
        
        #spikeTrain = np.zeros((self.MUnumber,2))
        
        for i in xrange(0,self.Number):
            if i == 0: terminalSpikeTrain = np.array(self.unit[i].terminalSpikeTrain)
            else: terminalSpikeTrain = np.append(terminalSpikeTrain, np.array(self.unit[i].terminalSpikeTrain))
        self.poolTerminalSpikes = terminalSpikeTrain
            
        self.poolTerminalSpikes = np.reshape(self.poolTerminalSpikes, (-1, 2))
        
        