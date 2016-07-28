'''
Created on Oct 26, 2015

@author: root
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
        self.unit = [] 
        
        for i in xrange(0, self.Number): self.unit.append(NeuralTractUnit(conf, pool, i))
        ## Vector with the instants of spikes in the terminal, in ms.
        self.poolTerminalSpikes = np.array([]) 
        ## Indicates the measure that the TargetFunction of the
        ## spikes follows. For now ita can be *ISI* (interspike
        ## interval) or *FR* (firing rate).
        self.target = conf.parameterSet('Target_' + pool, pool, 0)
        if self.target == 'ISI' :       
            exec 'def Targetfunction(t): return 1000.0/('  +  conf.parameterSet('TargetFunction_' + pool, pool, 0) + ')'
        else:
            exec 'def Targetfunction(t): return '   +  conf.parameterSet('TargetFunction_' + pool, pool, 0)
        
        ## The  mean firing rate of the neural tract units. 
        self.FR = conf.inputFunctionGet(Targetfunction) * conf.timeStep_ms/1000.0
        
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
        for i in self.unit: i.atualizeNeuralTractUnit(t, self.FR[self.timeIndex])
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
        
        