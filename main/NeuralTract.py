'''
Created on Oct 26, 2015

@author: root
'''
from NeuralTractUnit import NeuralTractUnit
import numpy as np

class NeuralTract(object):
    '''
    classdocs
    '''


    def __init__(self, conf, pool):
        '''
        Constructor

        - Inputs:
            + **conf**:

            + **pool**: 
        '''
        self.kind = 'NT'
        self.pool = pool
        self.Number = int(conf.parameterSet('Number_' + pool, pool, 0))
        
        self.unit = [] 
        
        for i in xrange(0, self.Number): self.unit.append(NeuralTractUnit(conf, pool, i))
        self.poolTerminalSpikes = np.array([]) 
        
        self.target = conf.parameterSet('Target_' + pool, pool, 0)
        if self.target == 'ISI' :       
            exec 'def Targetfunction(t): return 1000.0/('  +  conf.parameterSet('TargetFunction_' + pool, pool, 0) + ')'
        else:
            exec 'def Targetfunction(t): return '   +  conf.parameterSet('TargetFunction_' + pool, pool, 0)
        
        
        self.FR = conf.inputFunctionGet(Targetfunction) * conf.timeStep_ms/1000.0    
        
        
        self.timeIndex = 0
        ##
        print 'Descending Command '  + pool + ' built'
    
    def atualizePool(self, t):    
        for i in self.unit: i.atualizeNeuralTractUnit(t, self.FR[self.timeIndex])
        self.timeIndex +=1
        
        
    def listSpikes(self):
        
        #spikeTrain = np.zeros((self.MUnumber,2))
        
        for i in xrange(0,self.Number):
            if i == 0: terminalSpikeTrain = np.array(self.unit[i].terminalSpikeTrain)
            else: terminalSpikeTrain = np.append(terminalSpikeTrain, np.array(self.unit[i].terminalSpikeTrain))
        self.poolTerminalSpikes = terminalSpikeTrain
            
        self.poolTerminalSpikes = np.reshape(self.poolTerminalSpikes, (-1, 2))
        
        