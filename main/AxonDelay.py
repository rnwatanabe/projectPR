'''
Created on Oct 23, 2015

@author: root
'''

from Configuration import Configuration



class AxonDelay(object):
    '''
    classdocs
    '''

    
    def __init__(self, conf, nerve, pool, index):
        '''
        Constructor
        '''
        self.index = int(index)
        
        self.Length = float(conf.parameterSet('nerveLength_' + nerve, pool, index))     
        self.velocity_m_s = float(conf.parameterSet('axonCondVel',pool, index))
        self.stimulusPositiontoTerminal = float(conf.parameterSet('stimDistToTerm_'+nerve, pool, index))   
        
        self.latencyStimulusSpinal_ms = round((self.Length - self.stimulusPositiontoTerminal)/self.velocity_m_s*1000/conf.timeStep_ms, 0) * conf.timeStep_ms
        self.latencySpinalTerminal_ms = round((self.Length)/self.velocity_m_s*1000/conf.timeStep_ms, 0) * conf.timeStep_ms
        self.latencyStimulusTerminal_ms = round((self.stimulusPositiontoTerminal)/self.velocity_m_s*1000/conf.timeStep_ms, 0) * conf.timeStep_ms
        
    
        self.terminalSpikeTrain = float("-inf")

 
    def addTerminalSpike(self, t):
        self.terminalSpikeTrain = t+self.latencySpinalTerminal_ms
        
    
    def addSpinalSpike(self, t):    
        self.addTerminalSpike(t)