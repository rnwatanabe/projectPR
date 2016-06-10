'''
Class: AxonDelay
'''

from Configuration import Configuration



class AxonDelay(object):
    '''
    Class that implements a delay correspondent to the nerve. This class corresponds to the part of the axon that is
    modeled with no dynamics.
    '''

    
    def __init__(self, conf, nerve, pool, index):
        '''
        Constructor


        Inputs:
            - conf
        '''
        self.index = int(index)
        
        self.length_m = float(conf.parameterSet('nerveLength_' + nerve, pool, index))     
        self.velocity_m_s = float(conf.parameterSet('axonCondVel',pool, index))
        self.stimulusPositiontoTerminal = float(conf.parameterSet('stimDistToTerm_'+nerve, pool, index))   
        
        self.latencyStimulusSpinal_ms = round((self.length_m - self.stimulusPositiontoTerminal)/self.velocity_m_s*1000/conf.timeStep_ms, 0) * conf.timeStep_ms
        self.latencySpinalTerminal_ms = round((self.length_m)/self.velocity_m_s*1000/conf.timeStep_ms, 0) * conf.timeStep_ms
        self.latencyStimulusTerminal_ms = round((self.stimulusPositiontoTerminal)/self.velocity_m_s*1000/conf.timeStep_ms, 0) * conf.timeStep_ms
        
    
        self.terminalSpikeTrain = float("-inf")

 
    def addTerminalSpike(self, t):
        self.terminalSpikeTrain = t+self.latencySpinalTerminal_ms
        
    
    def addSpinalSpike(self, t):    
        self.addTerminalSpike(t)