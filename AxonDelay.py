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

from Configuration import Configuration

class AxonDelay(object):
    '''
    Class that implements a delay correspondent to the nerve. This class corresponds to the part of the axon that is
    modeled with no dynamics. Ideally this  class would not exist and all the axon would be modelled in the motor unit
    or sensory class with the proper dynamics. 
    '''

    
    def __init__(self, conf, nerve, pool, length, index):
        '''
        Constructor


        - Inputs:
            + **conf**: Configuration object with the simulation parameters.

            + **nerve**: string with type of the nerve. It can be *PTN* 
            (posterior tibial nerve) or *CPN* (common peroneal nerve).

            + **pool**: string with Motor unit pool to which the motor unit belongs.

            + **length**: ## float, length of the part of the nerve that is
            modelled as a delay, in m.

            + **index**: integer corresponding to the motor unit order in the pool, according to 
            the Henneman's principle (size principle).
        '''

        ## Integer corresponding to the motor unit order in the pool, according to 
        ## the Henneman's principle (size principle).
        self.index = int(index)
        
        ## Length, in m, of the part of the nerve that is modelled as a delay.
        self.length_m = length
        ## Velocity of conduction, in m/s, of the part of the nerve that is not modelled as a delay.     
        self.velocity_m_s = float(conf.parameterSet('axonDelayCondVel',pool, index))
        ## Distance, in m, of the stimulus position to the terminal. 
        self.stimulusPositiontoTerminal = float(conf.parameterSet('stimDistToTerm_'+nerve, pool, index))   
        ## time, in ms, that the signal takes to travel between the stimulus and the spinal cord.        
        self.latencyStimulusSpinal_ms = round((self.length_m - self.stimulusPositiontoTerminal)/self.velocity_m_s*1000/conf.timeStep_ms, 0) * conf.timeStep_ms
        ## time, in ms, that the signal takes to travel between the spinal cord and the terminal.
        self.latencySpinalTerminal_ms = round((self.length_m)/self.velocity_m_s*1000/conf.timeStep_ms, 0) * conf.timeStep_ms
        ## time, in ms, tat the signal takes to travel between the stimulus and the terminal.
        self.latencyStimulusTerminal_ms = round((self.stimulusPositiontoTerminal)/self.velocity_m_s*1000/conf.timeStep_ms, 0) * conf.timeStep_ms
        
        ## Float with instant, in ms, of the last spike in the terminal. 
        self.terminalSpikeTrain = float("-inf")

        self.stimulusFrequency_Hz = conf.parameterSet('stimFrequency_' + self.nerve, pool, 0)
        self.stimulusIntensity_mA = conf.parameterSet('stimIntensity_' + self.nerve, pool, 0)
        self.stimulusStart_ms = conf.parameterSet('stimStart_' + self.nerve, pool, 0)
        self.stimulusStop_ms = conf.parameterSet('stimStop_' + self.nerve, pool, 0)

        self.threshold_mA = conf.parameterSet('axonDelayThreshold', pool, 0)
        exec 'def AxonStimFunction(t): return '   +  conf.parameterSet('stimModulation_' + self.nerve, pool, 0)
            

 
    def addTerminalSpike(self, t):
        '''
        Indicates to the AxonDelay object that a spike has occurred in the Terminal.

        - Inputs:
            + **t**: current instant, in ms.
        '''
        self.terminalSpikeTrain = t + self.latencySpinalTerminal_ms

    def addSpinalSpike(self, t):
        '''
        Indicates to the AxonDelay object that a spike has occurred in the last 
        dynamical compartment of the motor unit.

        - Inputs:
            + **t**: current instant, in ms.    
        '''    
        self.addTerminalSpike(t)