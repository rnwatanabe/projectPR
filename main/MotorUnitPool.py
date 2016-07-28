'''
Created on Oct 8, 2015

@author: root
'''

import numpy as np
from MotorUnit import MotorUnit
from MuscularActivation import MuscularActivation
import math
from scipy.sparse import lil_matrix

 





class MotorUnitPool(object):
    '''
    Class that implements a motor unit pool. Encompasses a set of motor
    units that controls a single  muscle.
    '''


    def __init__(self, conf, pool):
        '''
        Constructor

        - Inputs:
            + **conf**: Configuration object with the simulation parameters.

            + **pool**: string with Motor unit pool to which the motor unit belongs.
        '''

        ## Indicates that is Motor Unit pool.
        self.kind = 'MU'

        ## Configuration object with the simulation parameters.
        self.conf = conf
        ## String with Motor unit pool to which the motor unit belongs.
        self.pool = pool
        MUnumber_S = int(conf.parameterSet('MUnumber_S_' + pool, pool, 0))
        MUnumber_FR = int(conf.parameterSet('MUnumber_FR_' + pool, pool, 0))
        MUnumber_FF = int(conf.parameterSet('MUnumber_FF_' + pool, pool, 0))
        ## Number of motor units.
        self.MUnumber = MUnumber_S + MUnumber_FR + MUnumber_FF
        
        ## List of MotorUnit objects.
        self.unit = []
        
        
        for i in xrange(0, self.MUnumber): 
            if i < MUnumber_S:
                self.unit.append(MotorUnit(conf, pool, i, 'S'))
            elif i < MUnumber_S + MUnumber_FR:
                self.unit.append(MotorUnit(conf, pool, i, 'FR'))
            else:
                self.unit.append(MotorUnit(conf, pool, i, 'FF'))

        ## Vector with the instants of spikes in the soma compartment, in ms.            
        self.poolSomaSpikes = np.array([])    
        ## Vector with the instants of spikes in the terminal, in ms.
        self.poolTerminalSpikes = np.array([])
        
        #activation signal
        self.Activation = MuscularActivation(self.conf,self.pool, self.MUnumber,self.unit)
        
        #Force
        ## Twitch- tetanus relationship (see atualizeForceNoHill function explanation)
        self.twTet = np.zeros((self.MUnumber, 1), dtype = float)
        ## Amplitude of the muscle unit twitch, in N (see atualizeForceNoHill function explanation).
        self.twitchAmp_N = np.zeros((self.MUnumber, 1), dtype = float)

        for i in xrange(0, self.MUnumber):
            self.twitchAmp_N[i] = self.unit[i].TwitchAmp_N
            self.twTet[i] = self.unit[i].twTet  
        ## Muscle force along time, in N.
        self.force = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ## String indicating whther a Hill model is used or not. For now, it can be *No*.
        self.hillModel = conf.parameterSet('hillModel',pool, 0)
        if self.hillModel == 'No': self.atualizeForce = self.atualizeForceNoHill
        
        self.timeIndex = 0;
        
        
        ##
        print 'Motor Unit Pool ' + pool + ' built'
        
    def atualizeMotorUnitPool(self, t):
        '''
        Update all parts of the Motor Unit pool. It consists
        to update all motor units, the activation signal and
        the muscle force.

        - Inputs:
            + **t**: current instant, in ms.
        '''
        for i in self.unit: i.atualizeMotorUnit(t)
        self.Activation.atualizeActivationSignal(t, self.unit)
        self.atualizeForce()
        self.timeIndex += 1

    
        
    
    def atualizeForceNoHill(self):
        '''
        Compute the muscle force when no muscle dynamics (Hill model) is used. This
        operation is vectorized. Each element of the vectors correspond to one motor
        unit. For each motor unit, the force is computed by the following formula:

        \f{equation}{
            F_{MU} = a_{sat}A_{MU}R_{MU}
        }
        where \f$a_{sat}\f$ is the saturated activation signal, \f$A_{MU}\f$ is the
        motor unit twitch amplitude, and  \f$R_{MU}\f$ is the relation between 
        the twitch amplitude and the tetanus of the motor unit. 

        Then the muscle force is obtained from: 

        \f{equation}{
            F = \limits\sum_{i=1}^N_{MU}F_{i}
        }
        where \f$N_{MU}\f$ is the number of motor units in the pool.
        '''
        self.force[self.timeIndex] = np.sum(self.Activation.activation_Sat * self.twitchAmp_N * self.twTet)            

    def listSpikes(self):
        '''
        List the spikes that occurred in the soma and in
        the terminal of the different motor units.
        '''
        for i in xrange(0,self.MUnumber):
            if i == 0:
                somaSpikeTrain = np.array(self.unit[i].somaSpikeTrain)
                terminalSpikeTrain = np.array(self.unit[i].terminalSpikeTrain)
            else:
                somaSpikeTrain = np.append(somaSpikeTrain, np.array(self.unit[i].somaSpikeTrain))
                terminalSpikeTrain = np.append(terminalSpikeTrain, np.array(self.unit[i].terminalSpikeTrain))
        self.poolSomaSpikes = somaSpikeTrain
        self.poolTerminalSpikes = terminalSpikeTrain
            
        self.poolSomaSpikes = np.reshape(self.poolSomaSpikes, (-1, 2))
        self.poolTerminalSpikes = np.reshape(self.poolTerminalSpikes, (-1, 2))
        
    
        
        
    
        
          