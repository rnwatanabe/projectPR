'''
Created on Jul, 28 2015

@author: root
'''

import numpy as np

class MuscleNoHill(object):

    def __init__(self, conf, pool, MUnumber, MUtypeInumber, unit):
        
        self.conf = conf
        self.pool = pool
        self.MUnumber = MUnumber
        self.MUtypeInumber = MUtypeInumber

        ## Twitch- tetanus relationship (see atualizeForceNoHill function explanation)
        self.twTet = np.zeros((self.MUnumber, 1), dtype = float)
        ## Amplitude of the muscle unit twitch, in N (see atualizeForceNoHill function explanation).
        self.twitchAmp_N = np.zeros((self.MUnumber, 1), dtype = float)

        for i in xrange(0, self.MUnumber):
            self.twitchAmp_N[i] = unit[i].TwitchAmp_N
            self.twTet[i] = unit[i].twTet  
        ## Muscle force along time, in N.
        self.force = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        

        self.timeIndex = 0;

    def atualizeForce(self, activation_Sat):
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
        self.force[self.timeIndex] = np.sum(activation_Sat * self.twitchAmp_N * self.twTet)
        self.timeIndex += 1
        