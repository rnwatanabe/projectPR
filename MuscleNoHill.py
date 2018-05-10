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
        ## This is used for normalization purposes. It is the maximum force that
        ## the muscle reach when the Hill model is not used. 
        self.maximumActivationForce = self.twitchAmp_N * self.twTet    
        ## Muscle force along time, in N.
        self.force = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        
        self.timeIndex = 0


        ##
        self.lengthNorm = 1
        self.velocityNorm = 0
        self.accelerationNorm = 0 

        ##
        self.momentArm_m = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)

        ##  
        self.m0 = float(self.conf.parameterSet('m0:' + pool, pool, 0))       
        ##  
        self.m1 = float(self.conf.parameterSet('m1:' + pool, pool, 0))
        ##  
        self.m2 = float(self.conf.parameterSet('m2:' + pool, pool, 0))
        ##  
        self.m3 = float(self.conf.parameterSet('m3:' + pool, pool, 0))
        ##  
        self.m4 = float(self.conf.parameterSet('m4:' + pool, pool, 0))

        ##  
        self.n0 = float(self.conf.parameterSet('n0:' + pool, pool, 0))       
        ##  
        self.n1 = float(self.conf.parameterSet('n1:' + pool, pool, 0))
        ##  
        self.n2 = float(self.conf.parameterSet('n2:' + pool, pool, 0))
        ##  
        self.n3 = float(self.conf.parameterSet('n3:' + pool, pool, 0))
        ##  
        self.n4 = float(self.conf.parameterSet('n4:' + pool, pool, 0))

    #@profile    
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
        self.force[self.timeIndex] = np.vdot(activation_Sat, self.maximumActivationForce)
        self.timeIndex += 1

    def atualizeMusculoTendonLength(self, ankleAngle):
        '''
        '''
        pass

    def atualizeMomentArm(self, ankleAngle):
        '''
        '''
        self.momentArm_m[self.timeIndex] = (self.n0 + self.n1 * ankleAngle + self.n2 * (ankleAngle ** 2) + 
                                                      self.n3 * (ankleAngle ** 3) + self.n4 * (ankleAngle ** 4))


    def reset(self):
        '''

        '''
        self.force = np.zeros((int(np.rint(self.conf.simDuration_ms/self.conf.timeStep_ms)), 1), dtype = float)        
        self.timeIndex = 0
        