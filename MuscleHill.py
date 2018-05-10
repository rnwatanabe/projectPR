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

import math
import numpy as np


class MuscleHill(object):

    def __init__(self, conf, pool, MUnumber, MUtypeInumber, unit):

        ##
        self.conf = conf
        ##
        self.pool = pool
        ##
        self.MUnumber = MUnumber
        ##
        self.MUtypeInumber = MUtypeInumber
        ##       
        self.timeIndex = 0
        
        ## Twitch-tetanus relationship (see atualizeForce function explanation)
        self.twTet = np.zeros((self.MUnumber, 1), dtype = float)
        ## Amplitude of the muscle unit twitch, in N (see atualizeForce function explanation).
        self.twitchAmp_N = np.zeros((self.MUnumber, 1), dtype = float)
        
        for i in xrange(0, self.MUnumber):
            self.twitchAmp_N[i] = unit[i].TwitchAmp_N
            self.twTet[i] = unit[i].twTet

        ## This is used for normalization purposes. It is the maximum force that
        ## the muscle reach when the Hill model is not used. 
        self.maximumActivationForce = sum(self.twitchAmp_N * self.twTet)   
        print(self.maximumActivationForce) 
        ## Muscle force along time, in N.
        self.force = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ##
        self.tendonForce_N = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ##
        self.contractileForce_N = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ##
        self.elasticForce_N = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ##
        self.viscousForce_N = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ##
        self.length_m = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ##
        self.velocity_m_ms = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ##
        self.acceleration_m_ms2 = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ##
        self.tendonLength_m = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ##
        self.pennationAngle_rad = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ##
        self.activationTypeI = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ##
        self.activationTypeII = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ##
        self.musculoTendonLength_m = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ##
        self.momentArm_m = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ##
        self.optimalLength_m = float(self.conf.parameterSet('optimalMuscleLength:' + pool, pool, 0))
        ##
        self.pennationAngleAtOptimalLengthSin = math.sin(float(self.conf.parameterSet('optimalPennationAngle:' + pool, pool, 0)))
        ## Maximum force of the Hill model, in N.
        self.maximumForce_N = float(self.conf.parameterSet('Fmax:' + pool, pool, 0))
        
        ##  
        self.elasticity = float(self.conf.parameterSet('muscleElasticity:' + pool, pool, 0))
        ##  
        self.strain = float(self.conf.parameterSet('muscleStrain:' + pool, pool, 0))
        ##  
        self.viscosity = float(self.conf.parameterSet('muscleViscosity:' + pool, pool, 0))
        ##  
        self.mass = float(self.conf.parameterSet('muscleMass:' + pool, pool, 0))
        ##  
        self.length_m[0] = float(self.conf.parameterSet('initialMuscleLength:' + pool, pool, 0))
        ##
        self.tendonElasticity = float(self.conf.parameterSet('tendonElasticity:' + pool, pool, 0))
        ##
        self.tendonLinearOnsetLength = float(self.conf.parameterSet('tendonLinearOnsetLength:' + pool, pool, 0))
        ##
        self.tendonCurvatureConstant = float(self.conf.parameterSet('tendonCurvatureConstant:' + pool, pool, 0))
        ##
        self.optimalTendonLength = float(self.conf.parameterSet('optimalTendonLength:' + pool, pool, 0))

        ##  
        self.lengthNorm = 0
        ##  
        self.velocityNorm = 0
        ##  
        self.accelerationNorm = 0
        ##  
        self.tendonLengthNorm = 0
        ##
        self.forceNorm = 0
        ##
        self.tendonForceNorm = 0
        
        ##  
        self.b_TypeI = float(self.conf.parameterSet('bTypeI:' + pool, pool, 0))
        ##  
        self.b_TypeII = float(self.conf.parameterSet('bTypeII:' + pool, pool, 0))
        ##  
        self.p_TypeI = float(self.conf.parameterSet('pTypeI:' + pool, pool, 0))
        ##  
        self.p_TypeII = float(self.conf.parameterSet('pTypeII:' + pool, pool, 0))
        ##  
        self.w_TypeI = float(self.conf.parameterSet('wTypeI:' + pool, pool, 0))
        ##  
        self.w_TypeII = float(self.conf.parameterSet('wTypeII:' + pool, pool, 0))
        ##
        self.d_TypeI = float(self.conf.parameterSet('dTypeI:' + pool, pool, 0))
        ##  
        self.d_TypeII = float(self.conf.parameterSet('dTypeII:' + pool, pool, 0))
        ##
        self.a0_TypeI = float(self.conf.parameterSet('a0TypeI:' + pool, pool, 0))
        ##  
        self.a0_TypeII = float(self.conf.parameterSet('a0TypeII:' + pool, pool, 0))       
        ##
        self.a1_TypeI = float(self.conf.parameterSet('a1TypeI:' + pool, pool, 0))
        ##  
        self.a1_TypeII = float(self.conf.parameterSet('a1TypeII:' + pool, pool, 0))
        ##
        self.a2_TypeI = float(self.conf.parameterSet('a2TypeI:' + pool, pool, 0))
        ##  
        self.a2_TypeII = float(self.conf.parameterSet('a2TypeII:' + pool, pool, 0))
        ##
        self.c0_TypeI = float(self.conf.parameterSet('c0TypeI:' + pool, pool, 0))
        ##  
        self.c0_TypeII = float(self.conf.parameterSet('c0TypeII:' + pool, pool, 0))       
        ##
        self.c1_TypeI = float(self.conf.parameterSet('c1TypeI:' + pool, pool, 0))
        ##  
        self.c1_TypeII = float(self.conf.parameterSet('c1TypeII:' + pool, pool, 0))
        ##
        self.Vmax_TypeI = float(self.conf.parameterSet('VmaxTypeI:' + pool, pool, 0))
        ##  
        self.Vmax_TypeII = float(self.conf.parameterSet('VmaxTypeII:' + pool, pool, 0))
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
        
        print 'Hill muscle of the ' + pool + ' muscle with maximum force of ' + str(self.maximumForce_N) + ' N  built.'
                                     
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
        self.atualizeActivation(activation_Sat)
        self.lengthNorm = self.length_m[self.timeIndex] / self.optimalLength_m
        self.velocityNorm = self.velocity_m_ms[self.timeIndex] / self.optimalLength_m
        self.accelerationNorm = self.acceleration_m_ms2[self.timeIndex] / self.optimalLength_m
        self.pennationAngle_rad[self.timeIndex] = self.computePennationAngle()
        self.tendonLength_m[self.timeIndex] = (self.musculoTendonLength_m[self.timeIndex] - 
                                               self.length_m[self.timeIndex] * math.cos(self.pennationAngle_rad[self.timeIndex]))
        self.tendonLengthNorm = self.tendonLength_m[self.timeIndex] / self.optimalTendonLength

        self.atualizeMuscleForce()
        self.atualizeTendonForce()

        self.force[self.timeIndex] = self.forceNorm * self.maximumForce_N
        self.tendonForce_N[self.timeIndex] = self.tendonForceNorm * self.maximumForce_N

        self.atualizeLenghtsAndVelocity()        
        
        self.timeIndex += 1

    def atualizeActivation(self, activation_Sat):
        '''
        '''
        self.activationTypeI[self.timeIndex] = (np.sum(activation_Sat[0:self.MUtypeInumber:1] * 
            self.twitchAmp_N[0:self.MUtypeInumber:1] * self.twTet[0:self.MUtypeInumber:1]) / 
            self.maximumActivationForce)
        self.activationTypeII[self.timeIndex] = (np.sum(activation_Sat[self.MUtypeInumber:-1:1] * 
            self.twitchAmp_N[self.MUtypeInumber:-1:1] * self.twTet[self.MUtypeInumber:-1:1]) / 
            self.maximumActivationForce)

    def computePennationAngle(self):
        '''
        '''
        return math.asin(self.pennationAngleAtOptimalLengthSin / self.lengthNorm)

    def computeForceLengthTypeI(self):
        '''
        '''
        return math.exp(-(math.fabs(self.lengthNorm ** self.b_TypeI - 1) / self.w_TypeI) ** self.p_TypeI)

    def computeForceLengthTypeII(self):
        '''
        '''    
        return math.exp(-(math.fabs(self.lengthNorm ** self.b_TypeII - 1) / self.w_TypeII) ** self.p_TypeII)

    def computeForceVelocityTypeI(self):
        '''
        '''
        if self.velocityNorm > 0.1:
            fv = (self.d_TypeI - (self.a0_TypeI+self.a1_TypeI * self.lengthNorm + self.a2_TypeI * self.lengthNorm ** 2)) / (self.d_TypeI + self.velocityNorm)
        else:
            fv = (self.Vmax_TypeI - self.velocityNorm) / (self.Vmax_TypeI + self.velocityNorm * (self.c0_TypeI + self.c1_TypeI * self.lengthNorm))
        return fv

    def computeForceVelocityTypeII(self):
        '''
        '''
        if self.velocityNorm > 0.1:
            fv = (self.d_TypeII - (self.a0_TypeII + self.a1_TypeII * self.lengthNorm + self.a2_TypeII * self.lengthNorm**2)) / (self.d_TypeII + self.velocityNorm)
        else:
            fv = (self.Vmax_TypeII - self.velocityNorm) / (self.Vmax_TypeII + self.velocityNorm * (self.c0_TypeII + self.c1_TypeII * self.lengthNorm))
        return fv
    
    def computeAcceleration(self):
        '''
        '''
        self.acceleration_m_ms2[self.timeIndex+1] =  ((self.tendonForce_N[self.timeIndex] - 
            self.force[self.timeIndex] * math.cos(self.pennationAngle_rad[self.timeIndex])) / 
            (self.mass * math.cos(self.pennationAngle_rad[self.timeIndex])) / 1000000)

        return self.acceleration_m_ms2[self.timeIndex+1]

    def dLdt(self):
        '''
        '''
        return  np.array([self.velocity_m_ms[self.timeIndex], self.computeAcceleration()])

    def atualizeMuscleForce(self):
        '''
        '''
        self.forceNorm  = (self.computeElasticElementForce() + self.computeViscousElementForce() + 
            self.computeTypeIActiveForce() + self.computeTypeIIActiveForce())

    def atualizeTendonForce(self):
        '''
        '''
        self.tendonForceNorm = (self.tendonCurvatureConstant * self.tendonElasticity * 
                math.log(math.exp((self.tendonLengthNorm - self.tendonLinearOnsetLength) / self.tendonCurvatureConstant) + 1))

    def computeElasticElementForce(self):
        '''
        '''
        return math.exp(self.elasticity / self.strain * (self.lengthNorm - self.strain - 1))

    def computeViscousElementForce(self):
        '''
        '''
        return self.viscosity * self.velocityNorm

    def computeTypeIActiveForce(self):
        '''
        '''
        return (self.activationTypeI[self.timeIndex] * self.computeForceLengthTypeI() * 
                    self.computeForceVelocityTypeI()) 

    def computeTypeIIActiveForce(self):
        '''
        '''
        return (self.activationTypeII[self.timeIndex] * self.computeForceLengthTypeII() * 
                    self.computeForceVelocityTypeII())

    def atualizeLenghtsAndVelocity(self):
        '''
        '''
        [self.length_m[self.timeIndex + 1],self.velocity_m_ms[self.timeIndex + 1]] = ([self.length_m[self.timeIndex],self.velocity_m_ms[self.timeIndex]] + 
                                                                                      self.conf.timeStep_ms * self.dLdt())
    
    def atualizeMusculoTendonLength(self, ankleAngle):
        '''
        '''
        self.musculoTendonLength_m[self.timeIndex] = (self.m0 + self.m1 * ankleAngle + self.m2 * (ankleAngle ** 2) + 
                                                      self.m3 * (ankleAngle ** 3) + self.m4 * (ankleAngle ** 4))

    def atualizeMomentArm(self, ankleAngle):
        '''
        '''
        self.momentArm_m[self.timeIndex] = (self.n0 + self.n1 * ankleAngle + self.n2 * (ankleAngle ** 2) + 
                                                      self.n3 * (ankleAngle ** 3) + self.n4 * (ankleAngle ** 4))

    def reset(self):
        '''

        '''
        self.timeIndex = 0
        self.tendonForce_N = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        self.contractileForce_N = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        self.elasticForce_N = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        self.viscousForce_N = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        self.length_m = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        self.velocity_m_ms = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        self.tendonLength_m = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        self.pennationAngle_rad = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        self.activationTypeI = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        self.activationTypeII = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        self.musculoTendonLength_m = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        self.momentArm_m = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        self.optimalLength_m = float(self.conf.parameterSet('optimalMuscleLength:' + pool, pool, 0))
        self.lengthNorm = 0
        self.velocityNorm = 0
        self.tendonLengthNorm = 0
        self.forceNorm = 0
        self.tendonForceNorm = 0