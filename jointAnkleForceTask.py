'''
    Neuromuscular simulator in Python.
    Copyright (C) 2017  Renato Naville Watanabe

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

    Contact: renato.watanabe@ufabc.edu.br
'''
import math
import numpy as np


class jointAnkleForceTask(object):

    def __init__(self, conf, pools):
        
        self.conf = conf
        self.muscles = []

        for i in xrange(0,len(pools)):
            if pools[i].pool == 'SOL' or pools[i].pool == 'MG' or pools[i].pool == 'LG' or pools[i].pool == 'TA':
                self.muscles.append(pools[i])

        ##
        self.ankleAngle_rad = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        self.ankleTorque_Nm = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)

        print 'Ankle joint for Force Task built'
    def atualizeAnkle(self, t, ankleAngle):
        '''
        Update the ankle joint.
        
        - Inputs:
            + **t**: current instant, in ms.

            + **ankleAngle**: ankle angle, in rad. 
        '''
        self.atualizeAngle(t, ankleAngle)
        for muscle in self.muscles:
            muscle.Muscle.atualizeMusculoTendonLength(ankleAngle)
            muscle.Muscle.atualizeMomentArm(ankleAngle)

    def atualizeAngle(self, t, ankleAngle):
        '''
        '''
        
        self.ankleAngle_rad[int(np.rint(t / self.conf.timeStep_ms))] = ankleAngle

    def computeTorque(self, t):
        '''
        '''
        torque = 0
        for muscle in self.muscles:
            torque += muscle.Muscle.force[int(np.rint(t / self.conf.timeStep_ms))] * muscle.Muscle.momentArm_m[int(np.rint(t / self.conf.timeStep_ms))]
        velocity = (self.ankleAngle_rad[int(np.rint(t / self.conf.timeStep_ms))] - 
                    self.ankleAngle_rad[int(np.rint(t / self.conf.timeStep_ms)) - 1]) / self.conf.timeStep_ms
        acceleration = (self.ankleAngle_rad[int(np.rint(t / self.conf.timeStep_ms))] - 
                    2*self.ankleAngle_rad[int(np.rint(t / self.conf.timeStep_ms)) - 1]+
                    self.ankleAngle_rad[int(np.rint(t / self.conf.timeStep_ms)) - 2]) / (self.conf.timeStep_ms**2)
        
<<<<<<< HEAD
        torque -= 0*1100*velocity + 0*320*self.ankleAngle_rad[int(np.rint(t / self.conf.timeStep_ms))] + 0*acceleration
=======
        torque -= 0*1100*velocity + 0*320*self.ankleAngle_rad[int(np.rint(t / self.conf.timeStep_ms))] + 0*7455*acceleration + 2.01*0.055*math.sin(self.ankleAngle_rad[int(np.rint(t / self.conf.timeStep_ms))])
>>>>>>> pabloAbur-parPool
        
        self.ankleTorque_Nm[int(np.rint(t / self.conf.timeStep_ms))] = torque
        
    def reset(self, t):
        '''
        '''
        self.ankleAngle_rad = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        self.ankleTorque_Nm = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)

        
        
    