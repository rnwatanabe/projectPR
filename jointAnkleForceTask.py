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


class jointAnkleForceTask(object):

    def __init__(self, conf, pools):
        
        self.conf = conf
        self.muscles = []

        for i in pools:
            if i.pool == 'SOL' or i.pool == 'MG' or i.pool == 'LG' or i.pool == 'TA':
                self.muscles.append(i)



        ##
        self.ankleAngle_rad = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)

    def atualizeAnkle(self, t, ankleAngle):
        self.atualizeAngle(t, ankleAngle)
        for muscle in self.muscles:
            muscle.Muscle.atualizeMusculoTendonLength(ankleAngle)
            muscle.Muscle.atualizeMomentArm(ankleAngle)

    def atualizeAngle(self, t, ankleAngle):
        '''
        '''
        
        self.ankleAngle_rad[int(np.rint(t / self.conf.timeStep_ms))] = ankleAngle
        


        
        
    