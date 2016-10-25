'''
Created on Oct, 06 2016

@author: root
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
        


        
        
    