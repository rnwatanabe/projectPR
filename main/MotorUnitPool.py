'''
Created on Oct 8, 2015

@author: root
'''

import numpy as np
from MotorUnit import MotorUnit
import math
from scipy.sparse import lil_matrix

 

def twitchSaturation(force, b):
    return 2.0 / (1 + np.exp(-b * force)) - 1

 
def computeForce(force, TwAmp, TwTet):
    return np.sum(force * TwAmp * TwTet)   


class MotorUnitPool(object):
    '''
    classdocs
    '''


    def __init__(self, conf, pool):
        '''
        Constructor
        '''
        self.kind = 'MU'
        self.conf = conf
        self.pool = pool
        MUnumber_S = int(conf.parameterSet('MUnumber_S_' + pool, pool, 0))
        MUnumber_FR = int(conf.parameterSet('MUnumber_FR_' + pool, pool, 0))
        MUnumber_FF = int(conf.parameterSet('MUnumber_FF_' + pool, pool, 0))  
        self.MUnumber = MUnumber_S + MUnumber_FR + MUnumber_FF
        
        self.unit = []
        
        
        for i in xrange(0, self.MUnumber): 
            if i < MUnumber_S:
                self.unit.append(MotorUnit(conf, pool, i, 'S'))
            elif i < MUnumber_S + MUnumber_FR:
                self.unit.append(MotorUnit(conf, pool, i, 'FR'))
            else:
                self.unit.append(MotorUnit(conf, pool, i, 'FF'))
            
        self.poolSomaSpikes = np.array([])    
        self.poolTerminalSpikes = np.array([])
        
        ##activation signal
        
        self.activationModel = conf.parameterSet('activationModel', pool, 0)
        
        
        
        if (self.activationModel == 'SOCDS'):
            self.ActMatrix = lil_matrix((self.MUnumber, 3*self.MUnumber), dtype = float)
            
            for i in xrange(0, self.MUnumber):
                
                self.ActMatrix[i,3*i:3*i+3] = [2*math.exp(-conf.timeStep_ms/self.unit[i].TwitchTc_ms), 
                                    -math.exp(-2*conf.timeStep_ms/self.unit[i].TwitchTc_ms), 
                                    math.pow(conf.timeStep_ms, 2.0)/self.unit[i].TwitchTc_ms*math.exp(1.0-conf.timeStep_ms/self.unit[i].TwitchTc_ms)]
             
            self.ActMatrix.tocsr()   
        
            self.an = np.zeros((3*self.MUnumber, 1), dtype = float)
              
        self.activation_nonSat = np.zeros((self.MUnumber, 1), dtype = float)
        self.bSat = np.zeros((self.MUnumber, 1), dtype = float)
        self.twTet = np.zeros((self.MUnumber, 1), dtype = float)
        self.twitchAmp_N = np.zeros((self.MUnumber, 1), dtype = float)
        for i in xrange(0, self.MUnumber):
            self.bSat[i] = self.unit[i].bSat
            self.twitchAmp_N[i] = self.unit[i].TwitchAmp_N
            self.twTet[i] = self.unit[i].twTet  
        
        self.activation_Sat = np.zeros((self.MUnumber, 1), dtype = float)    
        
        self.diracDeltaValue = 1.0 / conf.timeStep_ms
        
        #Force
        self.force = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        self.hillModel = conf.parameterSet('hillModel',pool, 0)
        if self.hillModel == 'No': self.atualizeForce = self.atualizeForceNoHill
        
        self.timeIndex = 0;
        
        
        ##
        print 'Motor Unit Pool ' + pool + ' built'
        
    def atualizeMotorUnitPool(self, t):
        for i in self.unit: i.atualizeMotorUnit(t)
        self.atualizeActivationSignal(t)
        self.atualizeForce()
        self.timeIndex += 1
    
        
    def atualizeActivationSignal(self, t):
        for i in xrange(self.MUnumber):
            self.an.itemset(3*i+1, self.an.item(3*i))
            self.an.itemset(3*i, self.activation_nonSat.item(i))            
            if self.unit[i].terminalSpikeTrain and abs(t - self.conf.timeStep_ms - self.unit[i].terminalSpikeTrain[-1][0]) < 1e-6: 
                self.an.itemset(3*i+2, self.diracDeltaValue)
            else: self.an.itemset(3*i+2,  0.0)
        
        self.activation_nonSat = self.ActMatrix.dot(self.an)        
        self.activation_Sat = twitchSaturation(self.activation_nonSat, self.bSat)
        
    
    def atualizeForceNoHill(self):
        
        self.force[self.timeIndex] = computeForce(self.activation_Sat, self.twitchAmp_N, self.twTet)            
        
        
           
    def listSpikes(self):
        
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
        
    
        
        
    
        
          