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
from MotorUnit import MotorUnit
from MuscularActivation import MuscularActivation
from MuscleNoHill import MuscleNoHill
from MuscleHill import MuscleHill
from MuscleSpindle import MuscleSpindle
from scipy.sparse import lil_matrix
import pathos.pools as pp

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
        self.t = 0

        self.pool1 = pp.ProcessPool(4)
        ## Indicates that is Motor Unit pool.
        self.kind = 'MU'

        ## Configuration object with the simulation parameters.
        self.conf = conf
        ## String with Motor unit pool to which the motor unit belongs.
        self.pool = pool
        MUnumber_S = int(conf.parameterSet('MUnumber_' + pool + '-S', pool, 0))
        MUnumber_FR = int(conf.parameterSet('MUnumber_' + pool + '-FR', pool, 0))
        MUnumber_FF = int(conf.parameterSet('MUnumber_' + pool + '-FF', pool, 0))
        ## Number of motor units.
        self.MUnumber = MUnumber_S + MUnumber_FR + MUnumber_FF
        ## Muscle thickness, in mm.
        self.muscleThickness_mm = float(self.conf.parameterSet('thickness:' + pool, pool, 0))

        ## Dictionary of MotorUnit objects.
        self.unit = dict()
        
        
        for i in xrange(0, self.MUnumber): 
            if i < MUnumber_S:
                self.unit[i] = MotorUnit(conf, pool, i, 'S', self.muscleThickness_mm, conf.skinThickness_mm)
            elif i < MUnumber_S + MUnumber_FR:
                self.unit[i] = MotorUnit(conf, pool, i, 'FR', self.muscleThickness_mm, conf.skinThickness_mm)
            else:
                self.unit[i] = MotorUnit(conf, pool, i, 'FF', self.muscleThickness_mm, conf.skinThickness_mm)

        ## Vector with the instants of spikes in the soma compartment, in ms.            
        self.poolSomaSpikes = np.array([])
        ## Vector with the instants of spikes in the last dynamical compartment, in ms.
        self.poolLastCompSpikes = np.array([])    
        ## Vector with the instants of spikes in the terminal, in ms.
        self.poolTerminalSpikes = np.array([])
        
        #activation signal
        self.Activation = MuscularActivation(self.conf,self.pool, self.MUnumber,self.unit)
        
        #Force
        ## String indicating whther a Hill model is used or not. For now, it can be *No*.
        self.hillModel = conf.parameterSet('hillModel', pool, 0)
        if self.hillModel == 'No': 
            self.Muscle = MuscleNoHill(self.conf, self.pool, self.MUnumber, MUnumber_S, self.unit)
        else:
            self.Muscle = MuscleHill(self.conf, self.pool, self.MUnumber, MUnumber_S, self.unit)
        
        # EMG 
        ## EMG along time, in mV.
        self.emg = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        
        # Spindle
        self.spindle = MuscleSpindle(self.conf, self.pool)


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
        self.t = t
        units = self.unit
        
        #for i in xrange(self.MUnumber): units[i].atualizeMotorUnit(t)
        self.pool1.map(self.atualizeUnit, range(self.MUnumber))
        self.Activation.atualizeActivationSignal(t, units)
        self.Muscle.atualizeForce(self.Activation.activation_Sat)
        self.spindle.atualizeMuscleSpindle(t, self.Muscle.lengthNorm,
                                           self.Muscle.velocityNorm, 
                                           self.Muscle.accelerationNorm, 
                                           31, 33)

    def atualizeUnit(self,i):
        
        self.unit[i].atualizeMotorUnit(self.t)      

    def listSpikes(self):
        '''
        List the spikes that occurred in the soma and in
        the terminal of the different motor units.
        '''
        for i in xrange(0,self.MUnumber):
            if i == 0:
                somaSpikeTrain = np.array(self.unit[i].somaSpikeTrain)
                lastCompSpikeTrain = np.array(self.unit[i].lastCompSpikeTrain)
                terminalSpikeTrain = np.array(self.unit[i].terminalSpikeTrain)
            else:
                somaSpikeTrain = np.append(somaSpikeTrain, np.array(self.unit[i].somaSpikeTrain))
                lastCompSpikeTrain = np.append(lastCompSpikeTrain, np.array(self.unit[i].lastCompSpikeTrain))
                terminalSpikeTrain = np.append(terminalSpikeTrain, np.array(self.unit[i].terminalSpikeTrain))
                
        self.poolSomaSpikes = np.reshape(somaSpikeTrain, (-1, 2))
        self.poolLastCompSpikes = np.reshape(lastCompSpikeTrain, (-1, 2))
        self.poolTerminalSpikes = np.reshape(terminalSpikeTrain, (-1, 2))

    def getMotorUnitPoolInstantEMG(self, t):
        '''

        '''
        emg = 0
        for i in xrange(self.MUnumber): emg += self.unit[i].getEMG(t)

        return emg

    def getMotorUnitPoolEMG(self):
        '''

        '''
        for i in xrange(0, len(self.emg)):
            self.emg[i] = self.getMotorUnitPoolInstantEMG(i * self.conf.timeStep_ms)


    def reset(self):
        '''

        '''

                   
        self.poolSomaSpikes = np.array([])
        self.poolLastCompSpikes = np.array([])    
        self.poolTerminalSpikes = np.array([])
        self.emg = np.zeros((int(np.rint(self.conf.simDuration_ms/self.conf.timeStep_ms)), 1), dtype=float)

        for i in xrange(self.MUnumber): self.unit[i].reset()
        self.Activation.reset()
        self.Muscle.reset()