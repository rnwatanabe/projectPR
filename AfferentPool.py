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

import numpy as np
from AfferentUnit import AfferentUnit
from scipy.sparse import lil_matrix
from numpy import random

class AfferentPool(object):
    '''
    Class that implements an afferent pool. Encompasses a set of axons.
    '''

    def __init__(self, conf, pool, muscle):
        '''
        Constructor

        - Inputs:
            + **conf**: Configuration object with the simulation parameters.

            + **pool**: string with Motor unit pool to which the motor unit belongs.
        '''

        ## Indicates that is Motor Unit pool.
        self.kind = 'AF'

        ## Configuration object with the simulation parameters.
        self.conf = conf
        ## String with Motor unit pool to which the motor unit belongs.
        self.pool = pool
        
        self.muscle = muscle
        self.AFnumber = int(conf.parameterSet('Number_' + pool + '-' + muscle, pool, 0))
                
        ## Dictionary of Axon objects.
        self.unit = dict()

        for i in xrange(0, self.AFnumber):
            self.unit[i] = AfferentUnit(conf, pool, muscle, i)

        ## Vector with the instants of spikes in the last dynamical compartment, in ms.
        self.poolLastCompSpikes = np.array([])
        ## Vector with the instants of spikes in the terminal, in ms.
        self.poolTerminalSpikes = np.array([])

        ##
        print 'Afferent Pool ' + pool + ' of muscle ' + muscle + ' built'

    def atualizeAfferentPool(self, t, proprioceptorFR):
        '''
        Update all parts of the Motor Unit pool. It consists
        to update all motor units, the activation signal and
        the muscle force.

        - Inputs:
            + **t**: current instant, in ms.

            + **proprioceptorFR**: proprioceptor firing rate, in Hz.
        '''

        units = self.unit
        for i in xrange(self.AFnumber): 
            units[i].atualizeAfferentUnit(t, max(0, (proprioceptorFR - 
                                                     units[i].frequencyThreshold_Hz)*self.conf.timeStep_ms/1000.0))

    def listSpikes(self):
        '''
        List the spikes that occurred in the soma and in
        the terminal of the different motor units.
        '''
        for i in xrange(0,self.AFnumber):
            if i == 0:
                lastCompSpikeTrain = np.array(self.unit[i].lastCompSpikeTrain)
                terminalSpikeTrain = np.array(self.unit[i].terminalSpikeTrain)
            else:
                lastCompSpikeTrain = np.append(lastCompSpikeTrain, np.array(self.unit[i].lastCompSpikeTrain))
                terminalSpikeTrain = np.append(terminalSpikeTrain, np.array(self.unit[i].terminalSpikeTrain))
                
        self.poolLastCompSpikes = np.reshape(lastCompSpikeTrain, (-1, 2))
        self.poolTerminalSpikes = np.reshape(terminalSpikeTrain, (-1, 2))

    def reset(self):
        '''

        '''
        self.poolLastCompSpikes = np.array([])
        self.poolTerminalSpikes = np.array([])
        for i in xrange(self.AFnumber): self.unit[i].reset()
