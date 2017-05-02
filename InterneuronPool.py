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
from Interneuron import Interneuron
from scipy.sparse import lil_matrix

 

class InterneuronPool(object):
    '''
    Class that implements a motor unit pool. Encompasses a set of motor
    units that controls a single  muscle.
    '''


    def __init__(self, conf, pool, group):
        '''
        Constructor

        - Inputs:
            + **conf**: Configuration object with the simulation parameters.

            + **pool**: string with Interneuron pool to which the motor unit belongs.
        '''

        ## Indicates that is Motor Unit pool.
        self.kind = 'IN'

        ## Configuration object with the simulation parameters.
        self.conf = conf
        ## String with Motor unit pool to which the motor unit belongs.
        self.pool = pool + '_' + group 
        ## Number of Neurons.
        self.Nnumber = int(conf.parameterSet('Number_' + self.pool, pool, 0))

        ## List of Interneuron objects.
        self.unit = dict()

        for i in xrange(0, self.Nnumber):
            self.unit[i] = Interneuron(conf, self.pool, i)

        ## Vector with the instants of spikes in the soma compartment, in ms.
        self.poolSomaSpikes = np.array([])
        ##
        print 'Interneuron Pool of ' + pool + ' ' + group + ' built'

    def atualizeInterneuronPool(self, t):
        '''
        Update all parts of the Motor Unit pool. It consists
        to update all motor units, the activation signal and
        the muscle force.

        - Inputs:
            + **t**: current instant, in ms.

        '''
        
        for i in xrange(len(self.unit)): self.unit[i].atualizeInterneuron(t)

    def listSpikes(self):
        '''
        List the spikes that occurred in the soma and in
        the terminal of the different motor units.
        '''
        for i in xrange(0,self.Nnumber):
            if i == 0: somaSpikeTrain = np.array(self.unit[i].somaSpikeTrain)
            else: somaSpikeTrain = np.append(somaSpikeTrain, np.array(self.unit[i].somaSpikeTrain))
        self.poolSomaSpikes = somaSpikeTrain
        self.poolSomaSpikes = np.reshape(self.poolSomaSpikes, (-1, 2))
        

          