'''
Created on Jul, 28 2015

@author: root
'''

import numpy as np
from Interneuron import Interneuron
from scipy.sparse import lil_matrix

 

class InterneuronPool(object):
    '''
    Class that implements a motor unit pool. Encompasses a set of motor
    units that controls a single  muscle.
    '''


    def __init__(self, conf, pool):
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
        self.pool = pool
        ## Number of Neurons.
        self.Nnumber = int(conf.parameterSet('number_' + pool, pool, 0))

        ## List of Interneuron objects.
        self.unit = []

        for i in xrange(0, self.Nnumber):
            self.unit.append(Interneuron(conf, pool, i))

        ## Vector with the instants of spikes in the soma compartment, in ms.
        self.poolSomaSpikes = np.array([])
        ##
        print 'Interneuron Pool of ' + pool + ' built'

    def atualizeInterneuronPool(self, t):
        '''
        Update all parts of the Motor Unit pool. It consists
        to update all motor units, the activation signal and
        the muscle force.

        - Inputs:
            + **t**: current instant, in ms.

        '''
        
        for i in self.unit: i.atualizeInterneuron(t)

    def listSpikes(self):
        '''
        List the spikes that occurred in the soma and in
        the terminal of the different motor units.
        '''
        for i in xrange(0,self.Nnumber):
            if i == 0:
                somaSpikeTrain = np.array(self.unit[i].somaSpikeTrain)
            else:
                somaSpikeTrain = np.append(somaSpikeTrain, np.array(self.unit[i].somaSpikeTrain))
        self.poolSomaSpikes = somaSpikeTrain
                    
        self.poolSomaSpikes = np.reshape(self.poolSomaSpikes, (-1, 2))
        

          