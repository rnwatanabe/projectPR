'''
    Neuromuscular simulator in Python.
    Copyright (C) 2018  Renato Naville Watanabe
                        Pablo Alejandro

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
from InterneuronNoChannel import InterneuronNoChannel
from scipy.sparse import lil_matrix

 

def runge_kutta(derivativeFunction,t, x, timeStep, timeStepByTwo, timeStepBySix):
    k1 = derivativeFunction(t, x)
    k2 = derivativeFunction(t + timeStepByTwo, x + timeStepByTwo * k1)
    k3 = derivativeFunction(t + timeStepByTwo, x + timeStepByTwo * k2)
    k4 = derivativeFunction(t + timeStep, x + timeStep * k3)
    
    return x + timeStepBySix * (np.add(np.add(np.add(k1, k2, order = 'C'), np.add(k2, k3, order='C')), np.add(k3, k4, order='C'), order='C'))

class InterneuronPoolNoChannel(object):
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
            self.unit[i] = InterneuronNoChannel(conf, self.pool, i)

        ## Vector with the instants of spikes in the soma compartment, in ms.
        self.poolSomaSpikes = np.array([])
        ##

        # This is used to get values from Interneuron.py and make computations
        # in InterneuronPool.py
        # TODO create it all here instead?
        self.totalNumberOfCompartments = 0

        for i in xrange(self.Nnumber):
            self.totalNumberOfCompartments = self.totalNumberOfCompartments \
                + self.unit[i].compNumber

        self.v_mV = np.zeros((self.totalNumberOfCompartments),
                             dtype = np.double)
             
        self.G = lil_matrix((self.totalNumberOfCompartments,
                          self.totalNumberOfCompartments), dtype = float)
        self.iInjected = np.zeros_like(self.v_mV, dtype = 'd')
        self.capacitanceInv = np.zeros_like(self.v_mV, dtype = 'd')
        self.iIonic = np.full_like(self.v_mV, 0.0)
        self.EqCurrent_nA = np.zeros_like(self.v_mV, dtype = 'd')

        # Retrieving data from Interneuron class
        for i in xrange(self.Nnumber):
            self.v_mV[i*self.unit[i].compNumber:i*self.unit[i].compNumber \
                    +self.unit[i].v_mV.shape[0]] = self.unit[i].v_mV
            # With only one compartment, it is a diagonal matrix
            self.G[i,i] = self.unit[i].G
            self.capacitanceInv[i*self.unit[i].compNumber: \
                    i*self.unit[i].compNumber \
                    +self.unit[i].capacitanceInv.shape[0]] \
                    = self.unit[i].capacitanceInv
            self.EqCurrent_nA[i*self.unit[i].compNumber: \
                    i*self.unit[i].compNumber \
                    +self.unit[i].EqCurrent_nA.shape[0]] \
                    = self.unit[i].EqCurrent_nA


        print 'Interneuron Pool of ' + pool + ' ' + group + ' built'

    def atualizeInterneuronPool(self, t):
        '''
        Update all parts of the Motor Unit pool. It consists
        to update all motor units, the activation signal and
        the muscle force.

        - Inputs:
            + **t**: current instant, in ms.

        '''
        
        np.clip(runge_kutta(self.dVdt, t, self.v_mV, self.conf.timeStep_ms,
                            self.conf.timeStepByTwo_ms,
                            self.conf.timeStepBySix_ms),
                            -30.0, 120.0, self.v_mV)
        
        for i in xrange(self.Nnumber):
            self.unit[i].atualizeInterneuron(t, self.v_mV[i*self.unit[i].compNumber:(i+1)*self.unit[i].compNumber])

    def dVdt(self, t, V): 
        #k = 0
        for i in xrange(self.Nnumber):
            for j in xrange(self.unit[i].compNumber):
                self.iIonic.itemset(i*self.unit[0].compNumber+j,
                                    self.unit[i].compartment[j].computeCurrent(t,
                                                                               V.item(i*self.unit[0].compNumber+j)))
                #k += 1
        return (self.iIonic + self.G.dot(V) + self.iInjected
                + self.EqCurrent_nA) * self.capacitanceInv
        '''
        self.GPU.csrmv('N', self.m, self.n, self.nnz,  1.0, self.descr, self.csrVal, self.csrRowPtr, self.csrColInd, V, 0.0, self.dVdtValue)              
        
        return (self.iIonic + self.dVdtValue + self.iInjected
                + self.EqCurrent_nA) * self.capacitanceInv
        '''

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
        

          
