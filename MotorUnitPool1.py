'''
Created on Jul, 28 2015

@author: root
'''

import numpy as np
from MotorUnit import MotorUnit
from MuscularActivation import MuscularActivation
from MuscleNoHill import MuscleNoHill
from MuscleHill import MuscleHill
from scipy.sparse import lil_matrix

 

import dill
from mpi4py import MPI
MPI.pickle.dumps = dill.dumps
MPI.pickle.loads = dill.loads 
import copy
import itertools

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
        
        ## List of MotorUnit objects.
        self.unit = []
        
        
        for i in xrange(0, self.MUnumber): 
            if i < MUnumber_S:
                self.unit.append(MotorUnit(conf, pool, i, 'S'))
            elif i < MUnumber_S + MUnumber_FR:
                self.unit.append(MotorUnit(conf, pool, i, 'FR'))
            else:
                self.unit.append(MotorUnit(conf, pool, i, 'FF'))

        ## Vector with the instants of spikes in the soma compartment, in ms.            
        self.poolSomaSpikes = np.array([])    
        ## Vector with the instants of spikes in the terminal, in ms.
        self.poolTerminalSpikes = np.array([])
        
        #activation signal
        self.Activation = MuscularActivation(self.conf,self.pool, self.MUnumber,self.unit)
        
        #Force
        ## String indicating whther a Hill model is used or not. For now, it can be *No*.
        self.hillModel = conf.parameterSet('hillModel',pool, 0)
        if self.hillModel == 'No': 
            self.Muscle = MuscleNoHill(self.conf, self.pool, self.MUnumber, MUnumber_S, self.unit)
        else:
            self.Muscle = MuscleHill(self.conf, self.pool, self.MUnumber, MUnumber_S, self.unit)
        
        ##
        #print 'Motor Unit Pool ' + pool + ' built'

        # MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size ()
        self.rank = self.comm.Get_rank ()
        self.procSize = len (self.unit) / self.size
        self.unitSlice=[]
        
    def atualizeMotorUnitPool(self, t):
        '''
        Update all parts of the Motor Unit pool. It consists
        to update all motor units, the activation signal and
        the muscle force.

        - Inputs:
            + **t**: current instant, in ms.
        '''
        #### Primeira tentativa a falhar
        #self.unitSlice = copy.deepcopy(self.unit[self.rank * self.procSize:(self.rank + 1) * self.procSize])
        #for i in self.unitSlice: i.atualizeMotorUnit(t)
        #self.aux = self.comm.allgather (self.unitSlice)
        #self.unit = copy.deepcopy(list(itertools.chain.from_iterable(self.aux)))
        ####

        ####
        # Forma original de se atualizar as unidades
        # Forma original
        #for i in self.unit: i.atualizeMotorUnit(t)
        # Forma levemente alterada
        #for i in xrange(self.rank * self.procSize,(self.rank + 1) * self.procSize):
        #    self.unit[i].atualizeMotorUnit(t)
        ####

        #### Alteracoes principais
        # Cada processo usa um pedaco de self.unit
        self.unittest = []
        for i in xrange(self.rank * self.procSize,(self.rank + 1) * self.procSize):
            self.unitSlice.append(self.unit[i])
        for i in self.unitSlice: i.atualizeMotorUnit(t)
        # Fazer atualize MotorUnit normalmente e entao usar allgather reproduz
        # o erro
        self.aux = self.comm.allgather (self.unitSlice)
        # Flattening, pois allgather retorna lista de listas
        for i in xrange(len(self.aux)):
            for j in xrange(len(self.aux[i])):
                self.unittest.append((self.aux[i])[j])
        # alternativa ao metodo acima
        #self.unittest=list(itertools.chain.from_iterable(self.aux)) 

        # Para checar se sao iguais
        #if self.rank==0:
        #    print self.unittest[0]==self.unit[0]
        #    print vars(self.unittest[0])
        #    print vars(self.unit[0])

        # Retornando valores para self.unit
        self.unit=[]
        for i in xrange(len(self.unittest)):
            #self.unit.append(copy.copy(self.unittest[i])) # alternativa
            self.unit.append(self.unittest[i])

        # Checando os objetos synapses
        #if self.rank==0:
        #    print vars(self.unit[0].transmitSpikesThroughSynapses[0]), '1'
        #    print vars(self.unit[0].transmitSpikesThroughSynapses[1]), '2'
        
        self.unitSlice=[]
        ## O valor dessa variavel e sempre zero qnd o erro ocorre
        #print self.unit[0].iIonic
        ####

        self.Activation.atualizeActivationSignal(t, self.unit)
        self.Muscle.atualizeForce(self.Activation.activation_Sat)   

    def listSpikes(self):
        '''
        List the spikes that occurred in the soma and in
        the terminal of the different motor units.
        '''
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
        
    
        
        
    
        
          
