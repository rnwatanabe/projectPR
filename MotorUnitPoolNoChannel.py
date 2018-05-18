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
from MotorUnitNoChannel import MotorUnitNoChannel
from MuscularActivation import MuscularActivation
from MuscleNoHill import MuscleNoHill
from MuscleHill import MuscleHill
from MuscleSpindle import MuscleSpindle
from scipy.sparse import lil_matrix
#import pyculib.sparse as pcu
import time
#from numba import jit, prange

def SpMV_viaMKL( A, x, numberOfBlocks, sizeOfBlock ):
    '''
    Wrapper to Intel's SpMV
    (Sparse Matrix-Vector multiply)
    For medium-sized matrices, this is 4x faster
    than scipy's default implementation
    Stephen Becker, April 24 2014
    stephen.beckr@gmail.com
    '''

    import numpy as np
    import scipy.sparse as sparse
    from ctypes import POINTER,c_void_p,c_int,c_char,c_double,byref,cdll
    mkl = cdll.LoadLibrary("libmkl_rt.so")

    SpMV = mkl.mkl_cspblas_dbsrgemv
    # Dissecting the "cspblas_dcsrgemv" name:
    # "c" - for "c-blas" like interface (as opposed to fortran)
    #    Also means expects sparse arrays to use 0-based indexing, which python does
    # "sp"  for sparse
    # "d"   for double-precision
    # "csr" for compressed row format
    # "ge"  for "general", e.g., the matrix has no special structure such as symmetry
    # "mv"  for "matrix-vector" multiply

    

    # The data of the matrix
    data    = A.data.ctypes.data_as(POINTER(c_double))
    indptr  = A.indptr.ctypes.data_as(POINTER(c_int))
    indices = A.indices.ctypes.data_as(POINTER(c_int))

    # Allocate output, using same conventions as input

    
    y = np.empty(numberOfBlocks*sizeOfBlock,dtype=np.double,order='F')  

    np_x = x.ctypes.data_as(POINTER(c_double))
    np_y = y.ctypes.data_as(POINTER(c_double))
    # now call MKL. This returns the answer in np_y, which links to y
    SpMV(byref(c_char("N")), byref(c_int(numberOfBlocks)), byref(c_int(sizeOfBlock)), data ,indptr, indices, np_x, np_y ) 

    return y

def runge_kutta(derivativeFunction,t, x, timeStep, timeStepByTwo, timeStepBySix):
    k1 = derivativeFunction(t, x)
    k2 = derivativeFunction(t + timeStepByTwo, x + timeStepByTwo * k1)
    k3 = derivativeFunction(t + timeStepByTwo, x + timeStepByTwo * k2)
    k4 = derivativeFunction(t + timeStep, x + timeStep * k3)
    
    return x + timeStepBySix * (np.add(np.add(np.add(k1, k2, order = 'C'), np.add(k2, k3, order='C')), np.add(k3, k4, order='C'), order='C'))

class MotorUnitPoolNoChannel(object):
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
                self.unit[i] = MotorUnitNoChannel(conf, pool, i, 'S', self.muscleThickness_mm, conf.skinThickness_mm)
            elif i < MUnumber_S + MUnumber_FR:
                self.unit[i] = MotorUnitNoChannel(conf, pool, i, 'FR', self.muscleThickness_mm, conf.skinThickness_mm)
            else:
                self.unit[i] = MotorUnitNoChannel(conf, pool, i, 'FF', self.muscleThickness_mm, conf.skinThickness_mm)

        # This is used to get values from MotorUnit.py and make computations
        # in MotorUnitPool.py
        # TODO create it all here instead?
        self.totalNumberOfCompartments = 0

        for i in xrange(self.MUnumber):
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

        # Retrieving data from Motorneuron class
        # Vectors or matrices from Motorneuron compartments are copied,
        # populating larger vectors or matrices that will be used for computations
        for i in xrange(self.MUnumber):
            self.v_mV[i*self.unit[i].compNumber:i*self.unit[i].compNumber \
                    +self.unit[i].v_mV.shape[0]] = self.unit[i].v_mV
            # Consists of smaller matrices on its diagonal
            self.G[i*self.unit[i].compNumber:i*self.unit[i].compNumber \
                    +self.unit[i].G.shape[0], \
                    i*self.unit[i].compNumber:i*self.unit[i].compNumber \
                    +self.unit[i].G.shape[1]] = self.unit[i].G
            self.capacitanceInv[i*self.unit[i].compNumber: \
                    i*self.unit[i].compNumber \
                    +self.unit[i].capacitanceInv.shape[0]] \
                    = self.unit[i].capacitanceInv
            self.EqCurrent_nA[i*self.unit[i].compNumber: \
                    i*self.unit[i].compNumber \
                    +self.unit[i].EqCurrent_nA.shape[0]] \
                    = self.unit[i].EqCurrent_nA
        self.sizeOfBlock = int(self.totalNumberOfCompartments/self.MUnumber)
        self.G = self.G.tobsr(blocksize=(self.sizeOfBlock, self.sizeOfBlock)) 
        '''
        self.G  = pcu.csr_matrix(self.G)
        self.GPU = pcu.Sparse(0)
        self.m, self.n = self.GGPU.shape
        self.nnz = self.GGPU.nnz
        self.descr = self.GPU.matdescr()
        self.csrVal = self.GGPU.data
        self.csrRowPtr = self.GGPU.indptr
        self.csrColInd = self.GGPU.indices
        self.dVdtValue = nhep.empty(self.totalNumberOfCompartments,dtype=np.double)  
        '''
        ## Vector with the instants of spikes in the soma compartment, in ms.            
        self.poolSomaSpikes = np.array([])
        ## Vector with the instants of spikes in the last dynamical compartment, in ms.
        self.poolLastCompSpikes = np.array([])    
        ## Vector with the instants of spikes in the terminal, in ms.
        self.poolTerminalSpikes = np.array([])
        
        #activation signal
        self.Activation = MuscularActivation(self.conf,self.pool, self.MUnumber,self.unit)
        
        #Force
        ## String indicating whther a Hill  model is used or not. For now, it can be *No*.
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

        np.clip(runge_kutta(self.dVdt, t, self.v_mV, self.conf.timeStep_ms,
                            self.conf.timeStepByTwo_ms,
                            self.conf.timeStepBySix_ms),
                            -30.0, 120.0, self.v_mV)
                            
        for i in xrange(self.MUnumber):
            self.unit[i].atualizeMotorUnit(t, self.v_mV[i*self.unit[i].compNumber:(i+1)*self.unit[i].compNumber])
        self.Activation.atualizeActivationSignal(t, self.unit)
        self.Muscle.atualizeForce(self.Activation.activation_Sat)
        self.spindle.atualizeMuscleSpindle(t, self.Muscle.lengthNorm,
                                           self.Muscle.velocityNorm, 
                                           self.Muscle.accelerationNorm, 
                                           31, 33)
    
    def dVdt(self, t, V): 
        
        for i in xrange(self.MUnumber):
            for j in xrange(self.unit[i].compNumber):
                self.iIonic.itemset(i*self.unit[0].compNumber+j,
                                    self.unit[i].compartment[j].computeCurrent(t,
                                                                               V.item(i*self.unit[0].compNumber+j)))
        return (self.iIonic + self.G.dot(V) + self.iInjected
                + self.EqCurrent_nA) * self.capacitanceInv
        
        
             
        #return (self.iIonic + SpMV_viaMKL(self.G,V,self.MUnumber, self.sizeOfBlock) + self.iInjected
        #        + self.EqCurrent_nA) * self.capacitanceInv
       

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
