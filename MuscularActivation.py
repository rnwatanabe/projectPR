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

    Contact: renato.watanabe@usp.br
'''

import numpy as np
import math
from scipy.sparse import lil_matrix
#from pyculib.sparse import csr_matrix

#from numba import jit

def SpMV_viaMKL( A, x):
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
     from ctypes import POINTER,c_void_p,c_int,c_char,c_double,byref,cdll, c_char_p
     mkl = cdll.LoadLibrary("libmkl_rt.so")

     SpMV = mkl.mkl_dcsrmv
     # Dissecting the "cspblas_dcsrgemv" name:
     # "c" - for "c-blas" like interface (as opposed to fortran)
     #    Also means expects sparse arrays to use 0-based indexing, which python does
     # "sp"  for sparse
     # "d"   for double-precision
     # "csr" for compressed row format
     # "ge"  for "general", e.g., the matrix has no special structure such as symmetry
     # "mv"  for "matrix-vector" multiply

     m,k = A.shape
        
     # The data of the matrix
     data    = A.data.ctypes.data_as(POINTER(c_double))
     indptb  = A.indptr.ctypes.data_as(POINTER(c_int))
     indpte  = (A.indptr+3).ctypes.data_as(POINTER(c_int))
     indices = A.indices.ctypes.data_as(POINTER(c_int))

     # Allocate output, using same conventions as input
     
     matdescra = 'GUNC'
     
     y = np.empty((m,1),dtype=np.double,order='F')  

     np_x = x.ctypes.data_as(POINTER(c_double))
     np_y = y.ctypes.data_as(POINTER(c_double))
     # now call MKL. This returns the answer in np_y, which links to y
     SpMV(byref(c_char("N")), byref(c_int(m)), byref(c_int(k)),byref(c_double(1.0)), byref(c_char_p(matdescra)), data ,indices, indptb, indpte, np_x, byref(c_double(0.0)), np_y) 

     return y


#@jit
def twitchSaturation(activationsat, b):
    '''
    Computes the muscle unit force after the nonlinear saturation. 

    \f{equation}{
        a_{sat} = \frac{1-e^{-b.a_{nSat}}}{1+e^{-b.a_{nSat}}}
    \f}

- Inputs:
    + **activationsat**: activation signal before the saturation.

    + **b**: saturation function parameter.

- Outputs:
    + Saturated force.
    '''
    return 2.0 / (1 + np.exp(-b * activationsat)) - 1

class MuscularActivation(object):


    def __init__(self, conf, pool, MUnumber, unit):

        self.conf = conf
        self.pool = pool
        self.MUnumber = MUnumber

        ## Model of the activation signal. For now, it can be *SOCDS* (second order critically damped system).
        self.activationModel = conf.parameterSet('activationModel', pool, 0)



        if self.activationModel == 'SOCDS':
            ## Matrix that multiplied by the vector formed as the formula below gives the activation
            ## signal at instant \f$n\f$:
            ## \f{equation}{
            ##    \resizebox{0.95\hsize}{!}{$Av(n) = \left[\begin{array}{ccccccccccc}a_1(n-1)&a_1(n-2)&e_1(n-1)&...&a_i(n-i)&a_i(n-2)&e_i(n-1)&...&a__{N_{MU}}(n-1)&a__{N_{MU}}(n-2)&e_{N_{MU}}(n-1)\end{array}\right]^T$}                    
            ## \f}
            ## where \f$a_i(n)\f$ is the activation signal of the motor unit \f$i\f$, \f$e_i(n)\f$ is
            ## 1/T (inverse of simulation time step, Dirac's delta approximation) if the motor unit \f$i\f$,
            ## fired at instant \f$n\f$. The vector \f$Av\f$ is updated every step at the function
            ## atualizeActivationSignal.
            ## The activation matrix itself is formed as:
            ## \f{equation}{
            ##      \resizebox{0.95\hsize}{!}{$\scriptstyle
            ##      A = \left[\begin{array}{ccccccccccc}\scriptscriptstyle  2\exp\left(-\frac{T}{T_{c_1}}\right)&\scriptscriptstyle -\exp\left(-2\frac{T}{T_{c_1}}\right)&\scriptscriptstyle  \frac{T^2}{T_{c_1}}\exp\left(1-\frac{T}{T_{c_1}} \right)&\scriptscriptstyle 0&\scriptscriptstyle ...&\scriptscriptstyle  0&\scriptscriptstyle  0& \scriptscriptstyle 0&\scriptscriptstyle 0&\scriptscriptstyle 0&\scriptscriptstyle 0\\
            ##                \scriptscriptstyle 0&\scriptscriptstyle 0&\scriptscriptstyle 0&\scriptscriptstyle \ddots&\scriptscriptstyle ...&&&&&\scriptscriptstyle ...&\scriptscriptstyle 0\\
            ##                \scriptscriptstyle 0&\scriptscriptstyle ...&&\scriptscriptstyle 0&\scriptscriptstyle 2\exp\left(-\frac{T}{T_{c_i}}\right)&\scriptscriptstyle -\exp\left(-2\frac{T}{T_{c_i}}\right)&\scriptscriptstyle \frac{T^2}{T_{c_i}}\exp\left(1-\frac{T}{T_{c_i}} \right)&\scriptscriptstyle 0&&&\scriptscriptstyle 0\\
            ##                \scriptscriptstyle0&\scriptscriptstyle0&\scriptscriptstyle...&&&\scriptscriptstyle0&\scriptscriptstyle 0&\scriptscriptstyle\ddots&\scriptscriptstyle0&\scriptscriptstyle0\\
            ##                \scriptscriptstyle0&\scriptscriptstyle0&\scriptscriptstyle0&\scriptscriptstyle...&&&&\scriptscriptstyle0&\scriptscriptstyle 2\exp\left(-\frac{T}{T_{c_{N_{MU}}}}\right)&\scriptscriptstyle -\exp\left(-2\frac{T}{T_{c_{N_{MU}}}}\right)&\scriptscriptstyle \frac{T^2}{T_{c_{{MU}}}}\exp\left(1-\frac{T}{T_{c_{N_{MU}}}} \right)\end{array}\right]$}
            ## \f} 
            ## The nonsaturated activation signal \f$a\f$ of all the motor units is obtained with:
            ## \f{equation}{
            ##   a = A.Av 
            ## \f}
            ## where each elemement o \f$a\f$ is the activation signal of a motor unit.
            self.ActMatrix = lil_matrix((self.MUnumber, 3*self.MUnumber), dtype = float)
            
            for i in xrange(0, self.MUnumber):
                self.ActMatrix[i,3*i:3*i+3] = [2*math.exp(-conf.timeStep_ms/unit[i].TwitchTc_ms),
                                    -math.exp(-2*conf.timeStep_ms/unit[i].TwitchTc_ms), 
                                    math.pow(conf.timeStep_ms, 2.0)/unit[i].TwitchTc_ms*math.exp(1.0-conf.timeStep_ms/unit[i].TwitchTc_ms)]

            self.ActMatrix = self.ActMatrix.tocsr() 
            #self.ActMatrix1 = csr_matrix(self.ActMatrix) 
            #print self.ActMatrix1
            ## Is a vector formed as:
            ## \f{equation}{
            ##    \resizebox{0.95\hsize}{!}{$Av(n) = \left[\begin{array}{ccccccccccc}a_1(n-1)&a_1(n-2)&e_1(n-1)&...&a_i(n-i)&a_i(n-2)&e_i(n-1)&...&a__{N_{MU}}(n-1)&a__{N_{MU}}(n-2)&e_{N_{MU}}(n-1)\end{array}\right]^T$}                    
            ## \f}
            ## It is multiplied by the matriz actMatrix to obtain the activation signal 
            ## (see actMatrix explanation)
            self.an = np.zeros((3*self.MUnumber, 1), dtype = float)

        ## The non-saturated activation signal of all motor units (see actMatrix explanation).
        self.activation_nonSat = np.zeros((self.MUnumber, 1), dtype = float)
        ## The parameter \f$b\f$ (see twitchSaturation function explanation) of 
        ## each motor unit.
        self.bSat = np.zeros((self.MUnumber,1), dtype = float)
       
        
        for i in xrange(0, self.MUnumber):
            self.bSat[i] = unit[i].bSat
            
        
        ## The non-saturated activation signal of all motor units (see actMatrix explanation).
        self.activation_Sat = np.zeros((self.MUnumber,1), dtype = float)    
        ## Dirac's delta approximation amplitude value. Is the inverse
        ## of the simulation time step (\f$1/T\f$). 
        self.diracDeltaValue = 1.0 / conf.timeStep_ms

    def atualizeActivationSignal(self, t, unit):
        '''
        Update the activation signal of the motor units.

        - Inputs:
            + **t**: current instant, in ms.        
        '''
               
        MUindices = np.arange(0, self.MUnumber)
        MUspike = np.array([], dtype = 'int')
        self.an[3*MUindices+1] = self.an[3*MUindices]
        self.an[3*MUindices] = self.activation_nonSat[MUindices]
        self.an[3*MUindices+2] =  0

        for i in xrange(self.MUnumber):
            if unit[i].terminalSpikeTrain and -1e-6 < (t - self.conf.timeStep_ms - unit[i].terminalSpikeTrain[-1][0]) < 1e-6: 
                MUspike = np.append(MUspike,i)
               
        self.an[3*MUspike+2] = self.diracDeltaValue
                
        self.activation_nonSat = self.ActMatrix.dot(self.an)  
        #self.ActMatrix1.
            
        self.activation_Sat = twitchSaturation(self.activation_nonSat, self.bSat)

    def reset(self):
        '''

        '''
        self.an = np.zeros((3*self.MUnumber, 1), dtype = float)
        self.activation_nonSat = np.zeros((self.MUnumber, 1), dtype = float)
        self.activation_Sat = np.zeros((self.MUnumber,1), dtype = float)    
