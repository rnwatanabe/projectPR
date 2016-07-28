'''
Created on Oct 8, 2015

@author: root
'''

import numpy as np
from MotorUnit import MotorUnit
import math
from scipy.sparse import lil_matrix

 

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
        MUnumber_S = int(conf.parameterSet('MUnumber_S_' + pool, pool, 0))
        MUnumber_FR = int(conf.parameterSet('MUnumber_FR_' + pool, pool, 0))
        MUnumber_FF = int(conf.parameterSet('MUnumber_FF_' + pool, pool, 0))
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
                self.ActMatrix[i,3*i:3*i+3] = [2*math.exp(-conf.timeStep_ms/self.unit[i].TwitchTc_ms),
                                    -math.exp(-2*conf.timeStep_ms/self.unit[i].TwitchTc_ms), 
                                    math.pow(conf.timeStep_ms, 2.0)/self.unit[i].TwitchTc_ms*math.exp(1.0-conf.timeStep_ms/self.unit[i].TwitchTc_ms)]
             
            self.ActMatrix.tocsr()   
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
        self.bSat = np.zeros((self.MUnumber, 1), dtype = float)
        ## Twitch- tetanus relationship (see atualizeForceNoHill function explanation)
        self.twTet = np.zeros((self.MUnumber, 1), dtype = float)
        ## Amplitude of the muscle unit twitch, in N (see atualizeForceNoHill function explanation).
        self.twitchAmp_N = np.zeros((self.MUnumber, 1), dtype = float)
        for i in xrange(0, self.MUnumber):
            self.bSat[i] = self.unit[i].bSat
            self.twitchAmp_N[i] = self.unit[i].TwitchAmp_N
            self.twTet[i] = self.unit[i].twTet  
        
        ## The non-saturated activation signal of all motor units (see actMatrix explanation).
        self.activation_Sat = np.zeros((self.MUnumber, 1), dtype = float)    
        ## Dirac's delta approximation amplitude value. Is the inverse
        ## of the simulation time step (\f$1/T\f$). 
        self.diracDeltaValue = 1.0 / conf.timeStep_ms
        
        #Force
        ## Muscle force along time, in N.
        self.force = np.zeros((int(np.rint(conf.simDuration_ms/conf.timeStep_ms)), 1), dtype = float)
        ## String indicating whther a Hill model is used or not. For now, it can be *No*.
        self.hillModel = conf.parameterSet('hillModel',pool, 0)
        if self.hillModel == 'No': self.atualizeForce = self.atualizeForceNoHill
        
        self.timeIndex = 0;
        
        
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
        for i in self.unit: i.atualizeMotorUnit(t)
        self.atualizeActivationSignal(t)
        self.atualizeForce()
        self.timeIndex += 1

    def atualizeActivationSignal(self, t):
        '''
        Update the activation signal of the motor units.

        - Inputs:
            + **t**: current instant, in ms.        
        '''
        for i in xrange(self.MUnumber):
            self.an[3*i+1] = self.an[3*i]
            self.an[3*i] = self.activation_nonSat[i]
            if self.unit[i].terminalSpikeTrain and abs(t - self.conf.timeStep_ms - self.unit[i].terminalSpikeTrain[-1][0]) < 1e-6: 
                self.an[3*i+2] = self.diracDeltaValue
            else: self.an[3*i+2] =  0.0
        
        self.activation_nonSat = self.ActMatrix.dot(self.an)        
        self.activation_Sat = twitchSaturation(self.activation_nonSat, self.bSat)
        
    
    def atualizeForceNoHill(self):
        '''
        Compute the muscle force when no muscle dynamics (Hill model) is used. This
        operation is vectorized. Each element of the vectors correspond to one motor
        unit. For each motor unit, the force is computed by the following formula:

        \f{equation}{
            F_{MU} = a_{sat}A_{MU}R_{MU}
        }
        where \f$a_{sat}\f$ is the saturated activation signal, \f$A_{MU}\f$ is the
        motor unit twitch amplitude, and  \f$R_{MU}\f$ is the relation between 
        the twitch amplitude and the tetanus of the motor unit. 

        Then the muscle force is obtained from: 

        \f{equation}{
            F = \limits\sum_{i=1}^N_{MU}F_{i}
        }
        where \f$N_{MU}\f$ is the number of motor units in the pool.
        '''
        self.force[self.timeIndex] = np.sum(self.activation_Sat * self.twitchAmp_N * self.twTet)            

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
        
    
        
        
    
        
          