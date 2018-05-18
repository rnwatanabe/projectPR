'''
    Neuromuscular simulator in Python.
    Copyright (C) 2018  Renato Naville Watanabe

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




from Synapse import Synapse
import math
import numpy as np


def calcGLeak(area, specificRes):
    '''
    Computes the leak conductance of the compartment.

    - Input:
        + **area**: area of the compartment in cm\f$^2\f$.
        + **specificRes**: specific resistance of the compartment
        in \f$\Omega.cm^2\f$.

    - Output:
        + Leak conductance in \f$\mu\f$S.

    It is compute according to the following formula:

    \f{equation}{
        g = 10^6 . \frac{A}{\rho}
    \f}
    where \f$A\f$ is the compartment area [cm\f$^2\f$], \f$\rho\f$ is
    the specific resistance [\f$\Omega.cm^2\f$] and \f$g\f$ is the
    compartment conductance [\muS].
    '''
    return (1e6 * area) / specificRes

class CompartmentNoChannel(object):
    '''
    Class that implements a neural compartment. For now it is implemented
    *dendrite* and *soma*.
    '''


    def __init__(self, kind, conf, pool, index, neuronKind):
        '''
        Constructor

        - Inputs:
            + **kind**: The kind of compartment. For now, it can be *soma*, *dendrite*, 
            *node* or *internode*.

            + **conf**: Configuration object with the simulation parameters.

            + **pool**: string with Motor unit pool to which the motor unit belongs.

            + **index**: integer corresponding to the motor unit order in the pool, according to 
            the Henneman's principle (size principle).

            + **neuronKind**: string with the type of the motor unit. It can be *S* (slow), *FR* (fast and resistant), 
            and *FF* (fast and fatigable).
        '''
        
        
        ## String with the type of the motor unit. It can be *S* (slow), *FR* (fast and resistant), 
        ## and *FF* (fast and fatigable).
        self.neuronKind = neuronKind
        ## List of summed synapses (see Lytton, 1996) that the Compartment do with other neural components.
        self.SynapsesOut = []
        
        ## List of summed synapses (see Lytton, 1996) that the Compartment receive from other neural components.
        self.SynapsesIn = [] 
        self.SynapsesIn.append(Synapse(conf, pool, index, kind, 'excitatory', neuronKind))
        self.SynapsesIn.append(Synapse(conf, pool, index, kind, 'inhibitory', neuronKind))
        
        ## The kind of compartment. For now, it can be *soma* or *dendrite*.
        self.kind = kind
        
        ## Integer corresponding to the motor unit order in the pool, according to 
        ## the Henneman's principle (size principle).
        self.index = index
        
        ## Length of the compartment, in \f$\mu\f$m.
        self.length_mum = float(conf.parameterSet('l@' + kind, pool, index))
        ## Diameter of the compartment, in \f$\mu\f$m.
        self.diameter_mum = float(conf.parameterSet('d@' + kind, pool, index))
        area_cm2 = float(self.length_mum * math.pi * self.diameter_mum * 1e-8)
        specifRes_Ohmcm2 = float(conf.parameterSet('res@' + kind, pool, index))
        ## Capacitance of the compartment, in nF.
        self.capacitance_nF = float(conf.parameterSet('membCapac', pool, index)) * area_cm2 * 1e3
        
        ## Equilibrium potential, in mV.
        self.EqPot_mV = float(conf.parameterSet('EqPot@' + self.kind, pool, index))

        ## Pump current in the compartment, in nA.
        self.IPump_nA = float(conf.parameterSet('IPump@' + self.kind, pool, index))

        ## Leak conductance of the compartment, in \f$\mu\f$S.
        self.gLeak_muS = calcGLeak(area_cm2, specifRes_Ohmcm2)
        

   
        
    #@profile    
    def computeCurrent(self, t, V_mV):
        '''
        Computes the active currents of the compartment. Active currents are the currents from the ionic channels
        and from the synapses.

        - Inputs:
            + **t**: current instant, in ms.

            + **V_mV**: membrane potential, in mV.
        '''
        
        I = 0.0

        if self.SynapsesIn[0].numberOfIncomingSynapses: I += self.SynapsesIn[0].computeCurrent(t, V_mV)
        if self.SynapsesIn[1].numberOfIncomingSynapses: I += self.SynapsesIn[1].computeCurrent(t, V_mV)
       
        
        return I

    def reset(self):
        '''

        '''
        for i in xrange(len(self.SynapsesIn)):
            self.SynapsesIn[i].reset()
        