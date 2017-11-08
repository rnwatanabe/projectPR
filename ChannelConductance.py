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
import math
from PulseConductanceState import PulseConductanceState
#from numba import jit


def compCondKs(V_mV, gmax, state, EqPot):
        '''
        Computes the conductance of a slow potassium Channel. 
        This function is assigned as self.compCond to a Ks Channel at the class constructor.
        
        - Input:
            + **V_mV**: membrane potential of the compartment in mV.
        
        - Output:
            + Conductance in \f$\mu\f$S.
        
        It is computed as:

        \f{equation}{
            g = g_{max}q^2(E_0-V)
        \f}
        where \f$E_0\f$ is the equilibrium potential of the compartment, \f$V\f$ is the membrane potential
        and \f$q\f$ is the state of a slow potassium channel.
        '''
        return gmax * (state[0].value * state[0].value) * (EqPot - V_mV)


def compCondKsaxon(V_mV, gmax, state, EqPot):
        '''
        Computes the conductance of a slow potassium Channel. 
        This function is assigned as self.compCond to a Ks Channel in the axons
        at the class constructor.
        
        - Input:
            + **V_mV**: membrane potential of the compartment in mV.
        
        - Output:
            + Conductance in \f$\mu\f$S.
        
        It is computed as:

        \f{equation}{
            g = g_{max}s(E_0-V)
        \f}
        where \f$E_0\f$ is the equilibrium potential of the compartment, \f$V\f$ is the membrane potential
        and \f$q\f$ is the state of a slow potassium channel.
        '''
        return gmax * state[0].value * (EqPot - V_mV)

def compCondNa(V_mV, gmax, state, EqPot):
        '''
        Computes the conductance of a Na Channel. This function is assigned as self.compCond to a Na Channel at the class constructor.
        -Input:
            + **V_mV**: membrane potential of the compartment in mV.
        
        - Output:
            + Conductance in \f$\mu\f$S.

        It is computed as:

        \f{equation}{
            g = g_{max}m^3h(E_0-V)
        \f}
        where \f$E_0\f$ is the equilibrium potential of the compartment, V is the membrane potential
        and \f$m\f$ and \f$h\f$ are the states of a sodium channel..
        '''
        return gmax * (state[0].value * state[0].value * state[0].value) * state[1].value * (EqPot - V_mV)

def compCondNap(V_mV, gmax, state, EqPot):
        '''
        Computes the conductance of a Na persistent Channel. This function is assigned as self.compCond to a Na Channel at the class constructor.
        -Input:
            + **V_mV**: membrane potential of the compartment in mV.
        
        - Output:
            + Conductance in \f$\mu\f$S.

        It is computed as:

        \f{equation}{
            g = g_{max}m_p^3(E_0-V)
        \f}
        where \f$E_0\f$ is the equilibrium potential of the compartment, V is the membrane potential
        and \f$m\f$ and \f$h\f$ are the states of a persistent sodium channel.
        '''
        return gmax * (state[0].value * state[0].value * state[0].value) * (EqPot - V_mV)

def compCondKf(V_mV, gmax, state, EqPot):
        '''
        Computes the conductance of a Kf Channel. 
        This function is assigned as self.compCond to a Kf Channel at the class constructor.
        
        - Input:
            + **V_mV**: membrane potential of the compartment in mV.
        
        Output:
            + Conductance in \f$\mu\f$S.

        It is computed as:

        \f{equation}{
            g = g_{max}n^4(E_0-V)
        \f}
        where \f$E_0\f$ is the equilibrium potential of the compartment, V is the membrane potential
        and \f$n\f$ is the state of a fast potassium channel..
        '''
        return gmax * (state[0].value * state[0].value * state[0].value * state[0].value) * (EqPot - V_mV)

def compCondH(V_mV, gmax, state, EqPot):
        '''
        Computes the conductance of a HCN Channel. 
        This function is assigned as self.compCond to a HCF Channel at 
        the class constructor.
        
        - Input:
            + **V_mV**: membrane potential of the compartment in mV.
        
        Output:
            + Conductance in \f$\mu\f$S.

        It is computed as:

        \f{equation}{
            g = g_{max}q_h(E_0-V)
        \f}
        where \f$E_0\f$ is the equilibrium potential of the compartment, V is the membrane potential
        and \f$n\f$ is the state of an HCN channel..
        '''
        return gmax * state[0].value * (EqPot - V_mV)

class ChannelConductance(object):
    '''
    Class that implements a model of the ionic Channels in a compartment.
    '''

    
    def __init__(self, kind, conf, compArea, pool, neuronKind, compKind, index):
        '''
        Constructor
        
        Builds an ionic channel conductance.

        -Inputs: 
            + **kind**: string with the type of the ionic channel. For now it 
            can be *Na* (Sodium), *Ks* (slow Potassium), *Kf* (fast Potassium) or 
            *Ca* (Calcium).

            + **conf**: instance of the Configuration class (see Configuration file).

            + **compArea**: float with the area of the compartment that the Channel belongs, in \f$\text{cm}^2\f$.

            + **pool**: the pool that this state belongs.

            + **neuronKind**: string with the type of the motor unit. It used for 
            motoneurons. It can be *S* (slow), *FR* (fast and resistant), and *FF* 
            (fast and fatigable).

            + **compKind**: The kind of compartment that the Channel belongs. 
            For now, it can be *soma*, *dendrite*, *node* or *internode*.

            + **index**: the index of the unit that this state belongs.          
        '''
        ## string with the type of the ionic channel. For now it 
        ## can be *Na* (Sodium), *Ks* (slow Potassium), *Kf* (fast Potassium) or 
        ## *Ca* (Calcium).
        self.kind = str(kind)
       
        
        ## Equilibrium Potential of the ionic channel, mV.
        self.EqPot_mV = float(conf.parameterSet('EqPot_' + kind + '@' + compKind, pool, index))
        ## Maximal conductance, in \f$\mu\f$S, of the ionic channel. 
        self.gmax_muS = compArea * float(conf.parameterSet('gmax_' + kind + ':' + pool + '-' + neuronKind + '@' + compKind, pool, index))
                        
        ## String with type of dynamics of the states. For now it accepts the string pulse.
        self.stateType = conf.parameterSet('StateType', pool, index)
        
        if self.stateType == 'pulse':
            ConductanceState = PulseConductanceState

        ## List of ConductanceState objects, representing each state of the ionic channel.
        self.condState = []
        
        if self.kind == 'Kf':
            self.condState.append(ConductanceState('n', conf, pool, neuronKind, compKind, index))
            ## Function that computes the conductance dynamics.
            self.compCond = compCondKf
        if self.kind == 'Ks':
            self.condState.append(ConductanceState('q', conf, pool, neuronKind, compKind, index))
            self.compCond = compCondKs
        if self.kind == 'Na':
            self.condState.append(ConductanceState('m', conf, pool, neuronKind, compKind, index))
            self.condState.append(ConductanceState('h', conf, pool, neuronKind, compKind, index))
            self.compCond = compCondNa
        if self.kind == 'Ca':
            pass  # to be implemented
        if self.kind == 'Nap':
            self.condState.append(ConductanceState('mp', conf, pool, neuronKind, compKind, index))
            self.compCond = compCondNap
        if self.kind == 'KsAxon':
            self.condState.append(ConductanceState('s', conf, pool, neuronKind, compKind, index))
            self.compCond = compCondKsaxon
        if self.kind == 'H':
            self.condState.append(ConductanceState('qh', conf, pool, neuronKind, compKind, index))
            self.compCond = compCondH            
        
        ## Integer with the number of states in the ionic channel.    
        self.lenStates = len(self.condState)          
    
    #@profile
    def computeCurrent(self, t, V_mV): 
        '''
        Computes the current genrated by the ionic Channel
        
        - Inputs:
            + **t**: instant in ms.
            + **V_mV**: membrane potential of the compartment in mV.
        
        - Outputs:
            + Ionic current, in nA
        '''        
         
        for i in xrange(0, self.lenStates): 
            self.condState[i].computeStateValue(t)        
                          
        return self.compCond(V_mV, self.gmax_muS, self.condState, self.EqPot_mV)

    def reset(self):
        '''

        '''
        for i in xrange(self.lenStates):
            self.condState[i].reset()

   
    
            
    
    
         
    
    
         
        
