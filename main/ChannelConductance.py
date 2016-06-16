'''
Author - Renato Naville Watanabe
'''

import numpy as np
import math
from PulseConductanceState import PulseConductanceState



class ChannelConductance(object):
    '''
    Class that implements a model of the ionic Channels in a compartment.
    '''

    
    def __init__(self, kind, conf, compArea, pool, index):
        '''
        Builds an ionic channel conductance.

        -Inputs: 
            + **kind**: string with the type of the ionic channel (Na, Ks, Kf or Ca).
            + **conf**: instance of the Configuration class (see Configuration file).
            + **compArea**: - float with the area of the compartment that the Channel belongs, in \f$\text{cm}^2\f$.
            + **pool** - the pool that this state belongs.
            + **index** - the index of the unit that this state belongs.          
        '''
        ## String with the type of the ionic channel (Na, Ks, Kf or Ca).
        self.kind = str(kind)
        ## List of ConductanceState objects, representing each state of the ionic channel.
        self.condState = []
        
        ## Equilibrium Potential of the ionic channel, mV.
        self.EqPot_mV = float(conf.parameterSet('EqPot_' + kind, pool, index))
        ## Maximal conductance, in \f$\mu\f$S, of the ionic channel. 
        self.gmax_muS = compArea * float(conf.parameterSet('gmax_' + kind, pool, index))                
        
        ## String with type of dynamics of the states. For now it accepts the string pulse.
        self.stateType = conf.parameterSet('StateType', pool, index)
        
        if self.stateType == 'pulse':
            ConductanceState = PulseConductanceState
           
        if(self.kind == 'Kf'):
            self.condState.append(ConductanceState('n', conf, pool, index))
            ## Function that computes the conductance dynamics.
            self.compCond = self.compCondKf
        if(self.kind == 'Ks'):
            self.condState.append(ConductanceState('q', conf, pool, index))
            self.compCond = self.compCondKs
        if(self.kind == 'Na'):
            self.condState.append(ConductanceState('m', conf, pool, index))
            self.condState.append(ConductanceState('h', conf, pool, index))
            self.compCond = self.compCondNa
        if(self.kind == 'Ca'):
            pass  # to be implemented
          
            
        
        ## Integer with the number of states in the ionic channel.    
        self.lenStates = len(self.condState)  
        
        
          
    
    
    def computeCurrent(self, t, V_mV): 
        '''
        Computes the current genrated by the ionic Channel
        
        - Inputs:
            + **t**: instant in ms.
            + **V_mV**: membrane potential of the compartment in mV.
        
        - Outputs:
            + Ionic current in nA
        '''        
        for i in xrange(0, self.lenStates): self.condState[i].computeStateValue(t)        
                          
        return self.compCond(V_mV)
   
    def compCondKf(self, V_mV):
        '''
        Computes the conductance of a Kf Channel. 
        This function is assigned as self.compCond to a Kf Channel at the class constructor.
        
        - Input:
            + **V_mV**: membrane potential of the compartment in mV.
        
        Output:
            + Conductance in \f$\mu\f$S.
        '''
        return self.gmax_muS * math.pow(self.condState[0].value, 4.0) * (self.EqPot_mV - V_mV)
            
    
    def compCondKs(self, V_mV):
        '''
        Computes the conductance of a Ks Channel. 
        This function is assigned as self.compCond to a Ks Channel at the class constructor.
        
        - Input:
            + **V_mV**: membrane potential of the compartment in mV.
        
        - Output:
            + Conductance in \f$\mu\f$S.
        '''
        return self.gmax_muS * math.pow(self.condState[0].value, 2.0) * (self.EqPot_mV - V_mV)
         
    
    def compCondNa(self, V_mV):
        '''
        Computes the conductance of a Na Channel. This function is assigned as self.compCond to a Na Channel at the class constructor.
        
        -Input:
            + **V_mV**: membrane potential of the compartment in mV.
        
        Output:
            + Conductance in \f$\mu\f$S.
        '''
        return self.gmax_muS * math.pow(self.condState[0].value, 3.0) * self.condState[1].value * (self.EqPot_mV - V_mV)
         
        