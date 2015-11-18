'''
Created on Sep 30, 2015

@author: root
'''



import math




def compValOn(v0, alpha, beta, t, t0): 
    return v0 * math.exp(beta*(t0 - t))
        
        

def compValOff(v0, alpha, beta, t, t0): 
    return 1.0 + (v0 - 1.0)  *  math.exp(alpha*(t0 - t))      


class PulseConductanceState(object):
    '''
    classdocs
    use the formalism of Destexhe(1997) to compute the Hodgkin-Huxley states of an ionic channel
    '''
    
    
    def __init__(self, kind, conf, pool,index):
        '''
        Constructor
        Inputs: kind: type of the state(m, h, n, q).
                    conf: an instance of the Configuration class with the functions to correctly parameterize the model. See the Configuration class.
                    pool: the pool that this state belongs.
                    index: the index of the unit that this state belongs.                    
        '''
        self.kind = kind
        self.value = float(0)
        
        
        self.v0 = 0.0
        self.t0 = float(0)
        
        self.state = False
        
        self.beta_ms1 = float(conf.parameterSet('beta_' + kind, pool, index))
        self.alpha_ms1 = float(conf.parameterSet('alpha_' + kind, pool,index))
        self.PulseDur_ms = float(conf.parameterSet('PulseDur_' + kind, pool, index)) 
        
        if (self.kind == 'm'):
            self.actType = 'activation'
        if (self.kind == 'h'):
            self.actType = 'inactivation'
        if (self.kind == 'n'):
            self.actType = 'activation'
        if (self.kind == 'q'):
            self.actType = 'activation'

        if (self.actType == 'activation'):
            self.computeValueOn = compValOn
            self.computeValueOff = compValOff            
        else:
            self.computeValueOn = compValOff
            self.computeValueOff = compValOn         
        

    def changeState(self,t):
        '''
        void function that modify the current situation (true/false) of the state
        inputs: t: instant t
        '''
        self.t0, self.v0,    self.state  = t, self.value, not self.state                
    
    
    def computeStateValue(self, t):
        '''
        compute the state value by using the approximation of Destexhe (1997) to compute the Hodgkin-Huxley states.
        input: t: instant t
        '''
        
        if (self.state):
            if (t - self.t0 > self.PulseDur_ms):
                self.changeState(t)
                self.value = self.computeValueOn(self.v0, self.alpha_ms1, self.beta_ms1, t, self.t0)                    
            else: self.value = self.computeValueOff(self.v0, self.alpha_ms1, self.beta_ms1, t, self.t0)
        else: self.value = self.computeValueOn(self.v0, self.alpha_ms1, self.beta_ms1, t, self.t0)

    
    