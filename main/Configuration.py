'''
Created on Sep 30, 2015

@author: root
'''

import numpy as np

class Configuration(object):
    '''
    classdocs
    '''


    def __init__(self, filename):
        '''
        Constructor
        '''
        self.confArray = open(filename,'r') 
        # this array will store all the contents of the configuration file
        self.confArray = np.genfromtxt(self.confArray, comments='%', dtype = ['S21', 'S60', 'S21'], delimiter = ',') 
        # variables timeStep and simDuration that are used in the whole system are set at the constructor of the Configuration class
        for i in xrange(0, len(self.confArray)):
            if self.confArray[i][0] == 'timeStep':
                self.timeStep_ms = float(self.confArray[i][1])
            if self.confArray[i][0] == 'simDuration':
                self.simDuration_ms = float(self.confArray[i][1])
        self.timeStepByTwo_ms = self.timeStep_ms / 2.0;
        self.timeStepBySix_ms = self.timeStep_ms / 6.0; 
           
        
        
    def parameterSet(self, paramTag, pool, index):
        '''
        function that returns the value of wished parameter specified in the paramTag variable.
        In the case of min/max parameters, the value returned is the specific to the index of the unit that called the
        function. 
        '''
        #get 
        for i in xrange(0, len(self.confArray)):
            if self.confArray[i][0] == 'MUnumber_S_' + pool:
                MUnumber_S = int(self.confArray[i][1])
            elif self.confArray[i][0] == 'MUnumber_FR_' + pool:
                MUnumber_FR = int(self.confArray[i][1])
            elif self.confArray[i][0] == 'MUnumber_FF_' + pool:
                MUnumber_FF = int(self.confArray[i][1])
        
        paramVec_S, paramVec_FR, paramVec_FF = np.array([]),np.array([]), np.array([])
                
        for i in xrange(0, len(self.confArray)): 
            if self.confArray[i][0] == paramTag:
                if (self.confArray[0][2] == ''):
                    return self.confArray[i][1]
            else:
                if (self.confArray[i][0] == paramTag + '_S_' + pool):
                    paramVec_S = np.linspace(float(self.confArray[i][1]), float(self.confArray[i][2]), MUnumber_S)
                elif (self.confArray[i][0] == paramTag + '_FR_' + pool):
                    paramVec_FR = np.linspace(float(self.confArray[i][1]), float(self.confArray[i][2]), MUnumber_FR)
                elif (self.confArray[i][0] == paramTag + '_FF_' + pool):
                    paramVec_FF = np.linspace(float(self.confArray[i][1]), float(self.confArray[i][2]), MUnumber_FF)
                       
        paramVec = paramVec_S
        if (paramVec_FR.size > 0):
            paramVec = np.concatenate(paramVec, paramVec_FR)
            if (paramVec_FF.size > 0):
                paramVec = np.concatenate(paramVec, paramVec_FF)
              
        return paramVec[index]
    
    
    def inputFunctionGet(self, function):       
        '''
        Returns a numpy array with the values of the function for the whole simulation.
        It is used to obtain before the simulation run all the values of the inputs.
        '''
        
        t = np.arange(0, self.simDuration_ms, self.timeStep_ms)
        
        return function(t)
        
        
        
        