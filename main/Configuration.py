'''
\mainpage ReMoto in Python

This program ...

'''

import numpy as np

class Configuration(object):
    '''
    Class that builds an object of Configuration, based on a configuration file.
    '''


    def __init__(self, filename):
        '''
        Constructor.
        
        Builds the Configuration object. A Configuration object is responsible to set the variables
        that are used in the whole system, such as timeStep and simDuration.
          
        - Inputs:
            + **filename**: name of the file with the parameter values. The extension  of the file should be .rmto.
          
        '''

        ## An array with all the simulation parameters.
        self.confArray = open(filename,'r') 
        # This array will store all the contents of the configuration file
        self.confArray = np.genfromtxt(self.confArray, comments='%', dtype = ['S29', 'S60', 'S21'], delimiter = ',') 
        
        for i in xrange(0, len(self.confArray)):
            if self.confArray[i][0] == 'timeStep':
                ## Time step of the numerical solution of the differential equation.
                self.timeStep_ms = float(self.confArray[i][1])
            if self.confArray[i][0] == 'simDuration':
                ## Total length of the simulation in ms.
                self.simDuration_ms = float(self.confArray[i][1])
        ## The variable  timeStep divided by two, for computaional efficiency.
        self.timeStepByTwo_ms = self.timeStep_ms / 2.0; 
        ## The variable  timeStep divided by six, for computaional efficiency.
        self.timeStepBySix_ms = self.timeStep_ms / 6.0; 
           
        
        
    def parameterSet(self, paramTag, pool, index):
        '''
        Function that returns the value of wished parameter specified in the paramTag variable.
        In the case of min/max parameters, the value returned is the specific to the index of the unit that called the
        function. 


- Inputs: 

    + **paramTag**: string with the name of the wished parameter as in the first column of the rmto file.

    + **pool**: pool from which the unit that will receive the parameter value belongs. For example SOL. 
    It is used only in the parameters that have a range.

    + **index**: index of the unit. It is is an integer.

- Outputs:
    
    + required parameter value
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

        - Inputs:
            + **function**: function from which is desired to obtain its values  during the simulation duration.
        
        - Output:
            + narray with the function values for each instant.
        '''
        
        t = np.arange(0, self.simDuration_ms, self.timeStep_ms)        
        return function(t)
        
     
    def determineSynapses(self, neuralSource):
        '''
        Function used to determine all the synapses that a given pool makes. It is used in the SynapsesFactory class.
        
- Inputs:
    + **neuralSource** - string with the pool name from which is desired to know what synapses it will make.

- Outputs:
    + array of strings with all the synapses target that the neuralSource will make.
        '''
        Synapses = []
        
        for i in xrange(0, len(self.confArray)):
            pos = self.confArray[i][0].find('Con_' + neuralSource)
            if (pos >= 0 and float(self.confArray[i][1]) > 0):
                posComp = self.confArray[i][0].find('__')
                Synapses.append([self.confArray[i][0][pos+len('Con_'  + neuralSource)+1:posComp], 
                                 self.confArray[i][0][posComp+2:]])
        return Synapses       
        