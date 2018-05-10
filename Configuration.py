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

    Contact: renato.watanabe@usp.br
'''

'''
\mainpage ReMoto in Python

This program is a neuronal simulation system, intended for studying spinal cord neuronal 
networks responsible for muscle control. These networks are affected by descending drive, 
afferent drive, and electrical nerve stimulation. The simulator may be used to investigate
phenomena at several levels of organization, e.g., at the neuronal membrane level or at 
the whole muscle behavior level (e.g., muscle force generation). This versatility is due 
to the fact that each element (neurons, synapses, muscle fibers) has its own specific 
mathematical model, usually involving the action of voltage- or neurotransmitter-dependent
ionic channels. The simulator should be helpful in activities such as interpretation of
results obtained from neurophysiological experiments in humans or mammals, proposal of 
hypothesis or testing models or theories on neuronal dynamics or neuronal network processing,
validation of experimental protocols, and teaching neurophysiology.

The elements that take part in the system belong to the following classes: motoneurons, 
muscle fibers (electrical activity and force generation), Renshaw cells, Ia inhibitory 
interneurons, Ib inhibitory interneurons, Ia and Ib afferents. The neurons are interconnected
by chemical synapses, which can be exhibit depression or facilitation.

The system simulates the following nuclei involved in flexion and extension of the human or
cat ankle: Medial Gastrocnemius (MG), Lateral Gastrocnemius (LG), Soleus (SOL), and Tibialis
Anterior (TA).

A web-based version can be found in [remoto.leb.usp.br](http://remoto.leb.usp.br/remoto/index.html).
The version to which this documentation  refers is from a Python program that can be found in
[github.com/rnwatanabe/projectPR](https://github.com/rnwatanabe/projectPR).

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
        self.confArray = np.genfromtxt(self.confArray, comments='%', dtype = ['S42', 'S60', 'S21'], delimiter = ',') 
        
        for i in xrange(0, len(self.confArray)):
            if self.confArray[i][0] == 'timeStep':
                ## Time step of the numerical solution of the differential equation.
                self.timeStep_ms = float(self.confArray[i][1])
            if self.confArray[i][0] == 'simDuration':
                ## Total length of the simulation in ms.
                self.simDuration_ms = float(self.confArray[i][1])
            if self.confArray[i][0] == 'skinThickness':
                ## skin thickness, in mm.
                self.skinThickness_mm = float(self.confArray[i][1])
            if self.confArray[i][0] == 'EMGAttenuationFactor':
                ## EMG attenuation factor, in 1/mm.
                self.EMGAttenuation_mm1 = float(self.confArray[i][1])
            if self.confArray[i][0] == 'EMGWideningFactor':
                ## EMG widening factor, in 1/mm.
                self.EMGWidening_mm1 = float(self.confArray[i][1])
            if self.confArray[i][0] == 'EMGNoiseEMG':
                ## EMG widening factor.
                self.EMGNoiseEMG = float(self.confArray[i][1])
            if self.confArray[i][0] == 'MUParameterDistribution':
                ## Distribution of the parameters along the motor units.
                self.MUParameterDistribution = self.confArray[i][1]
        ## The variable  timeStep divided by two, for computational efficiency.
        self.timeStepByTwo_ms = self.timeStep_ms / 2.0; 
        ## The variable  timeStep divided by six, for computational efficiency.
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
        
        if pool == 'SOL' or pool == 'MG' or pool == 'LG' or pool == 'TA':
            for i in xrange(0, len(self.confArray)):
                if self.confArray[i][0] == 'MUnumber_' + pool + '-S':
                    MUnumber_S = int(self.confArray[i][1])
                elif self.confArray[i][0] == 'MUnumber_' + pool + '-FR':
                    MUnumber_FR = int(self.confArray[i][1])
                elif self.confArray[i][0] == 'MUnumber_' + pool + '-FF':
                    MUnumber_FF = int(self.confArray[i][1])
            Nnumber = MUnumber_S + MUnumber_FR + MUnumber_FF 
        else:
            for i in xrange(0, len(self.confArray)):
                if self.confArray[i][0] == 'Number_' + pool:
                    Nnumber = int(self.confArray[i][1])
                    
                    
        paramVec_S, paramVec_FR, paramVec_FF, paramVec = np.array([]),np.array([]), np.array([]), np.array([])

        
        for i in xrange(0, len(self.confArray)):
            if self.confArray[i][0] == paramTag:
                if (self.confArray[0][2] == ''):                       
                    return self.confArray[i][1]
            else:
                if self.MUParameterDistribution == 'linear':       
                    if self.confArray[i][0] == paramTag + ':' + pool + '-S':
                        paramVec_S = np.linspace(float(self.confArray[i][1]), float(self.confArray[i][2]), MUnumber_S)
                        paramVec_S = paramVec_S + np.random.randn(len(paramVec_S))
                        paramVec = paramVec_S
                    elif self.confArray[i][0] == paramTag + ':' + pool + '-FR':
                        paramVec_FR = np.linspace(float(self.confArray[i][1]), float(self.confArray[i][2]), MUnumber_FR)
                    elif self.confArray[i][0] == paramTag + ':' + pool + '-FF':
                        paramVec_FF = np.linspace(float(self.confArray[i][1]), float(self.confArray[i][2]), MUnumber_FF)
                    elif self.confArray[i][0] == paramTag + ':' + pool + '-':
                        paramVec = np.linspace(float(self.confArray[i][1]), float(self.confArray[i][2]), Nnumber)                    
                elif self.MUParameterDistribution == 'exponential':           
                    if self.confArray[i][0] == paramTag + ':' + pool + '-S':
                        paramVec_S = np.array([float(self.confArray[i][1]), float(self.confArray[i][2])])
                    elif self.confArray[i][0] == paramTag + ':' + pool + '-FR':
                        paramVec_FR = np.array([float(self.confArray[i][1]), float(self.confArray[i][2])])
                    elif self.confArray[i][0] == paramTag + ':' + pool + '-FF':
                        paramVec_FF = np.array([float(self.confArray[i][1]), float(self.confArray[i][2])])
                    elif self.confArray[i][0] == paramTag + ':' + pool + '-':
                        try:
                            paramVec = float(self.confArray[i][1])*np.exp(1.0/Nnumber*np.log(float(self.confArray[i][2])/float(self.confArray[i][1])) * np.linspace(0,Nnumber,Nnumber))
                        except ZeroDivisionError:
                            paramVec = np.exp(1.0/Nnumber*np.log(float(self.confArray[i][2]) + 1) * np.linspace(0,Nnumber,Nnumber)) - 1
        
        if self.MUParameterDistribution == 'linear':           
            if paramVec_FR.size > 0:
                paramVec = np.hstack((paramVec, paramVec_FR))
                if paramVec_FF.size > 0:
                    paramVec = np.hstack((paramVec, paramVec_FF))
        elif self.MUParameterDistribution == 'exponential':           
            if paramVec_S.size > 0:
                indexUnits = np.linspace(0,Nnumber, Nnumber)
                if paramTag == 'twitchPeak':
                    paramVec = paramVec_S[0]*np.exp(1.0/Nnumber*np.log(paramVec_FF[1]/paramVec_S[0]) * np.linspace(0,Nnumber,Nnumber))   
                else:
                    paramVec = ((paramVec_S[0] - (paramVec_S[1]+paramVec_FR[0])/2.0) * np.exp(-5.0*indexUnits/MUnumber_S)
                                + ((paramVec_S[1]+paramVec_FR[0])/2.0 - paramVec_FF[1]) 
                                * (1 - np.exp(1.0/MUnumber_FF*np.log(((paramVec_FR[1]+paramVec_FF[0])/2.0 - (paramVec_S[1] + paramVec_FR[0])/2.0)/(paramVec_FF[1]- (paramVec_S[1]+paramVec_FR[0])/2.0)) * (Nnumber - indexUnits)))
                                + paramVec_FF[1]) 
                
        
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
            pos = self.confArray[i][0].find('Con:' + neuralSource)
            if pos >= 0 and float(self.confArray[i][1]) > 0:
                posUnitKind = self.confArray[i][0].find('-', pos+len('Con:' + neuralSource)+1)
                posComp = self.confArray[i][0].find('@')
                posKind = self.confArray[i][0].find('|')
                Synapses.append([self.confArray[i][0][pos+len('Con:' + neuralSource)+1:posUnitKind],
                                 self.confArray[i][0][posUnitKind+1:posComp],
                                 self.confArray[i][0][posComp+1:posKind],
                                 self.confArray[i][0][posKind+1:]])
        return Synapses
    
    def changeConfigurationParameter(self, parameter, value1, value2):
        '''
        '''
        for i in xrange(0, len(self.confArray)):
            if self.confArray[i][0] == parameter:
                self.confArray[i][1] = value1
                self.confArray[i][2] = value2
        