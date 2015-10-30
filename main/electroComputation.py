'''
Created on Oct 7, 2015

@author: root
'''
import numpy as np
import math


def calcGCoupling(cytR, lComp1, lComp2, dComp1, dComp2):
    '''
    calculates the coupling conductance between two compartments
    Inputs: cytR. Cytoplasmatic resistance in Ohm.cm
                lComp1, lComp2: length of the compartments in mum
                dComp1, dComp2 diameter of the compartments in mum
    Output: coupling conductance in MS
    '''
    rAxis1 = (cytR * lComp1) / (math.pi * math.pow(dComp1/2, 2))
    rAxis2 = (cytR * lComp2) / (math.pi * math.pow(dComp2/2, 2))
    
    return (1e2 * 2) / (rAxis1 + rAxis2)


def calcGLeak(area, specificRes):
    '''
    computes the leak conductance of the compartment
    input: area: area of the compartment in cm2
                specificRes: specific resistance of the compartment in Ohm.cm2
    output: gLeak in MS 
    '''    
    return (1e6 * area) / specificRes

def compGCouplingMatrix(gc):|
    '''
    computes the Coupling Matrix to be used in the dVdt function of the N compartments of the motor unit. 
    The Matrix uses the values obtained with the function calcGcoupling.
                _______________________________________________________
                |-gc[0]           gc[0]              0      ....                    .....          0        0          0|
                |gc[0]    -gc[0]-gc[1]    gc[1]    0  .....                     .....                           0|
                |  .      .                ...                        ...                                .....                      0|  
    GC =   |  :        . .  . ........................................                                                           :|
                |  0  .......   0         gc[i-1]    -gc[i-1]-gc[i]    gc[i]    0     .....            0        :|
                |  0 ...    ...............................                                ....        ..............................|
                |  0  .............................................gc[N-2]    -gc[N-2]-gc[N-1]    gc[N-1]|
                |  0 ..........................................................................  gc[N-1]        -gc[N-1]|
                |---------------------------------------------------------------------------------------------------|
    Inputs: the vector with N elements, with the coupling conductance of each compartment of the Motor Unit.
    Output: the GC matrix
    '''
    
    GC = np.zeros((len(gc),len(gc)))
    
    for i in xrange(0, len(gc)):
        if i == 0:
            GC[i,i:i+2] = [-gc[i], gc[i]] 
        elif i == len(gc) - 1:
            GC[i,i-1:i+1] = [gc[i-1], -gc[i-1]]  
        else:
            GC[i,i-1:i+2] = [gc[i-1], -gc[i-1]-gc[i], gc[i]]
                  
            
    return GC
    