'''
Created on Oct 26, 2015

@author: root
'''


import numpy as np

from NeuralTract import NeuralTract
from SynapticNoise import SynapticNoise

class SynapsesFactory(object):
    '''
    Class to build all the synapses in the system.
    '''


    def __init__(self, conf, pools):
        '''
        Constructor

        - Inputs:
            + **conf**: Configuration object with the simulation parameters.

            + **pools**: list of all the pools in the system.

        
        '''
        ## Total number of synapses in the system.
        self.numberOfSynapses = 0
        #pools.append(NeuralTract(conf, 'NoiseRC'))


        for poolOut in xrange(len(pools)):
            for unitOut in xrange(len(pools[poolOut].unit)):
                pools[poolOut].unit[unitOut].SynapsesOut = conf.determineSynapses(pools[poolOut].pool + '-' + 
                                                                                  pools[poolOut].unit[unitOut].kind)
                for synapseIn in xrange(len(pools[poolOut].unit[unitOut].SynapsesOut)):
                    conn = float(conf.parameterSet('Con_' + pools[poolOut].pool + '-' 
                                                   + pools[poolOut].unit[unitOut].kind + '>'
                                                   + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0]
                                                   + '-'
                                                   + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][1]
                                                   + '@'
                                                   + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][2]
                                                   + '|'
                                                   + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][3],
                                                   '', 0)) / 100.0
                    gmax = float(conf.parameterSet('gmax_' + pools[poolOut].pool + '-'
                                                   + pools[poolOut].unit[unitOut].kind + '>'
                                                   + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0]
                                                   + '-'
                                                   + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][1]
                                                   + '@'
                                                   + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][2]
                                                   + '|'
                                                   + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][3],
                                                   '', 0))
                    delay = float(conf.parameterSet('delay_' + pools[poolOut].pool + '-'
                                                    + pools[poolOut].unit[unitOut].kind + '>'
                                                    + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0]
                                                    + '-'
                                                    + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][1]
                                                    + '@'
                                                    + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][2]
                                                    + '|'
                                                    + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][3],
                                                    '', 0))
                    declineFactor = float(conf.parameterSet('dec_' + pools[poolOut].pool + '-'
                                                    + pools[poolOut].unit[unitOut].kind + '>'
                                                    + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0]
                                                    + '-'
                                                    + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][1]
                                                    + '@'
                                                    + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][2]
                                                    + '|'
                                                    + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][3],
                                                    '', 0))
                    dyn = conf.parameterSet('dyn_' + pools[poolOut].pool + '-'
                                            + pools[poolOut].unit[unitOut].kind + '>'
                                            + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0]
                                            + '-' + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][1]
                                            + '@' + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][2]
                                            + '|' + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][3],
                                            '', 0)
                    if dyn != 'None':
                        var = float(conf.parameterSet('var_' + pools[poolOut].pool + '-'
                                                      + pools[poolOut].unit[unitOut].kind + '>'
                                                      + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0]
                                                      + '-' + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][1]
                                                      + '@' + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][2]
                                                      + '|' + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][3],
                                                      '', 0))
                        tau = float(conf.parameterSet('tau_' + pools[poolOut].pool + '-'
                                                      + pools[poolOut].unit[unitOut].kind + '>'
                                                      + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0]
                                                      + '-' + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][1]
                                                      + '@' + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][2]
                                                      + '|' + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][3],
                                                      '', 0))
                    else:
                        var = 0
                        tau = 0
                    for poolIn in xrange(len(pools)):
                        if (pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0].find(pools[poolIn].pool)>=0):
                            for unitIn in xrange(len(pools[poolIn].unit)):
                                for compartmentIn in xrange(len(pools[poolIn].unit[unitIn].compartment)):
                                    if pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0] == pools[poolIn].pool and pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][1] == pools[poolIn].unit[unitIn].kind and pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][2] == pools[poolIn].unit[unitIn].compartment[compartmentIn].kind:
                                        if np.random.uniform(0.0, 1.0) <= conn:
                                            for synapse in xrange(len(pools[poolIn].unit[unitIn].compartment[compartmentIn].SynapsesIn)): 
                                                if pools[poolIn].unit[unitIn].compartment[compartmentIn].SynapsesIn[synapse].kind == pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][3]:
                                                    if np.isfinite(declineFactor):
                                                        neuronsDistance = np.abs(pools[poolIn].unit[unitIn].position_mm
                                                                                    - pools[poolOut].unit[unitOut].position_mm)
                                                        weight = declineFactor / (declineFactor + neuronsDistance**2)
                                                    else:
                                                        weight = 1
                                                    pools[poolIn].unit[unitIn].compartment[compartmentIn].SynapsesIn[synapse].addConductance(gmax*weight, delay, dyn, var, tau)
                                                    pools[poolOut].unit[unitOut].transmitSpikesThroughSynapses.append(pools[poolIn].unit[unitIn].compartment[compartmentIn].SynapsesIn[synapse])                                                            
                                                    pools[poolOut].unit[unitOut].indicesOfSynapsesOnTarget.append(len(pools[poolIn].unit[unitIn].compartment[compartmentIn].SynapsesIn[synapse].gmax_muS) - 1)
                                                    self.numberOfSynapses += 1

        

        NoiseSynapsesOut = conf.determineSynapses('Noise')
        for synapseIn in xrange(len(NoiseSynapsesOut)):
            pools.append(SynapticNoise(conf, NoiseSynapsesOut[synapseIn][0]))
            poolOut = len(pools) - 1
            gmax = float(conf.parameterSet('gmax_Noise>' + NoiseSynapsesOut[synapseIn][0] 
                                           + '-' + NoiseSynapsesOut[synapseIn][1]
                                           + '@' + NoiseSynapsesOut[synapseIn][2] + '|'
                                           + NoiseSynapsesOut[synapseIn][3],
                                           '', 0))
            delay = float(conf.parameterSet('delay_Noise>' + NoiseSynapsesOut[synapseIn][0]
                                            + '-' + NoiseSynapsesOut[synapseIn][1]
                                            + '@' + NoiseSynapsesOut[synapseIn][2] + '|'
                                            + NoiseSynapsesOut[synapseIn][3],
                                            '', 0))
            declineFactor = float(conf.parameterSet('dec_Noise>' + NoiseSynapsesOut[synapseIn][0]
                                                    + '-' + NoiseSynapsesOut[synapseIn][1]
                                                    + '@' + NoiseSynapsesOut[synapseIn][2] + '|'
                                                    + NoiseSynapsesOut[synapseIn][3],
                                                    '', 0))
            dyn = conf.parameterSet('dyn_Noise>' + NoiseSynapsesOut[synapseIn][0] 
                                    + '-' + NoiseSynapsesOut[synapseIn][1]
                                    + '@' + NoiseSynapsesOut[synapseIn][2] + '|'
                                    + NoiseSynapsesOut[synapseIn][3],
                                    '', 0)
            if dyn != 'None':
                var = float(conf.parameterSet('var_Noise>' + NoiseSynapsesOut[synapseIn][0]
                                              + '-' + NoiseSynapsesOut[synapseIn][1]
                                              + '@' + NoiseSynapsesOut[synapseIn][2] + '|' 
                                              + NoiseSynapsesOut[synapseIn][3],
                                              '', 0))
                tau = float(conf.parameterSet('tau_Noise>' + NoiseSynapsesOut[synapseIn][0]
                                              + '-' + NoiseSynapsesOut[synapseIn][1]
                                              + '@' + NoiseSynapsesOut[synapseIn][2]
                                              + '|' + NoiseSynapsesOut[synapseIn][3],
                                              '', 0))
            else:
                var = 0
                tau = 0
            for unitOut in xrange(len(pools[poolOut].unit)):
                for poolIn in xrange(len(pools)):
                    if NoiseSynapsesOut[synapseIn][0] == pools[poolIn].pool and pools[poolIn].kind != 'SN':
                        for unitIn in xrange(len(pools[poolIn].unit)):
                            for compartmentIn in xrange(len(pools[poolIn].unit[unitIn].compartment)):
                                if NoiseSynapsesOut[synapseIn][1] == pools[poolIn].unit[unitIn].kind and NoiseSynapsesOut[synapseIn][2] == pools[poolIn].unit[unitIn].compartment[compartmentIn].kind and pools[poolIn].unit[unitIn].index == pools[poolOut].unit[unitOut].index:
                                    for synapse in xrange(len(pools[poolIn].unit[unitIn].compartment[compartmentIn].SynapsesIn)): 
                                        if pools[poolIn].unit[unitIn].compartment[compartmentIn].SynapsesIn[synapse].kind == NoiseSynapsesOut[synapseIn][3]:
                                            if np.isfinite(declineFactor):
                                                neuronsDistance = np.abs(pools[poolIn].unit[unitIn].position_mm
                                                                            - pools[poolOut].unit[unitOut].position_mm)
                                                weight = declineFactor / (declineFactor + neuronsDistance**2)
                                            else:
                                                weight = 1
                                            pools[poolIn].unit[unitIn].compartment[compartmentIn].SynapsesIn[synapse].addConductance(gmax*weight, delay, dyn, var, tau)
                                            pools[poolOut].unit[unitOut].transmitSpikesThroughSynapses.append(pools[poolIn].unit[unitIn].compartment[compartmentIn].SynapsesIn[synapse])                                                            
                                            pools[poolOut].unit[unitOut].indicesOfSynapsesOnTarget.append(len(pools[poolIn].unit[unitIn].compartment[compartmentIn].SynapsesIn[synapse].gmax_muS) - 1)
                                            self.numberOfSynapses += 1
                  

        

        print 'All the ' + str(self.numberOfSynapses) +  ' synapses were built'                   
        