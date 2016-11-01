'''
Created on Oct 26, 2015

@author: root
'''


import numpy as np

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
        
        for poolOut in xrange(len(pools)):
            for unitOut in xrange(len(pools[poolOut].unit)):
                pools[poolOut].unit[unitOut].SynapsesOut = conf.determineSynapses(pools[poolOut].pool + '_' + 
                                                                                  pools[poolOut].unit[unitOut].kind)
                for synapseIn in xrange(len(pools[poolOut].unit[unitOut].SynapsesOut)):
                    conn = float(conf.parameterSet('Con_' + pools[poolOut].pool + '_' 
                                                   + pools[poolOut].unit[unitOut].kind + '_'
                                                   + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0]
                                                   + '@'
                                                   + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][1]
                                                   + '|'
                                                   + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][2],
                                                   '', 0)) / 100.0
                    gmax = float(conf.parameterSet('gmax_' + pools[poolOut].pool + '_'
                                                   + pools[poolOut].unit[unitOut].kind + '_'
                                                   + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0]
                                                   + '@'
                                                   + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][1]
                                                   + '|'
                                                   + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][2],
                                                   '', 0))
                    delay = float(conf.parameterSet('delay_' + pools[poolOut].pool + '_'
                                                    + pools[poolOut].unit[unitOut].kind + '_'
                                                    + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0]
                                                    + '@'
                                                    + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][1]
                                                    + '|'
                                                    + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][2],
                                                    '', 0))
                    declineFactor = float(conf.parameterSet('dec_' + pools[poolOut].pool + '_'
                                                    + pools[poolOut].unit[unitOut].kind + '_'
                                                    + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0]
                                                    + '@'
                                                    + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][1]
                                                    + '|'
                                                    + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][2],
                                                    '', 0))
                    dyn = conf.parameterSet('dyn_' + pools[poolOut].pool + '_'
                                            + pools[poolOut].unit[unitOut].kind + '_'
                                            + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0]
                                            + '@' + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][1]
                                            + '|' + pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][2],
                                            '', 0)
                    for poolIn in xrange(len(pools)):
                        if (pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0].find(pools[poolIn].pool)>=0):
                            for unitIn in xrange(len(pools[poolIn].unit)):
                                if pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][1] == '':
                                    pass
                                else:
                                    for compartmentIn in xrange(len(pools[poolIn].unit[unitIn].compartment)):
                                        if (pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][0] == pools[poolIn].pool + '_' + pools[poolIn].unit[unitIn].kind and pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][1] == pools[poolIn].unit[unitIn].compartment[compartmentIn].kind):
                                            if np.random.uniform(0.0, 1.0) <= conn:
                                                for synapse in xrange(len(pools[poolIn].unit[unitIn].compartment[compartmentIn].SynapsesIn)): 
                                                    if pools[poolIn].unit[unitIn].compartment[compartmentIn].SynapsesIn[synapse].kind == pools[poolOut].unit[unitOut].SynapsesOut[synapseIn][2]:
                                                        if np.isfinite(declineFactor):
                                                            neuronsDistance = np.abs(pools[poolIn].unit[unitIn].position_mm
                                                                                     - pools[poolOut].unit[unitOut].position_mm)
                                                            weight = declineFactor / (declineFactor + neuronsDistance**2)
                                                            print 'Distance' +  str(neuronsDistance) 
                                                            print 'Weight' + str(weight)
                                                        else:
                                                            weight = 1
                                                        pools[poolIn].unit[unitIn].compartment[compartmentIn].SynapsesIn[synapse].addConductance(gmax*weight, delay, dyn)
                                                        pools[poolOut].unit[unitOut].transmitSpikesThroughSynapses.append(pools[poolIn].unit[unitIn].compartment[compartmentIn].SynapsesIn[synapse])                                                            
                                                        pools[poolOut].unit[unitOut].indicesOfSynapsesOnTarget.append(len(pools[poolIn].unit[unitIn].compartment[compartmentIn].SynapsesIn[synapse].gmax_muS) - 1)
                                                        self.numberOfSynapses += 1
                            
        print 'All the ' + str(self.numberOfSynapses) +  ' synapses were built'                   
        