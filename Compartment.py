'''
Author: Renato Naville Watanabe
'''
from ChannelConductance import ChannelConductance

from Synapse import Synapse
import math


def calcGLeak(area, specificRes):
    '''
    Computes the leak conductance of the compartment.

    - Input:
        + **area**: area of the compartment in cm\f$^2\f$.
        + **specificRes**: specific resistance of the compartment
        in \f$\Omega.cm^2\f$.

    - Output:
        + Leak conductance in MS.

    It is compute according to the following formula:

    \f{equation}{
        g = 10^6 . \frac{A}{\rho}
    \f}
    where \f$A\f$ is the compartment area [cm\f$^2\f$], \f$\rho\f$ is
    the specific resistance [\f$\Omega.cm^2\f$] and \f$g\f$ is the
    compartment conductance [MS].
    '''
    return (1e6 * area) / specificRes

class Compartment(object):
    '''
    Class that implements a neural compartment. For now it is implemented
    *dendrite* and *soma*.
    '''


    def __init__(self, kind, conf, pool, index, neuronKind):
        '''
        Constructor

        - Inputs:
            + **kind**: The kind of compartment. For now, it can be *soma* or *dendrite*.

            + **conf**: Configuration object with the simulation parameters.

            + **pool**: string with Motor unit pool to which the motor unit belongs.

            + **index**: integer corresponding to the motor unit order in the pool, according to 
            the Henneman's principle (size principle).

            + **neuronKind**: string with the type of the motor unit. It can be *S* (slow), *FR* (fast and resistant), 
            and *FF* (fast and fatigable).
        '''
        
        ## List of ChannelConductance objects in the Compartment.
        self.Channels = []
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
        self.length_mum = float(conf.parameterSet('l_' + kind, pool, index))
        ## Diameter of the compartment, in \f$\mu\f$m.
        self.diameter_mum = float(conf.parameterSet('d_' + kind, pool, index))
        area_cm2 = float(self.length_mum * math.pi * self.diameter_mum * 1e-8)
        specifRes_Ohmcm2 = float(conf.parameterSet('res_' + kind, pool, index))
        ## Capacitance of the compartment, in nF.
        self.capacitance_nF = float(float(conf.parameterSet('membCapac', pool, index)) * area_cm2 * 1e3)
        print self.capacitance_nF
        ## Leak conductance of the compartment, in MS.
        self.gLeak = calcGLeak(area_cm2, specifRes_Ohmcm2)

        if (kind == 'soma'):
            self.Channels.append(ChannelConductance('Kf', conf, area_cm2, pool, neuronKind, index))
            self.Channels.append(ChannelConductance('Ks', conf, area_cm2, pool, neuronKind, index))
            self.Channels.append(ChannelConductance('Na', conf, area_cm2, pool, neuronKind, index))
        elif (kind == 'dendrite'):
            pass
        
        ## Integer with the number of ionic channels.
        self.numberChannels = len(self.Channels)

    def computeCurrent(self, t, V_mV):
        '''
        Computes the active currents of the compartment. Active currents are the currents from the ionic channels
        and from the synapses.

        - Inputs:
            + **t**: current instant, in ms.

            + **V_mV**: membrane potential, in mV.
        '''
        I = 0
        if self.numberChannels != 0:
            for i in xrange(0, self.numberChannels): I += self.Channels[i].computeCurrent(t, V_mV)
        if self.SynapsesIn[0].numberOfIncomingSynapses:
            I += self.SynapsesIn[0].computeCurrent(t, V_mV)
        if self.SynapsesIn[1].numberOfIncomingSynapses:
            I += self.SynapsesIn[1].computeCurrent(t, V_mV)

        return I