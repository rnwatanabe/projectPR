'''
Created on Oct 26, 2015

@author: root
'''

import math
import numpy as np

def compSynapCond(Gmax, Ron, Roff):
    '''
    Computes the synaptic conductance

    - Input:
        + **Gmax**: the sum of individual conductances of all synapses in 
        the compartment, in \f$\mu\f$S.

        + **Ron**: sum of the fraction of postsynaptic receptors
        that are bound to neurotransmitters of all the individual synapses
        that have neurotransmitters being released (during the pulse).

        + **Roff**: sum of the fraction of postsynaptic receptors
        that are bound to neurotransmitters of all the individual synapses
        that do not have neurotransmitters being released (before and after
        the pulse).

    - Output:
        + the synaptic conductance of all synapses in the compartment,
        in \f$\mu\f$S.

    It is computed by the following formula:

    \f{equation}{
        G = G_{max}(R_{on} + R_{off})
    \f}
    where \f$G\f$ is the synaptic conductance of all synapses in the compartment.
    '''
    return Gmax * (Ron + Roff)

def compRon(Non, rInf, Ron, t0, t, tauOn):
    '''
    Computes the fraction of postsynaptic receptors
    that are bound to neurotransmitters of all the individual synapses
    that have neurotransmitters being released (during the pulse).

    - Inputs:
        + **Non**: sum of the fractions of the individual conductances that are
        receiving neurotransmitter (during pulse) relative to
        the \f$G_{max}\f$ (\f$N_{on}=\limits\sum_{i=1}g_{i_{on}}/G_{max}\f$).

        + **rInf**: the fraction of postsynaptic receptors that
        would be bound to neurotransmitters after an infinite
        amount of time with neurotransmitter being released.

        + **Ron**: sum of the fraction of postsynaptic receptors
        that are bound to neurotransmitters of all the individual synapses
        that have neurotransmitters being released (during the pulse).

        + **t0**: instant that the last spike arrived to the compartment.

        + **t**: current instant, in ms.

        + **tauOn**: Time constant during a pulse, in ms.
        \f$\tau_{on}=\frac{1}{\alpha.T_{max} +\beta}\f$.
    - Outputs:
        + The fraction of postsynaptic receptors
        that are bound to neurotransmitters of all the individual synapses
        that have neurotransmitters being released

    It is computed by the following equation:

    \f{equation}{
        R_{on_{newValue}} = N_{on}r_{\infty}\Bigg[1-\exp\left(-\frac{t-t_0}{\tau_{on}}\right)\Bigg] + R_{on_{oldValue}}\exp\left(-\frac{t-t_0}{\tau_{on}}\right)                 
    \f}
    '''
    return Non * rInf + (Ron - Non * rInf) * math.exp((t0 - t) / tauOn)

def compRoff(Roff, t0, t, tauOff):
    '''
    Computes the fraction of postsynaptic receptors
    that are bound to neurotransmitters of all the individual synapses
    that do not have neurotransmitters being released (before and after
    the pulse).

    - Inputs:
        + **Roff**: sum of the fraction of postsynaptic receptors
        that are bound to neurotransmitters of all the individual synapses
        that do not have neurotransmitters being released (before and after
        the pulse).

        + **t0**: instant that the last spike arrived to the compartment.

        + **t**: current instant, in ms.

        + **tauOff**: time constant after a pulse, in ms.

    + Output:
        + The fraction of postsynaptic receptors
        that are bound to neurotransmitters of all the individual synapses
        that do not have neurotransmitters being released.

    It is computed by the following formula:

    \f{equation}{
        R_{off_{newValue}} = R_{off_{oldValue}}\exp\left(-\frac{t - t0}{\tau_{off}} \right)
    \f}

    '''
    return Roff * math.exp((t0 - t) / tauOff)

def compRiStart(ri, t, ti, tPeak, tauOff):
    '''
    Computes the fraction of bound postsynaptic receptors
    to neurotransmitters in individual synapses when the
    neurotransmitter begin (begin of the pulse).

    - Inputs:
        + **ri**: the fraction of postsynaptic receptors that
        were bound to neurotransmitters at the last state change.

        + **t**: current instant, in ms.

        + **ti**: The instant that the last pulse began.

        + **tPeak**: The duration of the pulse.

        + **tauOff**: Time constant after a pulse, in ms.

    - Output:
        + individual synapse state value.

    It is computed by the following equation:

    \f{equation}{
        r_{i_{newValue}} = r_{i_{oldValue}} \exp\left(\frac{t_i+T_{dur}-t}{\tau_{off}}\right)
    \f}
    '''
    return ri * math.exp((ti + tPeak - t) / tauOff)

def compRiStop(rInf, ri, expFinish):
    '''
    Computes the fraction of bound postsynaptic receptors
    to neurotransmitters in individual synapses when the
    neurotransmitter release stops (the pulse ends).

    - Inputs:
        + **rInf**: the fraction of postsynaptic receptors that
        would be bound to neurotransmitters after an infinite
        amount of time with neurotransmitter being released.

        + **ri**: the fraction of postsynaptic receptors
        that were bound to neurotransmitters at the last
        state change.

        + **expFinish**: Is the value of the exponential at the
        end of the pulse (\f$\exp(T_{dur}/\tau_{on})\f$). It is
        is computed before for computational efficiency.

    - Output:
        + individual synapse state value.

    It is computed by the following equation:

    \f{equation}{
        r_{i_{newValue}} = r_{\infty} + (r_{i_{oldValue}} - r_{\infty}) \exp\left(\frac{T_{dur}}{\tau_{on}}\right)
    \f}
    '''
    return rInf + (ri - rInf) * expFinish

def compRonStart(Ron, ri, synContrib):
    '''
    Incorporates a new conductance to the set of 
    conductances during a pulse.

    - Inputs:
        + **Ron**: sum of the fraction of postsynaptic receptors
        that are bound to neurotransmitters of all the individual synapses
        that have neurotransmitters being released (during the pulse).

        + **ri**: fraction of postsynaptic receptors that are
        bound to neurotransmitters of the individual synapses.

        + **synContrib**: individual conductance constribution 
        to the global synaptic conductance.

    + Output:
        + The new value of the sum of the fraction of postsynaptic receptors
        that are bound to neurotransmitters of all the individual synapses
        that have neurotransmitters being released (during the pulse).
    
    It is computed as:

    \f{equation}{
        R_{on_{newValue}} = R_{on_{oldValue}} + r_iS_{indCont}
    \f}
    '''
    return Ron + ri * synContrib

def compRoffStart(Roff, ri, synContrib):
    '''
    Incorporates a new conductance to the set of
    conductances that are not during a pulse.

    - Inputs:
        + **Roff**: sum of the fraction of postsynaptic receptors
        that are bound to neurotransmitters of all the individual synapses
        that do not have neurotransmitters being released (before and after
        the pulse).

        + **ri**: fraction of postsynaptic receptors that are
        bound to neurotransmitters of the individual synapses.

        + **synContrib**: individual conductance constribution
        to the global synaptic conductance.

    + Output:
        + The new value of the sum of the fraction of postsynaptic receptors
        that are bound to neurotransmitters of all the individual synapses
        that do not have neurotransmitters being released (before and after
        the pulse).

    It is computed as:

    \f{equation}{
        R_{off_{newValue}} = R_{off_{oldValue}} - r_iS_{indCont}
    \f}
    '''
    return Roff - ri * synContrib

def compRonStop(Ron, ri, synContrib):
    '''
    Removes a conductance from the set of
    conductances during a pulse.

    - Inputs:
        + **Ron**: sum of the fraction of postsynaptic receptors
        that are bound to neurotransmitters of all the individual synapses
        that have neurotransmitters being released (during the pulse).

        + **ri**: fraction of postsynaptic receptors that are
        bound to neurotransmitters of the individual synapses.

        + **synContrib**: individual conductance constribution 
        to the global synaptic conductance.

    + Output:
        + The new value of the sum of the fraction of postsynaptic receptors
        that are bound to neurotransmitters of all the individual synapses
        that have neurotransmitters being released (during the pulse).
    
    It is computed as:

    \f{equation}{
        R_{on_{newValue}} = R_{on_{oldValue}} - r_iS_{indCont}
    \f}
    '''
    return Ron - ri * synContrib

def compRoffStop(Roff, ri, synContrib):
    '''
    Removes a conductance from the set of
    conductances that are not during a pulse.

    - Inputs:
        + **Roff**: sum of the fraction of postsynaptic receptors
        that are bound to neurotransmitters of all the individual synapses
        that do not have neurotransmitters being released (before and after
        the pulse).

        + **ri**: fraction of postsynaptic receptors that are
        bound to neurotransmitters of the individual synapses.

        + **synContrib**: individual conductance constribution
        to the global synaptic conductance.

    + Output:
        + The new value of the sum of the fraction of postsynaptic receptors
        that are bound to neurotransmitters of all the individual synapses
        that do not have neurotransmitters being released (before and after
        the pulse).

    It is computed as:

    \f{equation}{
        R_{off_{newValue}} = R_{off_{oldValue}} + r_iS_{indCont}
    \f}
    '''
    return Roff + ri * synContrib

class Synapse(object):
    '''
    Implements the synapse model from Destexhe (1994)
    using the computational method from Lytton (1996).
    '''
    def __init__(self, conf, pool, index, compartment, kind, neuronKind):
        '''
        Constructor

        - Input:
            + **conf**: Configuration object with the simulation parameters.

            + **pool**: string with identification of the pool to which 
            the synapse belongs.

            + **index**: integer identification of the unit in the pool.

            + **compartment**: integer identification of the compartment of the unit 
            where the synapse is.

            + **kind**: string with the type of synapse. For now, it can be *excitatory*. 

            + **neuronKind**:
        '''
        self.pool = pool
        self.kind = kind
        self.neuronKind = neuronKind

        self.EqPot_mV = float(conf.parameterSet('EqPotSyn_' + pool + '_' + self.neuronKind + '_' + self.kind, pool, index))
        self.alpha_ms1 = float(conf.parameterSet('alphaSyn_' + self.kind + '_' + pool + '_'  + self.neuronKind, pool, index))
        self.beta_ms1 = float(conf.parameterSet('betaSyn_' + self.kind + '_' + pool + '_'  + self.neuronKind, pool, index))
        self.Tmax_mM = float(conf.parameterSet('TmaxSyn_' + self.kind + '_' + pool + '_'  + self.neuronKind, pool, index))
        ## Pulse duration, in ms.
        self.tPeak_ms = float(conf.parameterSet('tPeakSyn_' + self.kind + '_' + pool + '_' + self.neuronKind, pool, index))

        self.gmax_muS = np.array([])
        self.delay_ms = np.array([])
        self.dynamics = []

        ## The sum of individual conductances of all synapses in 
        ## the compartment, in \f$\mu\f$S (\f$G_{max} = \limits\sum_{i=1}^Ng_i\f$).
        self.gMaxTot_muS = 0
        self.numberOfIncomingSynapses = 0

        ## The fraction of postsynaptic receptors
        ## that would be bound to neurotransmitters
        ## after an infinite amount of time with
        ## neurotransmitter being released.
        self.rInf = (self.alpha_ms1 * self.Tmax_mM) / (self.alpha_ms1 * self.Tmax_mM + self.beta_ms1)
        ## Time constant during a pulse, in ms.
        ## \f$\tau_{on}=\frac{1}{\alpha.T_{max} +\beta}\f$
        self.tauOn = 1.0 / (self.alpha_ms1 * self.Tmax_mM + self.beta_ms1)
        ## Time constant after a pulse, in ms.
        ## \f$\tau_{off}=\frac{1}{\beta}\f$
        self.tauOff = 1.0 / self.beta_ms1
        ## Is the value of the exponential at the
        ## end of the pulse. It is computed as
        ## \f$\exp(T_{dur}/\tau_{on})\f$.
        self.expFinish = math.exp(- self.tPeak_ms / self.tauOn)

        ## Sum of the fractions of the individual conductances that are
        ## receiving neurotransmitter (during pulse) relative to
        ## the \f$G_{max}\f$. (\f$N_{on}=\limits\sum_{i=1}g_{i_{on}}/G_{max}). 
        self.Non = 0.0
        ## Sum of the fraction of postsynaptic receptors
        ## that are bound to neurotransmitters of all the individual synapses
        ## that have neurotransmitters being released (during the pulse). 
        self.Ron = 0.0
        ## Sum of the fraction of postsynaptic receptors
        ## that are bound to neurotransmitters of all the individual synapses
        ## that do not have neurotransmitters being released (before and after
        ## the pulse).
        self.Roff = 0.0
        ## Instant that the last spike arrived to the compartment.
        self.t0 = 0.0

        self.conductanceState = np.array([])
        self.tBeginOfPulse = np.array([])
        self.tEndOfPulse = np.array([])
        ## List with the fractions of postsynaptic receptors
        ## that are bound to neurotransmitters of the individual 
        ## synapses.
        self.ri = np.array([])
        ## List with the instants of spike arriving at each 
        ## conductance, in ms.
        self.ti = np.array([])
        ## List of individual conductance constribution 
        ## to the global synaptic conductance
        ## (\f$S_{indCont} = \frac{g_{i_{max}}{G_{max}}\f$).
        self.synContrib = np.array([])
        self.startDynamicFunction = []
        self.stopDynamicFunction = []


    def computeCurrent(self, t, V_mV):
        '''
        Computes the current on the compartment due to the synapse.

        - Inputs:
            + **t**: current instant, in ms.

            + **V_mV**: membrane potential of the compartment that the
            synapse belongs, in mV.

        - Output:
            + The current on the compartment due to the synapse.
        '''
        if len(self.tEndOfPulse) == 0:
            self.tBeginOfPulse = np.ones_like(self.gmax_muS,
            dtype=float) * float("-inf")
            self.tEndOfPulse = np.ones_like(self.gmax_muS,
            dtype=float) * float("-inf")
            self.conductanceState = np.zeros_like(self.gmax_muS,
                dtype=int)
            self.ri = np.zeros_like(self.gmax_muS, dtype=float)
            self.ti = np.zeros_like(self.gmax_muS, dtype=float)
            self.synContrib = self.gmax_muS / self.gMaxTot_muS
            for dyn in xrange(len(self.dynamics)):
                if self.dynamics[dyn] == 'None':
                    self.startDynamicFunction.append(self.startConductanceNone)
                    self.stopDynamicFunction.append(self.stopConductanceNone)
                else:
                    self.startDynamicFunction.append(self.startConductanceDynamics)
            self.computeCurrent = self.computeCurrent2

        return self.computeConductance(t) * (self.EqPot_mV - V_mV)

    def computeCurrent2(self, t, V_mV):
        '''
        The same function of computeCurrent. It overrides this function for
        computational efficiency.

        - Inputs:
            + **t**: current instant, in ms.

            + **V_mV**: membrane potential of the compartment that the
            synapse belongs, in mV.
        '''
        return self.computeConductance(t) * (self.EqPot_mV - V_mV)

    def computeConductance(self, t):
        '''

        - Inputs:
            + **t**: current instant, in ms.
        '''
        self.Ron = compRon(self.Non, self.rInf, self.Ron, self.t0,
                        t, self.tauOn)
        self.Roff = compRoff(self.Roff, self.t0, t, self.tauOff)

        self.startConductanceNone(t,
            np.where(np.abs(t-self.tBeginOfPulse < 1e-3))[0])
        self.stopConductanceNone(t,
            np.where(np.abs(t-self.tEndOfPulse) < 1e-3)[0])

        return compSynapCond(self.gMaxTot_muS, self.Ron, self.Roff)

    def startConductanceNone(self, t, idxBeginPulse):
        '''
        - Inputs:
            + **t**: current instant, in ms.

            + **idxBeginPulse**: vector with the indices of the conductances
                that the pulse begin at time **t**.
        '''
        for synapseNumber in idxBeginPulse:
            if self.conductanceState[synapseNumber] == 0:
                self.ri.itemset(synapseNumber,
                    compRiStart(self.ri.item(synapseNumber),
                    t, self.ti.item(synapseNumber), self.tPeak_ms, self.tauOff))
                self.ti.itemset(synapseNumber, t)
                self.Ron = compRonStart(self.Ron, self.ri.item(synapseNumber),
                            self.synContrib.item(synapseNumber))
                self.Roff = compRoffStart(self.Roff,
                            self.ri.item(synapseNumber),
                            self.synContrib.item(synapseNumber))
                self.Non += self.synContrib.item(synapseNumber)
                self.t0 = t
                self.conductanceState.itemset(synapseNumber, 1)

            self.tEndOfPulse.itemset(synapseNumber, t + self.tPeak_ms)
            self.tBeginOfPulse.itemset(synapseNumber, -1000000)

    def startConductanceDynamics(self, t, synapsesNumber):
        '''
        '''
        pass

    def stopConductanceNone(self, t, idxEndPulse):
        '''
        - Inputs:
            + **t**: current instant, in ms.

            + **idxEndPulse**: vector with the indices of the conductances
                that the pulse end at time **t**.
        '''
        for synapseNumber in idxEndPulse:
            self.ri.itemset(synapseNumber,
                compRiStop(self.rInf, self.ri.item(synapseNumber),
                            self.expFinish))
            self.t0 = t
            self.Ron = compRonStop(self.Ron, self.ri.item(synapseNumber),
                    self.synContrib.item(synapseNumber))
            self.Roff = compRoffStop(self.Roff, self.ri.item(synapseNumber),
                        self.synContrib.item(synapseNumber))
            self.Non -= self.synContrib.item(synapseNumber)
            self.tEndOfPulse.itemset(synapseNumber, -10000)
            self.conductanceState.itemset(synapseNumber, 0)

    def stopConductanceDynamics(self, t, synapseNumber):
        '''
        
        '''
        pass

    def receiveSpike(self, t, synapseNumber):
        '''

        - Inputs:
            + **t**:

            + **synapseNumber**:
        '''

        self.tBeginOfPulse[synapseNumber] = t + self.delay_ms[synapseNumber]

    def addConductance(self, gmax, delay, dynamics):
        '''
        Adds a synaptic conductance to the compartment. As the computation 
        is performed once for each compartment at each time step, the data of 
        each individual synapse is integrate in a big synapse.

        - Inputs:
            + **gmax**: the maximum conductance of the individual 
            synase, in \f$\mu\f$S.

            + **delay**: transmission delay between the transmitter of the
            spike and the receiver compartment, in ms.

            + **dynamics**: type of the synapse dynamics. For now it 
            can be *None*.

        '''
        self.gMaxTot_muS += gmax
        self.numberOfIncomingSynapses += 1
        self.gmax_muS = np.append(self.gmax_muS, gmax)
        self.delay_ms = np.append(self.delay_ms, delay)
        self.dynamics.append(dynamics)
