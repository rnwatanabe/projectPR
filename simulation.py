import cProfile
import profile
import time


import matplotlib.pyplot as plt
import numpy as np

from Configuration import Configuration
from MotorUnitPool import MotorUnitPool
from InterneuronPool import InterneuronPool
from NeuralTract import NeuralTract
from MPSynapsesFactory import SynapsesFactory

def initialize():
    conf = Configuration('confTest.rmto')

    pools = dict()
    pools[0] = MotorUnitPool(conf, 'SOL')
    pools[1] = NeuralTract(conf, 'CMExt')
    pools[2] = InterneuronPool(conf, 'RC', 'ext')

    for i in xrange(0,len(pools[0].unit)):
        pools[0].unit[i].createStimulus()

    Syn = SynapsesFactory(conf, pools)

    t = np.arange(0.0, conf.simDuration_ms, conf.timeStep_ms)

    return pools, t

def simulator(pools, t):

    somaV = np.zeros_like(t)

    for i in xrange(0, len(t)):
        #for j in xrange(len(pools[0].unit)):
        #    pools[0].iInjected[0] = 10
        pools[1].atualizePool(t[i])
        pools[0].atualizeMotorUnitPool(t[i]) 
        pools[2].atualizeInterneuronPool(t[i])
        # Stores membrane potential of the first RC
        somaV[i] = pools[2].v_mV[1]
        # Stores membrane potential of the soma of the first unit of the pool
        #somaV[i] = pools[0].v_mV[1]
        # Stores membrane potential of the dendrite of the second unit of the pool
        #somaV[i] = pools[0].v_mV[2]

    pools[0].listSpikes()
    
    #plt.figure()
    #plt.plot(pools[0].poolSomaSpikes[:, 0],
    #         pools[0].poolSomaSpikes[:, 1]+1, '.')

    #plt.figure()
    #plt.plot(pools[1].poolTerminalSpikes[:, 0],
    #         pools[1].poolTerminalSpikes[:, 1]+1, '.')
    #
    #
    #plt.figure()
    #plt.plot(pools[0].poolSomaSpikes[:, 0],
    #         pools[0].poolSomaSpikes[:, 1]+1, '.')

    #plt.figure()
    #plt.plot(t, pools[0].Muscle.force, '-')

    plt.figure()
    plt.plot(t, somaV, '-')
    #plt.plot(t, dendV, '-')

    plt.figure()
    plt.plot(t, pools[0].unit[0].nerveStimulus_mA)
    
if __name__ == '__main__':

    tic = time.time()
    pools, t = initialize()

    simulator(pools, t)
    toc =time.time()

    plt.show()
    print str(toc - tic) + ' seconds'
