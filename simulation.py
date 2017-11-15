import cProfile
import profile
import time


import matplotlib.pyplot as plt
import numpy as np

from Configuration import Configuration
from MotorUnitPool import MotorUnitPool
from InterneuronPool import InterneuronPool
from NeuralTract import NeuralTract
from SynapsesFactory import SynapsesFactory

# TODO
from multiprocessing import Process

def initialize():
    conf = Configuration('confTest.rmto')

    pools = dict()
    pools[0] = MotorUnitPool(conf, 'SOL')
    #pools[1] = NeuralTract(conf, 'CMExt')
    #pools[2] = InterneuronPool(conf, 'RC', 'ext')

    Syn = SynapsesFactory(conf, pools)

    t = np.arange(0.0, conf.simDuration_ms, conf.timeStep_ms)

    return pools, t

def simulator(pools, t, rank, nbrProcesses):

    # TODO
    processChunk = len(pools[0].unit) / (nbrProcesses)
    processUnits = range(rank * processChunk, rank * processChunk + processChunk)

    dendV = np.zeros_like(t)
    somaV = np.zeros_like(t)

    tic = time.clock()
    for i in xrange(0, len(t)):
        for j in xrange(len(pools[0].unit)):
            pools[0].unit[j].iInjected[1] = 10
        #pools[1].atualizePool(t[i])
        pools[0].atualizeMotorUnitPool(t[i], processUnits)
        #pools[2].atualizeInterneuronPool(t[i])
        somaV[i] = pools[0].unit[rank].v_mV[0] 
    toc = time.clock()
    print str(toc - tic) + ' seconds'

    pools[0].listSpikes()
    pools[1].listSpikes()
    
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

    #plt.figure()
    #plt.plot(t, dendV, '-')

    #plt.figure()
    #plt.plot(t, somaV, '-')
    
if __name__ == '__main__':

    pools, t = initialize()

    # TODO
    nbrProcesses = 2
    for i in xrange(nbrProcesses):
        p = Process(target=simulator, args=(pools, t, i, nbrProcesses))
        p.start()
    print "==="
    print "Main process awaits with join"
    print "==="
    p.join()
    #simulator(pools, t)

    plt.show()
