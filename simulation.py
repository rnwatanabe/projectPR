'''
Created on Sep 30, 2015

@author: root
'''


import cProfile
import profile
import time


import matplotlib.pyplot as plt
import numpy as np
from Configuration import Configuration
from MotorUnitPool import MotorUnitPool
from InterneuronPool import InterneuronPool
from AfferentPool import AfferentPool
from NeuralTract import NeuralTract
from SynapsesFactory import SynapsesFactory
from jointAnkleForceTask import jointAnkleForceTask


def simulator():

    conf = Configuration('confTest.rmto')
    pools = dict()
    pools[0] = MotorUnitPool(conf, 'SOL')
    pools[1] = NeuralTract(conf, 'CMExt')
    pools[2] = AfferentPool(conf, 'Ia','SOL')

    #pools.append(InterneuronPool(conf, 'RC', 'ext'))

    #ankle = jointAnkleForceTask(conf, pools)
    Syn = SynapsesFactory(conf, pools)
    del Syn

    t = np.arange(0.0, conf.simDuration_ms, conf.timeStep_ms)

    dendV = np.zeros_like(t)
    somaV = np.zeros_like(t)
    internodeV = np.zeros_like(t)
    nodeV = np.zeros_like(t)

    tic = time.time()
    for i in xrange(0, len(t)):
        #for j in xrange(len(pools[0].unit)):
        #    pools[0].unit[j].iInjected[1] = 10
        pools[1].atualizePool(t[i])
        pools[0].atualizeMotorUnitPool(t[i])
        pools[2].atualizeAfferentPool(t[i], pools[0].spindle.IaFR_Hz)
        dendV[i] = pools[0].unit[2].v_mV[0]
        somaV[i] = pools[0].unit[2].v_mV[1] 
    toc = time.time()
    print str(toc - tic) + ' seconds'

    pools[0].listSpikes()
    pools[1].listSpikes()
    #pools[2].listSpikes()
      
    np.savetxt('../results/MNspikes_noRC.txt', pools[0].poolTerminalSpikes)
    #np.savetxt('../results/NTspikes_noRC.txt', pools[1].poolTerminalSpikes)
    #np.savetxt('../results/RCspikes.txt', pools[2].poolSomaSpikes)
    np.savetxt('../results/SOLforce_noRC.txt', pools[0].Muscle.force)
    '''
    plt.figure()
    plt.plot(pools[1].poolTerminalSpikes[:, 0],
             pools[1].poolTerminalSpikes[:, 1]+1, '.')
    
    
    plt.figure()
    plt.plot(pools[0].poolSomaSpikes[:, 0],
             pools[0].poolSomaSpikes[:, 1]+1, '.')

    plt.figure()
    plt.plot(t, pools[0].Muscle.force, '-')

    plt.figure()
    plt.plot(t, dendV, '-')

    plt.figure()
    plt.plot(t, somaV, '-')
    '''
if __name__ == '__main__':

    #cProfile.run('simulator()', sort = 'cumtime')
    
    np.__config__.show()
    
    
    simulator()
    '''
    plt.show()
    '''
