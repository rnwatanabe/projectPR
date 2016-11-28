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
from NeuralTract import NeuralTract
from SynapsesFactory import SynapsesFactory
from jointAnkleForceTask import jointAnkleForceTask


def simulator():

    conf = Configuration('confTest.rmto')

    pools = []
    pools.append(MotorUnitPool(conf, 'SOL'))
    pools.append(NeuralTract(conf, 'CM_ext'))
    #pools.append(InterneuronPool(conf, 'RC'))
    #ankle = jointAnkleForceTask(conf, pools)
    Syn = SynapsesFactory(conf, pools)
    del Syn
    
    t = np.arange(0.0, conf.simDuration_ms, conf.timeStep_ms)

    tic = time.clock()
    for i in xrange(0, len(t)-1):
        #ankle.atualizeAnkle(t[i], 0)
        pools[1].atualizePool(t[i])
        pools[0].atualizeMotorUnitPool(t[i])
        #pools[3].atualizePool(t[i])
        #pools[2].atualizeInterneuronPool(t[i])
    toc = time.clock()
    print str(toc - tic) + ' seconds'

    pools[0].listSpikes()
    pools[1].listSpikes()
    #pools[2].listSpikes()
    
    np.savetxt('../results/MNspikes_noRC.txt', pools[0].poolTerminalSpikes)
    np.savetxt('../results/NTspikes_noRC.txt', pools[1].poolTerminalSpikes)
    #np.savetxt('../results/RCspikes.txt', pools[2].poolSomaSpikes)
    np.savetxt('../results/SOLforce_noRC.txt', pools[0].Muscle.force)

    
    plt.figure()
    plt.plot(pools[1].poolTerminalSpikes[:, 0],
             pools[1].poolTerminalSpikes[:, 1]+1, '.')

    
    
    '''
    plt.figure()
    plt.plot(pools[0].poolTerminalSpikes[:, 0],
             pools[0].poolTerminalSpikes[:, 1]+1, '.')

    plt.figure()
    plt.plot(pools[2].poolSomaSpikes[:, 0],
             pools[2].poolSomaSpikes[:, 1]+1, '.')
    '''
    '''
    print pools[0].Muscle.maximumActivationForce

    plt.figure()
    plt.plot(t, pools[0].Muscle.activationTypeI, '-')
    
    plt.figure()
    plt.plot(t, pools[0].Muscle.tendonForce_N, '-')
    '''
    
    plt.figure()
    plt.plot(t, pools[0].Muscle.force, '-')
    
    '''
    plt.figure()
    plt.plot(t, pools[0].Muscle.length_m, '-')

    plt.figure()
    plt.plot(t, ankle.ankleAngle_rad, '-')
    '''

    plt.show()
    
if __name__ == '__main__':

    #cProfile.run('simulator()', sort = 'calls')
    np.__config__.show()
    simulator()
