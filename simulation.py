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
    
    pools = dict()
    pools[0] = MotorUnitPool(conf, 'SOL')
    pools[1] = NeuralTract(conf, 'CMExt')

    #pools.append(InterneuronPool(conf, 'RC'))

    #ankle = jointAnkleForceTask(conf, pools)
    Syn = SynapsesFactory(conf, pools)
    del Syn
    
    t = np.arange(0.0, conf.simDuration_ms, conf.timeStep_ms)

    dendV = np.zeros_like(t)
    somaV = np.zeros_like(t)
    nodeV1 = np.zeros_like(t)
    nodeV2 = np.zeros_like(t)

    tic = time.clock()
    for i in xrange(0, len(t)):
        #ankle.atualizeAnkle(t[i], 0)
        #for j in xrange(len(pools[0].unit)):
        #    pools[0].unit[j].iInjected[1] = 10
        pools[1].atualizePool(t[i])
        pools[0].atualizeMotorUnitPool(t[i])
        dendV[i] = pools[0].unit[2].v_mV[0]
        somaV[i] = pools[0].unit[2].v_mV[1] 
        #nodeV1[i] = pools[0].unit[2].v_mV[3]
        #nodeV2[i] = pools[0].unit[2].v_mV[31]
        #pools[3].atualizePool(t[i])
        #pools[2].atualizeInterneuronPool(t[i])
    toc = time.clock()
    print str(toc - tic) + ' seconds'

    pools[0].listSpikes()
    pools[1].listSpikes()
    #pools[2].listSpikes()
    
    np.savetxt('../results/MNspikes_noRC.txt', pools[0].poolTerminalSpikes)
    #np.savetxt('../results/NTspikes_noRC.txt', pools[1].poolTerminalSpikes)
    #np.savetxt('../results/RCspikes.txt', pools[2].poolSomaSpikes)
    np.savetxt('../results/SOLforce_noRC.txt', pools[0].Muscle.force)


    plt.figure()
    plt.plot(pools[1].poolTerminalSpikes[:, 0],
             pools[1].poolTerminalSpikes[:, 1]+1, '.')
    
    
    
    
    plt.figure()
    plt.plot(pools[0].poolTerminalSpikes[:, 0],
             pools[0].poolTerminalSpikes[:, 1]+1, '.')
    '''
    plt.figure()
    plt.plot(pools[0].poolLastCompSpikes[:, 0],
             pools[0].poolLastCompSpikes[:, 1]+1, '.')        

    plt.figure()
    plt.plot(pools[0].poolTerminalSpikes[:, 0],
             pools[0].poolTerminalSpikes[:, 1]+1, '.')         
    '''             
    '''         
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

    #print 'M = ' + str(np.mean(pools[0].Muscle.force[int(1000/conf.timeStep_ms):-1]))
    #print 'SD = ' + str(np.std(pools[0].Muscle.force[int(1000/conf.timeStep_ms):-1]))

    plt.figure()
    plt.plot(t, dendV, '-')

    plt.figure()
    plt.plot(t, somaV, '-')

    

    plt.figure()
    plt.plot(t, nodeV1, '-')

    plt.figure()
    plt.plot(t, nodeV2, '-')
    
    '''
    plt.figure()
    plt.plot(t, pools[0].Muscle.length_m, '-')

    plt.figure()
    plt.plot(t, ankle.ankleAngle_rad, '-')
    '''

    pools[0].getMotorUnitPoolEMG()

    plt.figure()
    plt.plot(t, pools[0].emg, '-')


    plt.figure()
    plt.plot(t, pools[0].unit[0].nerveStimulus_mA, '-')

    
if __name__ == '__main__':

    #cProfile.run('simulator()', sort = 'tottime')
    
    np.__config__.show()
    
    
    simulator()
    
    plt.show()