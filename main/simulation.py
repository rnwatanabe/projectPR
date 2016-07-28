'''
Created on Sep 30, 2015

@author: root
'''


from Configuration import Configuration
import numpy as np
import matplotlib.pyplot as plt
import time
from MotorUnitPool import MotorUnitPool
import cProfile
import profile
from NeuralTract import NeuralTract
from SynapsesFactory import SynapsesFactory



def simulador():     

    conf = Configuration('confTest.rmto')

    pools = []
    pools.append(MotorUnitPool(conf, 'SOL'))
    pools.append(NeuralTract(conf, 'CM_ext'))
    Syn = SynapsesFactory(conf, pools)
    del Syn

    t = np.arange(0.0, conf.simDuration_ms, conf.timeStep_ms)

    tic = time.clock()
    for i in xrange(0,len(t)): 
        pools[1].atualizePool(t[i])
        pools[0].atualizeMotorUnitPool(t[i])
    toc = time.clock()
    print str(toc - tic) + ' seconds'

    pools[1].listSpikes()
    plt.figure()
    plt.plot(pools[1].poolTerminalSpikes[:, 0],
        pools[1].poolTerminalSpikes[:, 1]+1, '.')

    pools[0].listSpikes()
    plt.figure()
    plt.plot(pools[0].poolTerminalSpikes[:, 0],
        pools[0].poolTerminalSpikes[:, 1]+1, '.')

    plt.figure()
    plt.plot(t, pools[0].Muscle.force, '-')

    plt.show()

if __name__ == '__main__':
    
    #cProfile.run('simulador()', sort = 'calls')
    np.__config__.show()
    simulador()
    
    