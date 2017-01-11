#!/usr/bin/env /home/intel/intelpython27/bin/python
import sys
sys.path.insert(0, '..')
import time
import matplotlib.pyplot as plt
#matplotlib inline  
import numpy as np

from Configuration import Configuration
from MotorUnitPool2 import MotorUnitPool
from InterneuronPool import InterneuronPool
from NeuralTract import NeuralTract
from SynapsesFactory import SynapsesFactory

# Initialze MPI part
# Failed first try
#size = comm.Get_size ()
#rank = comm.Get_rank ()

conf = Configuration('confTest.rmto')
#conf.simDuration_ms = 10 # Here I change simulation duration without changing the Configuration file.

# Time vector for the simulation
t = np.arange(0.0, conf.simDuration_ms, conf.timeStep_ms)
# Failed first try
#totalSize = (conf.simDuration_ms - 0.0)
#procSize = totalSize / size
#t = np.arange(rank * procSize, (rank + 1) * procSize, conf.timeStep_ms)

membPotential = np.zeros_like(t, dtype = 'd')
dendV = np.zeros_like(t)
somaV = np.zeros_like(t)

pools = dict()
pools[0] = MotorUnitPool(conf, 'SOL')
#pools[1] = NeuralTract(conf, 'CMExt')
#pools[2] = InterneuronPool(conf, 'RC')
Syn = SynapsesFactory(conf, pools)

tic = time.clock()
for i in xrange(0, len(t)-1):
    for j in xrange(len(pools[0].unit)):
        pools[0].unit[j].iInjected[1] = 10
    #pools[1].atualizePool(t[i]) # NeuralTract
    pools[0].atualizeMotorUnitPool(t[i]) # MN pool
    dendV[i] = pools[0].unit[2].v_mV[0]
    somaV[i] = pools[0].unit[2].v_mV[1]
    #pools[3].atualizePool(t[i]) # RC synaptic Noise
    #pools[2].atualizeInterneuronPool(t[i]) # RC pool
toc = time.clock()
print str(toc - tic) + ' seconds'

pools[0].listSpikes()
pools[1].listSpikes()
#pools[2].listSpikes()
plt.figure()
plt.plot(pools[1].poolTerminalSpikes[:, 0],
        pools[1].poolTerminalSpikes[:, 1]+1, '.')
plt.figure()
plt.plot(pools[0].poolSomaSpikes[:, 0],
            pools[0].poolSomaSpikes[:, 1]+1, '.')
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


'''
plt.figure()
plt.plot(t, pools[0].Muscle.length_m, '-')

plt.figure()
plt.plot(t, ankle.ankleAngle_rad, '-')
'''

plt.show()
