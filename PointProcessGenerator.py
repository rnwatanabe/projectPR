'''
Created on Oct 23, 2015

@author: root
'''
import numpy as np
import math
import random





def  gammaPoint(GammaOrder, GammaOrderInv):
    '''
    Generates a number according to a Gamma Distribution with an integer order **GammaOrder**.

    - Inputs:
        + **GammaOrder**: integer order of the Gamma distribution.

        + **GammaOrderInv**:  inverse of the GammaOrder. This is necessary
                                for computational efficiency.

    - Outputs:
        + The number generated from the Gamma distribution.

    The number is generated according to:

    \f{equation}{
        \Gamma = -\frac{1}{\lambda}\ln(\limits\prod_{i=1}^{\lambda} U(0,1))
    \f}
    where \f$\lambda\f$ is the order of the Gamma distribution and U(a,b) is
    a uniform distribution from a to b.

    '''
    aux = np.prod(np.random.uniform(0.0, 1.0, GammaOrder))

    return - GammaOrderInv * math.log(aux)

class PointProcessGenerator(object):
    '''
    Generator of point processes.
    '''
    
    def __init__(self, GammaOrder, index):
        '''
        Constructor

        - Inputs:
            + **GammaOrder**: integer order of the Gamma distribution.

            + **index**: integer corresponding to the unit order in the pool.
        '''
        ## Integer order of the Gamma distribution. 
        ## Gamma order 1 is Poisson process and order 10 is a Gaussian process.
        self.GammaOrder = int(GammaOrder);
        ## Inverse of the GammaOrder. This is necessary
        ## for computational efficiency.
        self.GammaOrderInv = 1.0 / GammaOrder
        ## Integer corresponding to the unit order in the pool to which this
        ## generator is associated.
        self.index = index

        ## Auxiliary variable cummulating a value that indicates
        ## whether there will be a new spike or not.
        self.y = 0.0

        ## Spike threshold. When the auxiliary variable y reaches the value
        ## of threshold, there is a new spike.
        self.threshold = gammaPoint(self.GammaOrder, self.GammaOrderInv)
        ## List of spike instants of the generator.
        self.points = []

    def atualizeGenerator(self, t, firingRate):
        '''

        - Inputs:
            + **t**: current instant, in ms.

            + **firingRate**: instant firing rate, in spikes/s.
        '''
        self.y += firingRate
        if self.y >= self.threshold and t != 0:
            self.points.append([t, self.index])
            self.y = 0.0
            self.threshold = gammaPoint(self.GammaOrder, self.GammaOrderInv)
