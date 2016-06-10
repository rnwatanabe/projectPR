'''
Created on Oct 23, 2015

@author: root
'''
import numpy as np
import math
import random





def  gammaPoint(GammaOrder):
        
        aux = 1.0
        
        for i in xrange(0, GammaOrder):
            aux *= np.random.uniform(0.0, 1.0)
        
        return -(1.0/GammaOrder)*math.log(aux)
    
    

class PointProcessGenerator(object):
    '''
    classdocs
    '''
    
    def __init__(self, GammaOrder, index):
        '''
        Constructor
        '''
        
        self.GammaOrder = int(GammaOrder);  # Gamma order 1 is Poisson process
        self.GammaOrderInv = 1.0 / GammaOrder
        self.index = index
        
        self. y = 0.0
        
        self.threshold = gammaPoint(self.GammaOrder)
        self.points = []
     
    
    def atualizeGenerator(self,  t, FR):
       
        self.y += FR
        
        if (self.y >=self.threshold and t != 0):
            self.points.append([t, self.index])
            self.y = 0.0
            self.threshold = gammaPoint(self.GammaOrder)
     
    
   
        
        
        