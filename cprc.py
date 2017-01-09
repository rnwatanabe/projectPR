#!/usr/bin/env python2
import numpy as np
from MotorUnit import MotorUnit

import dill
from mpi4py import MPI
MPI.pickle.dumps = dill.dumps
MPI.pickle.loads = dill.loads 

comm = MPI.Comm.Get_parent()
common_comm=comm.Merge(True)
rank = common_comm.Get_rank ()


t = None
while True:
    t = common_comm.bcast (t, root = 0)
    unit = common_comm.recv(source=0,tag=rank)
    #print type(unit)
    #print vars(unit[0])
    for i in unit: i.atualizeMotorUnit(t)
    common_comm.send(unit,dest=0,tag=rank)
