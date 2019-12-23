###
# Helpful tools for analysis.
###
import numpy as np
import pandas as pd

from math import *

import torch
from torch.autograd import Variable, Function
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

# Convert variable to Torch variables
def tovar(x):
    return Variable(torch.FloatTensor(x).cuda(), requires_grad = False)

def toivar(x):
    return Variable(torch.LongTensor(x).cuda(), requires_grad = False)

class PlanetLoader():
	def __init__(self,fname="NEA_radmasstpersmasspnum.csv"):
            # Fields: planet radius [R_jupiter], planet mass [M_jupiter], number of planets in system, stellar mass [Msun],
            # planet orbital period [days], planet equilibrium temperature [K]
            self.fields = ["pl_name", "pl_radj", "pl_bmassj", "pl_pnum", "st_mass", "pl_orbper", "pl_eqt"]

            # Load in the planet dataset
            data = pd.read_csv(fname)
            sdata = np.array(data[self.fields[1:]].iloc[:,:])

            # 2018.08.09: numpy.array doesn't allow different data types, but planet names are useful when testing data. Create seperate arrays.
            ndata = np.array(data[self.fields[0]].iloc[:])

            # We will randomize the data order and then take the first 400 planets as the training set, the rest as the test set
            # Use a fixed random seed for train-test split to ensure replication
            self.rs = np.random.RandomState(12345)
            dataidx = self.rs.permutation(sdata.shape[0])

            # Divide up training and testing data
            self.train_data = sdata[dataidx[0:400]]
            self.test_data = sdata[dataidx[400:]]
            self.data = sdata

            # name arrays
            self.train_names = ndata[dataidx[0:400]]
            self.test_names = ndata[dataidx[400:]]
            self.names = ndata


def msiniDistribution(masses, msini):
    # This is the inclination value associated with a given probe mass (mv) and a given msin(i). iv = angle, i
	iv = np.arcsin(msini/masses)
	# prob. distribtion over m, given known m*sin(i)
    ipv = (msini/masses) * (msini/masses**2)*1.0/np.sqrt(1.0-(msini/masses)**2)
	return iv, ipv
