###
### Predicting planetary mass for a transit observed planet
### Code outputs mdist distributions for all planets with measured radius but no mass
###

dir = "../"
import sys, os
sys.path.insert(0, dir)

import numpy as np
from math import *
from util import tovar, toivar, PlanetLoader, massDistributionFromPrior
from boltzmann import PlanetGenerator, Hamiltonian
import datetime
import re
import csv
import pandas as pd
import scipy.stats as st

# Transit data: no mass measurement
def TransDataImputeMass(features):
    mdist, _ = generator.imputeDistribution(features, mask, N=ndist, mcsteps=mcs) # Calculate distribution of possible properties
    return mdist

#-- Parameters you should set --#
planet = "kepler452b_" # planet name
# features: fill in features that you know from observations
# feature order: planet radius, planet mass, nplants, m star, period, Teff
features = np.array([0.145, 0, 1, 1.037, 384.8, 285.9])
#------------------------------#

# Load in the pre-trained boltzmann network: best of the 10 tried
generator = PlanetGenerator(pretrained="boltzmann.th")

datestr = str(datetime.date.today()) # add date to filename to avoid future sadness
dir_name = "data/"

# unmask properties that you want imputed (set "0" in mask)
# property order:  planet radius, planet mass, number of planets in system, stellar mass, orbital period & equilibrium temperature
mask = np.array([1,0,1,1,1,1]) # no mass present
maskstr = '_mask'+re.sub('[\s+]','', str(mask)) # used for filenames

# Network parameters, see appendix of paper
mcs = 2000
ndist = 2000

# file that will be appended to during calculation (useful for many values)
filename = os.path.join(dir_name, planet+"trans_"+datestr+".txt")

# let's roll
# mdist is a 6D array, where the second column is the mass distribution
mdist = TransDataImputeMass(features=features)

# Save data: property distributions
pyfile = os.path.join(dir_name, planet+"trans_mdists"+maskstr+"_mcsteps"+str(mcs)+"_ndist"+str(ndist)+"_"+datestr+".npy")
np.save(pyfile, mdist)
