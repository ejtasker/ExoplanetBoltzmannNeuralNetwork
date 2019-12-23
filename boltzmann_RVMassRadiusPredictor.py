###
### Predicting planetary mass and radius for radial velocity (RV) observed planet
### Code outputs mdist distributions for all planets with msini but no radius
### Note: this is the raw mass distribution, without the correction for the known msin(i) value.
### (see util.py for a tool to combine with msin(i))
###

dir = "../"
import sys, os
sys.path.insert(0, dir)

import numpy as np
from math import *
from boltzmann import PlanetGenerator, Hamiltonian
import datetime
import re
import csv
import pandas as pd
import scipy.stats as st

# RV data: no mass or radius measurement
def RVDataImputeMass(features):
    mdist, _ = generator.imputeDistribution(features, mask, N=ndist, mcsteps=mcs) # Calculate distribution of possible properties
    return mdist


#-- Parameters you should set --#
planet = "proxb_"  # planet name
# features: fill in features that you know from observations
# feature order: planet radius, planet mass, nplants, m star, period, Teff
features = np.array([0, 0, 1, 0.12, 11.2, 263.55])
#------------------------------#

# Network parameters, see appendix of paper
mcs = 2000
ndist = 2000

# Load in the pre-trained boltzmann network
generator = PlanetGenerator(pretrained="boltzmann.th")

datestr = str(datetime.date.today()) # add date to filename
dir_name = "data/"

# unmask properties that you want imputed (set "0" in mask)
# property order:  planet radius, planet mass, number of planets in system, stellar mass, orbital period & equilibrium temperature
mask = np.array([0,0,1,1,1,1]) # no mass or radius present. For transit mass prediction, mask is [1,0,1,1,1,1]
maskstr = '_mask'+re.sub('[\s+]','', str(mask)) # used for filenames

# file that will be appended to during calculation (useful for many values)
filename = os.path.join(dir_name, planet+"rv_"+datestr+".txt")

# let's roll
# mdist is a 6D array, where the first column is the radius distribution, the second is the mass distribution.
mdist = RVDataImputeMass(features=features)

# Save data: property distributions
pyfile = os.path.join(dir_name, planet+"rv_mdists"+maskstr+"_mcsteps"+str(mcs)+"_ndist"+str(ndist)+"_"+datestr+".npy")
np.save(pyfile, mdist)
