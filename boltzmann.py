import numpy as np
from math import *

# network uses pytorch
import torch
from torch.autograd import Variable, Function
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

from util import tovar, toivar

class Hamiltonian(nn.Module):
    def __init__(self):
        super(Hamiltonian,self).__init__()

        # Node layers (architecture)
        self.l1 = nn.Linear(6,128).cuda() # 6 inputs to 128 nodes (pass in planets all at once, so each input has about 350 lines for training)
        self.l2 = nn.Linear(128,128).cuda() # 128 to 128 nodes
        self.l3 = nn.Linear(128,128).cuda() # 128 to 128 nodes
        self.l4 = nn.Linear(128,1).cuda() # 128 to single node

        # Adam optimizer ~ stochastic gradient descent to update network weights iteratively
        self.adam = torch.optim.Adam(self.parameters(), lr = 1e-3)

    # neural network structure
    def forward(self,x):

        # Push each layer of nodes through an 'exponetial linear unit' (elu)
        z = F.elu(self.l1(x))
        z = F.elu(self.l2(z))
        z = F.elu(self.l3(z))

        # Using elu on the final layer prevents runaway negative energy
        y = F.elu(self.l4(z))+1  # +1 there so that zero is minmum possible energy, just for convienience.
        return y

class PlanetGenerator():
    def __init__(self, pretrained=None):
        # fields to log: planet radius, planet mass, stellar mass, orbital period and eq. temperature
        self.logidx = [0,1,3,4,5]

        # Mean and standard deviation for the dataset normalization used during training
        self.mu = np.array([-0.37862956, -1.20962733,  1.78363636,  0.00947734,  1.81017754,  6.99668626])
        self.std = np.array([0.89933643,  2.04433828,  1.48889998,  0.44675035,  1.38930857,  0.56938733])

        # Load in the pretrained network
        self.net = Hamiltonian()

        if pretrained:
            self.net.load_state_dict(torch.load(pretrained))

	# NN works with the 'standard score': (value - mean)/standard deviation
	# Gives a distribution of mean = 0 and standard deviation = 1
	# Log appropriate fields, compute standard score. Input is sdata equivalent values.
	#
	# These means and standard deviations must be the ones used to train the net, not
	# whatever the current dataset under analysis is! (i.e. don't change unless you re-train)

    def forward_transform(self,d):
        d2 = d.copy()
        d2[:,self.logidx] = np.log(d2[:,self.logidx]+1e-4)
        return (d2-self.mu)/self.std

    # Reverse this back into normal value space
    def backward_transform(self,d):
        d2 = self.std*(d.copy())+self.mu
        d2[:,self.logidx] = np.exp(d2[:,self.logidx])-1e-4
        return d2

    # 'mcstep' performs a number of masked Monte-Carlo steps based on the energy from the neural network on a set of planet properties
    # This is the main ingredient of producing new planets, but since it works in the transformed space its more convenient to use
    # wrapper functions later on in the code.
    #
    # No fields are masked while generating the initial distribution for the data (just later when using it to recreate missing values)
    # We're not using masks, so always 0
    # Input is all planets in 6 fields, e.g. for training data a 350 x 6 array
    def mcstep(self, x, mask, N=1000, stepsize = 0.05):
        # Monte-Carlo updates
        for i in range(N):

            # Array with Gaussian random numbers with standard deviation 'stepsize' for each of the 350 x 6 fields
            dx = stepsize*np.random.randn(x.shape[0],6)

            # zero the one with mask values: no masks for training so dx = dx*1
            # When filling in missing values, the non-missing values have dx = 0 (immutable data)
            dx = dx*(1-mask)

            # Pass planets (350 x 6 values for training set) and a set with slightly offset properties
            # (Note this dx off-set works because we normalised all the fields)
            # Return exp(z), where z is the NN representation from the final node everything is squished into
            E1 = self.net.forward(tovar(x))
            E2 = self.net.forward(tovar(x+dx))

            # dE = change in 'energy' of the output
            # clamp forces all inputs to be between -30, 0 (values outside are set to min or max value)
            # limits because: 1e-30 ~ 0 and e0 ~ 1
            dE = torch.clamp((E1-E2),-30,0)

            # Calculation probability as the exponential of the energy difference between planet and offset planet
            P = torch.exp(dE).cpu().data.numpy()

            # Random number between 0 and 1
            r = np.random.rand(x.shape[0],1)

            # If random r is less than probability, P, then accept new planet x+dx.
            # This is in a loop, so repeat N times
            x = x + dx*(r<P)

        return x, E1 # new planets sampled via monte carlo walk

    def train(self, data, mcsteps=1000):
        # data passed in is train_data
        BS = data.shape[0]

        # This is now the 400 x 6 data set of planets
        y = self.forward_transform(data.copy())

        # Pass training planets through network and compute mean value of the one node representation
        Ey = self.net.forward(tovar(y)).mean()

        # This is also the 400 x 6 data set of planets
        x = y.copy()

        # Set all mask values to zero for generating initial data representation
        zmask = np.zeros((x.shape[0],x.shape[1]))

        # Make a planet by monte carlo sampling of the NN energy representation for the planet distribution
        x, _ = self.mcstep(x, zmask, mcsteps)

        # Pass planet through network and compute mean value of one node representation again
        Ex = self.net.forward(tovar(x)).mean()
        i = 0

        # Update the network to increase the energy of fake planets and decrease the energy of real ones
        # For loop is in case of doing multiple update steps per generation (higher values = faster convergence, but less stable)
        for i in range(1):
            self.net.zero_grad()

            loss1 = Ey
            loss2 = -Ex

            loss = loss1 + loss2
            loss.backward()

            self.net.adam.step()

        #return loss1.cpu().detach().item(), loss2.cpu().detach().item(), x, y  # v. latest pytorch version
        return loss1.cpu().data.numpy()[0], loss2.cpu().data.numpy()[0], x, y


    # Convenience function that returns the energies of (NumPy formatted) data
    def energy(self,data):
        EG = self.net.forward(tovar(data)).cpu().data.numpy()[:,0]
        return EG

    # This function takes a single incomplete planet record and generating a distribution of possibilities consistent with that record
    def imputeDistribution(self, datarow, mask, N=2000, mcsteps=1000):
        imputedata = self.forward_transform(datarow.copy().reshape((1,datarow.shape[0])).repeat(N,axis=0))

       	# Replace masked values with random initializations
        for i in range(mask.shape[0]):
            if mask[i] == 0:
                imputedata[:,i] = np.random.randn(imputedata.shape[0])

        imputedata, energies = self.mcstep(imputedata, mask, mcsteps)
        return self.backward_transform(imputedata), energies.cpu().data.numpy()
