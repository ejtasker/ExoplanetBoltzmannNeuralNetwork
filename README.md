# ExoplanetBoltzmannNeuralNetwork

Code for the neural network presented in the journal paper:

Tasker, Laneauville & Guttenberg, the Astronomical Journal, 2020.
https://arxiv.org/pdf/1911.11035.pdf

The code can impute missing values for planet radius, planet mass, planet period, planet equilibrium temperature, stellar mass and number of planets in the system based on the included training data.

Files:

boltzmann.th : the pre-trained neural network that can be used to impute values

NEA_radmasstpersmasspnum.csv: the training and test data (also available from the journal in a machine-readable format)

boltzmann.py : the neural network

util.py : added tools for analysis

boltzmann_RVMassRadiusPredictor.py : demo code to calculate the radius and mass of a planet observed via the radial velocity method

boltzmann_TransitMassPredictor.py : as above, but for a transit detection.
