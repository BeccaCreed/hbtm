###############################################################################
## Importing General Python Modules

import numpy as np
# import matplotlib.pyplot as plt
# import PID as PID
# import sklearn.preprocessing as skl
# import sklearn.neural_network as NN
# from sklearn.model_selection import train_test_split
# import time
import random

###############################################################################
## Importing Files

# from conduction1D import Conduction1D
from piddatacreation import PIDDataCreation
# from neuralnetwork import NeuralNetwork
# from obtaintemperatures import ObtainTemperatures
# from nninitalizing import NNInitalizing
# from nnpredictanderror import NNPredictAndError

###############################################################################
## Parameters and Constants

# Constants
H = 1
L = 1
K = 1
ALPHA = 1

# Eiganvalues Parameters
NUMBER_OF_EIGENVALUES = 100
ERROR_LIMIT = 1e-12

# Initial Conditions
qGen0 = 0 # Initial qgen
tempInf0 = 0 # Initial Too

# PID Parameters
PROPORTIONAL = 9.0
INTERGRAL = 7.0
DERIVATIVE = 0.5
SET_POINT = 1

# Neural Net Parameters
HIDDEN_LAYER_SIZES = (80, 8)
ACTIVATION = 'tanh'
MAX_ITER = 100000
RANDOM_STATE = 1
TOL = .00000001

# Training Chop Value
_Q_START = 25
'''
We do not display the first x-amount entries of _q when plotting since the 
values are exhorbinently high as the body pumps in heat to get the core
temperature to around 37 degrees celcius. The plot start at this point then.
We only train the Neural Network on this later range of Q and Time values.
'''
        
###############################################################################
###############################################################################
## Creating Training Data

# Numer of elements in Vector
N = 500
_tTrainingSeconds = (1/100)*np.arange(N+1)

caseConstant_Too = np.zeros(len(_tTrainingSeconds[:-1]))

caseLow_Too = (0.25*np.sin((_tTrainingSeconds[:-1])*np.pi))+0.25

caseMed_Too = (0.25*np.sin((_tTrainingSeconds[:-1])*np.pi*2))+0.25

caseHigh_Too = (0.25*np.sin((_tTrainingSeconds[:-1])*np.pi*4))+0.25

caseSquare_Too = ((0.25*np.sin(np.square(_tTrainingSeconds[:-1])*np.pi*0.5))+
                  0.25)

caseSqrt_Too = (0.25*np.sin(np.sqrt(_tTrainingSeconds[:-1])*np.pi))+0.25
    
caseDouble_Too = ((0.20*np.sin((_tTrainingSeconds[:-1])*np.pi))+0.25+
             (0.05*np.sin((_tTrainingSeconds[:-1])*np.pi*10)))

caseTriple_Too = ((0.15*np.sin((_tTrainingSeconds[:-1])*np.pi))+0.25+
             (0.05*np.sin((_tTrainingSeconds[:-1])*np.pi*10))+
             (0.05*np.sin((_tTrainingSeconds[:-1])*np.pi*20)))
    
caseRandom_Too = np.zeros(N)
caseRandom_Too[0] = 0.25
for i in range(N-1):
    if caseRandom_Too[i] >= 0.5:
        caseRandom_Too[i+1] = caseRandom_Too[i]-0.01
    elif caseRandom_Too[i] <= 0:
        caseRandom_Too[i+1] = caseRandom_Too[i]+0.01
    else:
        if random.randint(1,2) == 1:
            caseRandom_Too[i+1] = caseRandom_Too[i]+0.01
            
        else:
            caseRandom_Too[i+1] = caseRandom_Too[i]-0.01

caseRamp_Too = -(1/10)*_tTrainingSeconds[:-1]+0.5

a = np.zeros(100)
caseStep_Too = np.hstack((a+0.5, a+0.4, a+0.3, a+0.2, a+0.1))

if __name__ == '__main__':
    
    '''
    training_Too = np.vstack((caseLow_Too, caseMed_Too, caseHigh_Too,
                               caseSquare_Too, caseSqrt_Too,
                               caseDouble_Too, caseTriple_Too,
                               caseRandom_Too,
                               caseRamp_Too, caseStep_Too))
    '''
    training_Too = caseLow_Too[np.newaxis]

    numberOfTrainingCases = np.shape(training_Too)[0] 
    # Creating Testing Data from PID Controller    
    _tVector = _tTrainingSeconds
    trainingCases = []
    for i in range(numberOfTrainingCases):
        _TooVector = training_Too[i,:]
        case = PIDDataCreation(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, 
                               ERROR_LIMIT, PROPORTIONAL, INTERGRAL,
                               DERIVATIVE, SET_POINT, _TooVector, _tVector)
        trainingCases.append(case)        
    
    # Saving PID Testing Data for Plotting Later
    outputQual = 4
    numberOfTraining = len(trainingCases)
    PID_data_to_store = np.full(((outputQual*numberOfTraining), N), np.nan)
    for i in range(numberOfTraining):
        PID_data_to_store[(outputQual*i)+0,:] = trainingCases[i]._TooVector
        PID_data_to_store[(outputQual*i)+1,:] = trainingCases[i]._coreTemp
        PID_data_to_store[(outputQual*i)+2,:] = trainingCases[i]._surfTemp
        PID_data_to_store[(outputQual*i)+3,:] = trainingCases[i]._q
    
    np.savetxt('pidTrainingData.dat', PID_data_to_store)
    
    testing_Too = np.vstack((caseConstant_Too, caseLow_Too, caseHigh_Too,
                              caseDouble_Too, caseRandom_Too))
    
    # Creating Testing Data from PID Controller    
    testingCases = []
    _tVector = _tTrainingSeconds
    numberOfTestingCases = np.shape(testing_Too)[0]
    for i in range(numberOfTestingCases):
        _TooVector = testing_Too[i,:]
        case = PIDDataCreation(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, 
                               ERROR_LIMIT, PROPORTIONAL, INTERGRAL,
                               DERIVATIVE, SET_POINT, _TooVector, _tVector)
        testingCases.append(case)        
    
    # Saving PID Testing Data for Plotting Later
    outputQual = 4
    numberOfTests = len(testingCases)
    PID_data_to_store = np.full(((outputQual*numberOfTests), N), np.nan)
    for i in range(numberOfTests):
        PID_data_to_store[(outputQual*i)+0,:] = testingCases[i]._TooVector
        PID_data_to_store[(outputQual*i)+1,:] = testingCases[i]._coreTemp
        PID_data_to_store[(outputQual*i)+2,:] = testingCases[i]._surfTemp
        PID_data_to_store[(outputQual*i)+3,:] =testingCases[i]._q
        
    np.savetxt('pidTestingData.dat', PID_data_to_store)
