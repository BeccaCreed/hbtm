###############################################################################
## Importing General Python Modules

import numpy as np
# import matplotlib.pyplot as plt
# import PID as PID
# import sklearn.preprocessing as skl
# import sklearn.neural_network as NN
# from sklearn.model_selection import train_test_split
# import time

###############################################################################
## Importing Files

# from conduction1D import Conduction1D as C1D
from piddatacreation import PIDDataCreation
#from neuralnetwork import NeuralNetwork
#from obtaintemperatures import ObtainTemperatures
 
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

# PID Parameters
# All defined later except setpoint
SET_POINT = 1

###############################################################################
## Creating Training Data

# Numer of elements in Vector
N = 500

_TooTraining1 = np.zeros(N) #Constant ambient temperature
_tTrainingSeconds = (1/100)*np.arange(N+1)

_TooTraining2 = np.zeros(N)
for i in range(N):
    _TooTraining2[i] = (0.25*np.sin((_tTrainingSeconds[i])*np.pi))+0.25

###############################################################################
## Create Coloring Lines
colors = ['blue', 'lightblue', 'darkblue', 'green', 'lightgreen', 'darkgreen', 
          'red', 'orange', 'magenta']

###############################################################################

if __name__ == '__main__':
    
    # Calling PID Data Creation to create training data
    # It takes an array of T-infinity's and uses a PID Controller to adjust the
    # Q-value. The class calls on the 1D Conduction Solution to get the
    # core and surface temperature values.

    # Creating PID Parameters for Testing
    pVector = np.array([5, 7, 9])
    iVector = np.array([3, 5, 7])
    dVector = np.array([0.1, 0.3, 0.5])
    
    '''
    Iterating over all combinations of PID Parameters defined above.
    It sets the parameters, and then calls the PIDDataCreation class with
    those given parameters. First, it uses the vector, where Too=0
    '''
    result = []
    for i in range(len(pVector)):
        a = []
        for j in range(len(iVector)):
            b = []
            for k in range(len(dVector)):
                PROPORTIONAL = pVector[i]
                INTERGRAL = iVector[j]
                DERIVATIVE = dVector[k]
                pidDCtraining = PIDDataCreation(H, L, K, ALPHA,
                                                NUMBER_OF_EIGENVALUES,
                                                ERROR_LIMIT, PROPORTIONAL,
                                                INTERGRAL, DERIVATIVE,
                                                SET_POINT, _TooTraining1,
                                                _tTrainingSeconds)
                b.append(pidDCtraining)
            a.append(b)
        result.append(a)

    
    # Exporting Data for Too=0
    totalCases = len(pVector)*len(iVector)*len(dVector)
    pLen = len(pVector)
    iLen = len(iVector)
    outputDataConstant = np.full((N, ((3*totalCases)+1)), np.nan)
    outputDataConstant[:,0] = _tTrainingSeconds[:-1]
    for i in range(len(pVector)):
        for j in range(len(iVector)):
            for k in range(len(dVector)):
                columnEntry = 3*((i*(pLen**2))+(j*iLen)+k)+1
                outputDataConstant[:,columnEntry+0] = result[i][j][k]._coreTemp
                outputDataConstant[:,columnEntry+1] = result[i][j][k]._surfTemp
                outputDataConstant[:,columnEntry+2] = result[i][j][k]._q     
    np.savetxt('PIDTuningOutputConstant.dat', outputDataConstant)
    
    # Training SineWave Data
    result=[]
    for i in range(len(pVector)):
        a = []
        for j in range(len(iVector)):
            b = []
            for k in range(len(dVector)):
                PROPORTIONAL = pVector[i]
                INTERGRAL = iVector[j]
                DERIVATIVE = dVector[k]
                pidDCtraining = PIDDataCreation(H, L, K, ALPHA,
                                                NUMBER_OF_EIGENVALUES,
                                                ERROR_LIMIT, PROPORTIONAL,
                                                INTERGRAL, DERIVATIVE,
                                                SET_POINT, _TooTraining2,
                                                _tTrainingSeconds)
                b.append(pidDCtraining)
            a.append(b)
        result.append(a)
    
    # Exporting SineWave Data
    totalCases = len(pVector)*len(iVector)*len(dVector)
    pLen = len(pVector)
    iLen = len(iVector)
    outputDataSine = np.full((N, ((3*totalCases)+1)), np.nan)
    outputDataSine[:,0] = _tTrainingSeconds[:-1]
    for i in range(len(pVector)):
        for j in range(len(iVector)):
            for k in range(len(dVector)):
                columnEntry = 3*((i*(pLen**2))+(j*iLen)+k)+1
                outputDataSine[:,columnEntry+0] = result[i][j][k]._coreTemp
                outputDataSine[:,columnEntry+1] = result[i][j][k]._surfTemp
                outputDataSine[:,columnEntry+2] = result[i][j][k]._q     
    np.savetxt('PIDTuningOutputSine.dat', outputDataSine)
