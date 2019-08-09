###############################################################################
## Importing General Python Modules

import numpy as np
# import matplotlib.pyplot as plt
# import PID as PID
# import sklearn.preprocessing as skl
# import sklearn.neural_network as NN
# import time
# import random
import sys

###############################################################################
## Importing Files

# from conduction1D import Conduction1D
# from piddatacreation import PIDDataCreation
from neuralnetwork import NeuralNetwork
# from obtaintemperatures import ObtainTemperatures
# from nninitalizing import NNInitalizing
from nnpredictanderror import NNPredictAndError

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
nodes = int(sys.argv[2])
layers = int(sys.argv[3])
HIDDEN_LAYER_SIZES = (layers*(nodes,))
ACTIVATION = 'tanh'
MAX_ITER = 100000
RANDOM_STATE = 0
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
## Creating Training Data

# Numer of elements in Vector
N = 500
_tTrainingSeconds = (1/100)*np.arange(N+1)

###############################################################################
## creatingIdentifier

def creatingIdentifier(jobNumber):
    
    if jobNumber <= 3:
        ltr = 'A'
    elif jobNumber <= 7:
        ltr = 'B'
    else:
        ltr = 'C'
        
    remain4 = (jobNumber%4)
    if remain4 == 0:
        num = '0'
    elif remain4 == 1:
        num = '1'
    elif remain4 == 2:
        num = '2'
    else:
        num = '3'
        
    return(ltr+num)
    
###############################################################################
## Selecting Inputs Function

def buildingInputs(Tsurf, Tcore, _Q_START, identifier):
    
    if identifier[0] == 'A':
        trainingQuanity = Tsurf[np.newaxis]
    elif identifier[0] == 'B':
        trainingQuanity = Tcore[np.newaxis]
    else:
        trainingQuanity = np.vstack((Tsurf,Tcore))
        
    if identifier[1] == '0':
        output = trainingQuanity[:,_Q_START:].T
    elif identifier[1] == '1':
        t0 = trainingQuanity[:,_Q_START:]
        t1 = trainingQuanity[:,(_Q_START-1):-1]
        output = np.vstack((t0,t1)).T
    elif identifier[1] == '2':
        t0 = trainingQuanity[:,_Q_START:]
        t1 = trainingQuanity[:,(_Q_START-1):-1]
        t2 = trainingQuanity[:,(_Q_START-2):-2]
        output = np.vstack((t0,t1,t2)).T
    else:
        t0 = trainingQuanity[:,_Q_START:]
        t1 = trainingQuanity[:,(_Q_START-1):-1]
        t2 = trainingQuanity[:,(_Q_START-2):-2]
        t3 = trainingQuanity[:,(_Q_START-3):-3]
        output = np.vstack((t0,t1,t2,t3)).T
        
    return(output)

###############################################################################
## Create Coloring Lines
colors = ['blue', 'green', 'red', 'orange', 'magenta']

labeling = ['Low Freq Sine', 'Med Freq Sine', 'High Freq Sine',
            '0.25*Sin((t^2)*0.5*pi)+0.25', '0.25*Sin((t^0.5)*pi)+0.25',
            'Double Sine', 'Triple Sine',
            'Random Walk',
            'Ramp', 'Step']

###############################################################################
## Creating Neural Network and Testing

def createAndTrainNN(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, ERROR_LIMIT,
                     HIDDEN_LAYER_SIZES, ACTIVATION,  MAX_ITER, RANDOM_STATE,
                     TOL, _Q_START, _tVector, identifier,
                     trainToo, trainCoreTemp, trainSurfTemp, trainQ,
                     testToo, testCoreTemp, testSurfTemp, testQ):
    
    '''
    This section creates the Neural Network. The training data from the PID
    controller is calculated in another sectoin.
    '''
    
    # Defining Training Inputs
    trainingInputs = buildingInputs(trainSurfTemp, trainCoreTemp, _Q_START,
                                    identifier)
    # Defining Training Targets
    trainingTargets = trainQ[_Q_START:][np.newaxis].T
    # Initialzing Neural Network Object
    NN = NeuralNetwork(HIDDEN_LAYER_SIZES, ACTIVATION,  MAX_ITER, RANDOM_STATE,
                       TOL, _Q_START)
    
    # Training Neural network
    NN._trainNN(trainingInputs, trainingTargets)
        
    testingCases = np.shape(testToo)[0]
    _qError = np.full((testingCases,), np.nan)
    _TcoreError = np.full((testingCases,), np.nan)
    _TsurfError = np.full((testingCases,), np.nan)
    
    for i in range(testingCases):
        # Defining variables of alrady calculated PID Data
        _Too = testToo[i,:]
        _Tcore = testCoreTemp[i,:]
        _Tsurf = testSurfTemp[i,:]
        _Q = testQ[i,:]
        # Intializing NNPredictAndError Class Object
        nnPredict = NNPredictAndError(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES,
                                      ERROR_LIMIT, _Q_START, _Q, _Too,
                                      _tVector, NN)
        # Defining testing inputs
        testInputs = buildingInputs(_Tsurf, _Tcore, _Q_START, identifier)
        # Getting predicted Q's from Neural Network
        nnPredict._predictQ(testInputs)
        # Calling getTempeatures Function
        nnPredict._calcuateTemperatrues()
        # Calculating All Errors
        _qError[i] = nnPredict.qRmse(_Q)
        _TcoreError[i] = nnPredict.TCoreRmse(_Tcore)
        _TsurfError[i] = nnPredict.TSurfRmse(_Tsurf)

    return(_qError, _TcoreError, _TsurfError)
    
###############################################################################

if __name__ == '__main__':
    
    # Calling PID Data Creation to create training data
    # It takes an array of T-infinity's and uses a PID Controller to adjust the
    # Q-value. The class calls on the 1D Conduction Solution to get the
    # core and surface temperature values.
    
    # Defining Neural Network Training List
    
    jobArrayValue = int(sys.argv[1])
    identifier = creatingIdentifier(jobArrayValue)
    
    pidTrainingData = np.loadtxt('pidTrainingData.dat')
    trainingToo = pidTrainingData[0::4,:]
    trainingCoreTemp = pidTrainingData[1::4,:]
    trainingSurfTemp = pidTrainingData[2::4,:]
    trainingQ = pidTrainingData[3::4,:]
    
    trainToo = trainingToo[0]
    trainCoreTemp = trainingCoreTemp[0]
    trainSurfTemp = trainingSurfTemp[0]
    trainQ = trainingQ[0]

    # Redefining variable for easier access later
    _tVector = _tTrainingSeconds
        
    pidTestingData = np.loadtxt('pidTestingData.dat')
    testToo = pidTestingData[0::4,:]
    testCoreTemp = pidTestingData[1::4,:]
    testSurfTemp = pidTestingData[2::4,:]
    testQ = pidTestingData[3::4,:]
    testCases = np.shape(testToo)[0]
    
    rangeOfRandomSeeds = range(100)
    numberOfRandomSeeds = len(rangeOfRandomSeeds)
    
    qError = np.full((numberOfRandomSeeds,testCases), np.nan)
    TcoreError = np.full((numberOfRandomSeeds,testCases), np.nan)
    TsurfError = np.full((numberOfRandomSeeds,testCases), np.nan)
    
    for i in rangeOfRandomSeeds:
        RANDOM_STATE = i
        (_qError, _TcoreError, _TsurfError) = createAndTrainNN(H, L, K, ALPHA,
        NUMBER_OF_EIGENVALUES, ERROR_LIMIT, HIDDEN_LAYER_SIZES, ACTIVATION, 
        MAX_ITER, RANDOM_STATE, TOL, _Q_START, _tVector, identifier,
        trainToo, trainCoreTemp, trainSurfTemp, trainQ,
        testToo, testCoreTemp, testSurfTemp, testQ)
        qError[i,:] = _qError
        TcoreError[i,:] = _TcoreError
        TsurfError[i,:] = _TsurfError
    
    rowOfNan = np.full((testCases,), np.nan)
    nodesString = str(nodes).zfill(3)
    layersString = str(layers).zfill(2)
    outputError = np.vstack((rowOfNan, qError, rowOfNan, TcoreError, rowOfNan, 
                             TsurfError))
    np.savetxt('./output/zulu_%s_%s_%s_error.dat' % 
               (identifier, nodesString, layersString), outputError)
