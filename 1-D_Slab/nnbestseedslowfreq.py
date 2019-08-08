###############################################################################
## Importing General Python Modules

import numpy as np
# import matplotlib.pyplot as plt
# import PID as PID
# import sklearn.preprocessing as skl
# import sklearn.neural_network as NN
# import time
# import random

###############################################################################
## Importing Files

#from conduction1D import Conduction1D
from piddatacreation import PIDDataCreation
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
nodes = int(sys.argv[3])
layers = int(sys.argv[4])
HIDDEN_LAYER_SIZES = (nodes, layers)
ACTIVATION = 'tanh'
MAX_ITER = 100000
RANDOM_STATE = int(sys.argv[2])
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
    elif identifier[1] == 'B':
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

if __name__ == '__main__':
    
    # Calling PID Data Creation to create training data
    # It takes an array of T-infinity's and uses a PID Controller to adjust the
    # Q-value. The class calls on the 1D Conduction Solution to get the
    # core and surface temperature values.
    
    # Defining Neural Network Training List
    
    jobArrayValue = int(sys.argv[1])
    identifier = creatingIdentifier(jobArrayValue)
    
    bestSeeds = {'A0':51, 'A1':73, 'A2':29, 'A3':85,
                 'B0':85, 'B1':25, 'B2':6, 'B3':12,
                 'C0':85, 'C1':25, 'C2':6, 'C3':12}
    RANDOM_STATE = bestSeeds[identifier]
    
    pidTrainingData = np.loadtxt('pidTrainingData.dat')
    trainingToo = pidTrainingData[0::4,:]
    trainCoreTemp = pidTrainingData[1::4,:]
    trainSurfTemp = pidTrainingData[2::4,:]
    trainingQ = pidTrainingData[3::4,:]

    nnCases = np.shape(trainingToo)[0]

    # Redefining variable for easier access later
    _tVector = _tTrainingSeconds
    
    '''
    This loop creates the Neural Networks. From the PID controlling class, it
    creates the data 
    '''
    trainingCases = []
    for i in range(nnCases):
        # Defining variables of alrady calculated PID Data
        _TooVector = trainingToo[i,:]
        _Too = trainingToo[i,:]
        _Tcore = trainCoreTemp[i,:]
        _Tsurf = trainSurfTemp[i,:]
        _q = trainingQ[i,:]
        # Defining Training Inputs
        trainingInputs = buildingInputs(_Tsurf, _Tcore, _Q_START, identifier)
        # Defining Training Targets
        trainingTargets = _q[_Q_START:][np.newaxis].T
        # Initialzing Neural Network Object
        case = NeuralNetwork(HIDDEN_LAYER_SIZES, ACTIVATION,  MAX_ITER,
                             RANDOM_STATE, TOL, _Q_START)
        # Training Neural network
        case._trainNN(trainingInputs, trainingTargets)
        # Appending case to list where the objects are stored
        trainingCases.append(case)
    
    pidTestingData = np.loadtxt('pidTestingData.dat')
    testingToo = pidTestingData[0::4,:]
    testingCoreTemp = pidTestingData[1::4,:]
    testingSurfTemp = pidTestingData[2::4,:]
    testingQ = pidTestingData[3::4,:]
    
    nnTests = np.shape(testingToo)[0]
    
    qError = np.full((len(trainingCases), len(testingToo)),np.nan)
    TCoreError = np.full((len(trainingCases), len(testingToo)),np.nan)
    TSurfError = np.full((len(trainingCases), len(testingToo)),np.nan)
    
    predictCases = []
    for i in range(nnCases):
        juliette = []
        neuralNetwork = trainingCases[i]
        for j in range(nnTests):
            # Defining variables of alrady calculated PID Data
            _Too = testingToo[j,:]
            _Tcore = testingCoreTemp[j,:]
            _Tsurf = testingSurfTemp[j,:]
            _Q = testingQ[j,:]
            # Intializing NNPredictAndError Class Object
            nnPredict = NNPredictAndError(H, L, K, ALPHA,
                                          NUMBER_OF_EIGENVALUES, ERROR_LIMIT,
                                          _Q_START, _Q, _Too, _tVector, 
                                          neuralNetwork)
            # Defining testing inputs
            testInputs = buildingInputs(_Tsurf, _Tcore, _Q_START, identifier)
            # Getting predicted Q's from Neural Network
            nnPredict._predictQ(testInputs)
            # Calling getTempeatures Function
            nnPredict._calcuateTemperatrues()
            # Calculating All Errors
            qError[i,j] = nnPredict.qRmse(_Q)
            TCoreError[i,j] = nnPredict.TCoreRmse(_Tcore)
            TSurfError[i,j] = nnPredict.TSurfRmse(_Tsurf)
            # Appending Results to juliette list
            juliette.append(nnPredict)
        # Appending results for all testing cases for a neural net to list
        predictCases.append(juliette)
    
    rowOfNan = np.full((nnTests,), np.nan)
    rdmst = str(RANDOM_STATE).zfill(2)
    outputError = np.vstack((rowOfNan, qError, rowOfNan, TCoreError, rowOfNan, 
                             TSurfError))
    np.savetxt('./output/zulu_%s_%s_%.0f_%.0f_error.dat' % 
               (identifier, rdmst, nodes, layers), outputError)
    
    outputQual = 3
    testingOffset = outputQual*nnTests
    outputData = np.full((((outputQual*nnCases*nnTests)+1), N), np.nan)
    for i in range(nnCases):
        for j in range(nnTests):
            startRow = (testingOffset*i)+(outputQual*j)
            outputData[startRow+0,:] = predictCases[i][j]._coreTemp
            outputData[startRow+1,:] = predictCases[i][j]._surfTemp
            outputData[startRow+2,:] = predictCases[i][j]._qVector

    np.savetxt('./output/yankee_%s_%s_%.0f_%.0f_Data.dat' % 
               (identifier, rdmst, nodes, layers), outputData)

    np.savetxt('zulu_%s_nnVerficationData.dat' % identifier, outputData)
