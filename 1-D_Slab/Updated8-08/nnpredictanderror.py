 ###############################################################################
## Importing General Python Modules

import numpy as np
#import matplotlib.pyplot as plt
#import PID as PID
#import sklearn.preprocessing as skl
#import sklearn.neural_network as NN
#from sklearn.model_selection import train_test_split
#import time
#import random
#import itertools

###############################################################################
## Importing Files

from conduction1D import Conduction1D
#from piddatacreation import PIDDataCreation
#from neuralnetwork import NeuralNetwork
#from obtaintemperatures import ObtainTemperatures

###############################################################################
        
class NNPredictAndError:
    
    def __init__(self, H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, ERROR_LIMIT, 
                 _Q_START, starting_QVrray, _TooVector, _tVector, NeuralNet):
        
        # Calling and Initializing Conduction1D Object
        self.C1D = Conduction1D(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, 
                                ERROR_LIMIT, _TooVector, _tVector)
        
        self._Q_START = _Q_START
        
        self.NeuralNet = NeuralNet
        
        self.starting_QVrray = starting_QVrray
        
        # Putting Too and T Vectors into Object
        self._TooVector = _TooVector
        self._tVector = _tVector
        
        # Preallocating Space for Core and Surface Temperature Vectors
        self._coreTemp = np.full((len(_TooVector)),np.nan)
        self._surfTemp = np.full((len(_TooVector)),np.nan)
        
        
        
    def _predictQ(self, testingInputs):
        
        NeuralNet = self.NeuralNet
        
        predictedQ = NeuralNet._predictFromNN(testingInputs)
        
        self._constructQVector(predictedQ)
        
    def _constructQVector(self, predictedQ):
        
        _Q_START = self._Q_START
        starting_QVrray = self.starting_QVrray
        
        '''
        These are the initial values from the PID which the neural network is
        not trained on as the Qvalues are very high inorder to raise the core
        tempeature to appproximatly 37 degrees.
        '''
        part1 = starting_QVrray[:_Q_START]
        
        '''
        These are the q-values from the trained neural network that will be
        used once the intial values from the PID bring the core temperature
        up to approximatly 37 degrees.
        '''
        
        part2 = predictedQ
        
        self._qVector = np.hstack((part1, part2))
        
    def _calcuateTemperatrues(self):
        
        C1D = self.C1D
        
        L = C1D.L
        
        _qVector = self._qVector
        
        _TooVector = self._TooVector
        length = len(_TooVector)
        
        _coreTemp = self._coreTemp
        _surfTemp = self._surfTemp
        
        for i in range(length):
                        
            #Get Temperature at Core
            _coreTemp[i] = C1D._getTemp(0)
            
            #Get Temperature at Surface
            _surfTemp[i] = C1D._getTemp(L)
            
            C1D._q[i] = _qVector[i]
            
    def qRmse(self, _qActual):
                
        _Q_START = self._Q_START
        
        _qVector = self._qVector
        
        predicted = _qVector[_Q_START:]
        
        actual = _qActual[_Q_START:]
        
        return np.sqrt(((predicted - actual)**2).mean())
    
    def TCoreRmse(self, _TCoreActual):
                
        _Q_START = self._Q_START
        
        _coreTemp = self._coreTemp
        
        predicted = _coreTemp[_Q_START:]
        
        actual = _TCoreActual[_Q_START:]
        
        return np.sqrt(((predicted - actual)**2).mean())
    
    def TSurfRmse(self, _TSurfActual):
                
        _Q_START = self._Q_START
        
        _surfTemp = self._surfTemp
        
        predicted = _surfTemp[_Q_START:]
        
        actual = _TSurfActual[_Q_START:]
        
        return np.sqrt(((predicted - actual)**2).mean())