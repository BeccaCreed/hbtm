###############################################################################
## Importing General Python Modules

# import numpy as np
# import matplotlib.pyplot as plt
# import PID as PID
# import sklearn.preprocessing as skl
# import sklearn.neural_network as NN
# from sklearn.model_selection import train_test_split
# import time
# import random

###############################################################################
## Importing Files

# from conduction1D import Conduction1D
from piddatacreation import PIDDataCreation
from neuralnetwork import NeuralNetwork
# from obtaintemperatures import ObtainTemperatures

###############################################################################
class NNInitalizing:
    
    def __init__(self, HIDDEN_LAYER_SIZES, ACTIVATION, MAX_ITER, RANDOM_STATE,
                 TOL, _Q_START, _TooVector, _tVector):
        
        # Storing Too and times as object attribute
        self._TooVector = _TooVector
        self._tVector = _tVector
        
        self.NN = NeuralNetwork(HIDDEN_LAYER_SIZES, ACTIVATION, MAX_ITER,
                                RANDOM_STATE, TOL, _Q_START)

    def _trainNeuralNetwork(self, inputs, targets):
        
        NN = self.NN
        
        _surfTemp = self._surfTemp
        
        _q = self._q
        
        # Training Neural network
        NN._trainNN(_surfTemp, _q)