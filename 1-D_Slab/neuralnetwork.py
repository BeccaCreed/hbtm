###############################################################################
## Importing General Python Modules

import numpy as np
# import sklearn.preprocessing as skl
# import sklearn.neural_network as NN
# from sklearn.model_selection import train_test_split

###############################################################################
## Importing Files

# from conduction1D import Conduction1D
# from piddatacreation import PIDDataCreation
from SKLearn import SKlearn
 

###############################################################################

class NeuralNetwork:
    
    def __init__(self, HIDDEN_LAYER_SIZES, ACTIVATION, MAX_ITER, RANDOM_STATE,
                 TOL, _Q_START):
        
        self.SKL = SKlearn(HIDDEN_LAYER_SIZES, ACTIVATION, MAX_ITER,
                           RANDOM_STATE, TOL)
        
        self._Q_START = _Q_START
        
    def _trainNN(self, trainingInputs, trainingTargets):
        
        SKL = self.SKL
        
        # PreProcess Inputs
        SKL._defineInputProcessing(trainingInputs)
        preprocessedInputs = SKL._getPreProcessedInput(trainingInputs)
        
        # PreProcess Targets
        SKL._defineTargetProcessing(trainingTargets)
        preprocessedTargets = SKL._getPreProcessedTarget(trainingTargets)
        '''
        NOTE:
        sklearn.neural_network.MLPRegressor (abbbreviated for us as NN.mlp) 
        "fit" function requires a 1D-array.
        Comparativly, sklearn.preprocessing.StandardScaler requires a 2D-array.
        That is why we have to expand the dimensions above, but then collapse
        the dimensions below
        '''
        preprocessedTargets = preprocessedTargets.flatten('F')
        
        # Training Neural Network
        SKL._trainNeuralNetwork(preprocessedInputs, preprocessedTargets)
        
    def _predictFromNN(self, testInputs):
        
        SKL = self.SKL

        # PreProcess Inputs
        preprocessedInputs = SKL._getPreProcessedInput(testInputs)
        
        # Predict Targets from Inputs
        unprocessedTargets = SKL._predictTarget(preprocessedInputs)
        
        # PostProcess Targets
        _Q_Predicted = SKL._getPostProcessedTarget(unprocessedTargets)
        
        return(_Q_Predicted)