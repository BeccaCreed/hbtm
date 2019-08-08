###############################################################################
## Importing General Python Modules

# import numpy as np
import sklearn.preprocessing as SKLPP
import sklearn.neural_network as NN
# from sklearn.model_selection import train_test_split

###############################################################################
## Importing Files

# from conduction1D import Conduction1D as C1D
# from piddatacreation import PIDDataCreation
 

###############################################################################

class SKlearn:
    
    def __init__(self, HIDDEN_LAYER_SIZES, ACTIVATION,
                 MAX_ITER, RANDOM_STATE, TOL):
        
        # Initalzing Neural Network Parameters
        self.HIDDEN_LAYER_SIZES = HIDDEN_LAYER_SIZES
        self.ACTIVATION = ACTIVATION
        self.MAX_ITER = MAX_ITER
        self.RANDOM_STATE = RANDOM_STATE
        self.TOL = TOL
        
        self.scalerInput = SKLPP.StandardScaler(copy=True)
        self.scalerTarget = SKLPP.StandardScaler(copy=True)
        
        self.mlp = NN.MLPRegressor(hidden_layer_sizes=HIDDEN_LAYER_SIZES,
                                   activation=ACTIVATION, max_iter=MAX_ITER,
                                   random_state = RANDOM_STATE, tol=TOL )        
    
    def _trainNeuralNetwork(self, preprocessedInputs, preprocessedTargets):
        
        '''
        Trains Neural Network
        '''
        
        mlp = self.mlp
        
        mlp.fit(preprocessedInputs, preprocessedTargets)
        
    def _predictTarget(self, preprocessedInputs):
        
        '''
        Uses already trained neural network
        Takes input arguments of processed inputs
        Returns unprocessed target
        '''
        
        mlp = self.mlp
        
        return(mlp.predict(preprocessedInputs))
    
    def _defineInputProcessing(self, trainingInputs):
        
        '''
        Takes training inputs
        Calculates mean an standard deviation
        Stores later in scalerInput object for later use with the function
        scalerInput.transform in _getPreProcessedInput
        '''
        
        scalerInput = self.scalerInput
        
        scalerInput.fit(trainingInputs)        
    
    def _defineTargetProcessing(self, trainingTargets):
        
        '''
        Takes training targets
        Calculates mean an standard deviation
        Stores later in scalerTarget object for later use with the function
        scalerTarget.transform in _getPreProcessedTarget and 
        scalerTarget.inverse_Transform in _getPostProcessedTarget
        '''
        
        scalerTarget = self.scalerTarget
        
        scalerTarget.fit(trainingTargets)           
    
    def _getPreProcessedInput(self, inputs):
        
        '''
        Takes set of input quantitites
        Uses already calculated mean and standard deviation of training set
        above in _defineInputProcessing to scale inputs into nerual network
        '''
        
        scalerInput = self.scalerInput
        
        return(scalerInput.transform(inputs))
    
    def _getPreProcessedTarget(self, targets):
        
        '''
        Takes set of target training quantitites
        Uses already calculated mean and standard deviation of training set
        above in _defineTargetProcessing to scale targets for the nerual
        network training
        '''
        
        scalerTarget = self.scalerTarget
        
        return(scalerTarget.transform(targets))

    def _getPostProcessedTarget(self, scaledTargets):
        
        '''
        Takes set of target quantitites from the neural network output
        Uses already calculated mean and standard deviation of training set
        above in _defineTargetProcessing to return scaled outputs from the
        neural network to their actual values
        '''
            
        scalerTarget = self.scalerTarget
        
        return(scalerTarget.inverse_transform(scaledTargets))