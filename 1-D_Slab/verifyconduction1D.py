###############################################################################
## Importing General Python Modules

import numpy as np
import matplotlib.pyplot as plt
# import PID as PID
# import sklearn.preprocessing as skl
# import sklearn.neural_network as NN
# from sklearn.model_selection import train_test_split
# import time

###############################################################################
## Importing Files

from conduction1D import Conduction1D
# from piddatacreation import PIDDataCreation
# from neuralnetwork import NeuralNetwork
# from obtaintemperatures import ObtainTemperatures
 
###############################################################################

class VeriftyConuction1D:
    
    def __init__(self, H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, ERROR_LIMIT,
                 _qVector, _TooVector, _tVector, _xVector):
        
        # Calling and Initializing Conduction1D Object
        self.C1D = Conduction1D(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, 
                                ERROR_LIMIT, _TooVector, _tVector)
                
        self._qVector = _qVector
        self._TooVector = _TooVector
        self._tVector = _tVector
        self._xVector = _xVector
        
        # Preallocating Space for Core and Surface Temperature Vectors
        self._Temp = np.full(((len(_TooVector)),(len(_xVector))),np.nan)
        
        # Calling getTempeatures Function
        self.getTemperatures()
        
    def getTemperatures(self):
        
        C1D = self.C1D
        
        _qVector = self._qVector
        
        _TooVector = self._TooVector
        outerLength = len(_TooVector)
        
        _tVector = self._tVector
        
        _xVector = self._xVector
        innerLength = len(_xVector)
        
        _Temp = self._Temp
        
        for i in range(outerLength):
            
            C1D._t[i] = _tVector[i]
            
            # Obtaining tempeatures at given locations in _xVector
            for j in range(innerLength):
                
                _Temp[i,j] = C1D._getTemp(_xVector[j])
                
            # Appeding _q value to Conduction1D object
            C1D._q[i] = _qVector[i]


if __name__ == '__main__':

    # Setting all physical parameters
    H = 1
    L = 1
    K = 1
    ALPHA = 1
    
    # Setting Eigenvalues and maximum error for eigenvalue convergence
    NUMBER_OF_EIGENVALUES = 100
    ERROR_LIMIT = 1e-12
    
    # Creating time vector
    dt = 1
    endTime = 20+dt
    _tVector = np.arange(0,endTime,dt)
    N = len(_tVector)
    
    # Creating Position vector where temps are obtained at for given time steps
    _xVector = np.arange(0, (L+(L/100)), L/100)
    
    
    # Calling VerifyConduction1D object for q=0 and Too=0
    _qVector = np.zeros(N)
    _TooVector = np.zeros(N)
    V00 = VeriftyConuction1D(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, ERROR_LIMIT, 
                             _qVector, _TooVector, _tVector, _xVector)
    
    # Calling VerifyConduction1D object for q=0 and Too=1
    _qVector = np.zeros(N)
    _TooVector = np.ones(N)
    V01 = VeriftyConuction1D(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, ERROR_LIMIT, 
                             _qVector, _TooVector, _tVector, _xVector)
    
    # Calling VerifyConduction1D object for q=1 and Too=0
    _qVector = np.ones(N)
    _TooVector = np.zeros(N)
    V10 = VeriftyConuction1D(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, ERROR_LIMIT, 
                             _qVector, _TooVector, _tVector, _xVector)
    
    # Calling VerifyConduction1D object for q=1 and Too=1
    _qVector = np.ones(N)
    _TooVector = np.ones(N)
    V11 = VeriftyConuction1D(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, ERROR_LIMIT, 
                             _qVector, _TooVector, _tVector, _xVector)
    
    # Plotting 
    fig1, axs1 = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='all')
    for i in range(len(_tVector)-4):
        if i == 0:
            # Intiail line in black
            axs1[0,0].plot(_xVector, V00._Temp[i], 'k', label=('t = %.0f' % _tVector[i]))
            axs1[0,1].plot(_xVector, V01._Temp[i], 'k', label=('t = %.0f' % _tVector[i]))
            axs1[1,0].plot(_xVector, V10._Temp[i], 'k', label=('t = %.0f' % _tVector[i]))
            axs1[1,1].plot(_xVector, V11._Temp[i], 'k', label=('t = %.0f' % _tVector[i]))
        else:
            axs1[0,0].plot(_xVector, V00._Temp[i], label=('t = %.0f' % _tVector[i]))
            axs1[0,1].plot(_xVector, V01._Temp[i], label=('t = %.0f' % _tVector[i]))
            axs1[1,0].plot(_xVector, V10._Temp[i], label=('t = %.0f' % _tVector[i]))
            axs1[1,1].plot(_xVector, V11._Temp[i], label=('t = %.0f' % _tVector[i]))
    axs1[0, 0].axvline(x=0, color='k', linestyle='--')
    axs1[0, 0].axvline(x=1, color='k', linestyle='--')
    axs1[0, 0].set_ylabel('Temperature (Celcisus)')
    axs1[0, 0].set_title('qgen=0, Too=0')
    axs1[0, 1].axvline(x=0, color='k', linestyle='--')
    axs1[0, 1].axvline(x=1, color='k', linestyle='--')
    axs1[0, 1].set_title('qgen=0, Too=1')
    axs1[1, 0].axvline(x=0, color='k', linestyle='--')
    axs1[1, 0].axvline(x=1, color='k', linestyle='--')
    axs1[1, 0].set_xlabel('x-position (m)')
    axs1[1, 0].set_ylabel('Temperature (Celcisus)')
    axs1[1, 0].set_title('qgen=1, Too=0')
    axs1[1, 1].axvline(x=0, color='k', linestyle='--')
    axs1[1, 1].axvline(x=1, color='k', linestyle='--')
    axs1[1, 1].set_xlabel('x-position (m)')
    axs1[1, 1].set_title('qgen=1, Too=1')

    # Defining Legend
    axs1[1, 1].legend(loc='upper center', bbox_to_anchor=(-0.1, -0.15), shadow=True, ncol=4)
    fig1.suptitle('Temperatures for 1-D Conduction Solution \n H=1, L=1, K=1, ALPHA=1')

    plt.gcf().subplots_adjust(bottom=0.20)
    plt.show()

    # Printing out Tempeatures for given senarios
    print('qgen=0, Too=0: Steady-State Temperatures')
    print('Core: %.5f' % V00._Temp[-1][0])
    print('Surface: %.5f' % V00._Temp[-1][-1])
    print('')
    print('qgen=0, Too=1: Steady-State Temperatures')
    print('Core: %.5f' % V01._Temp[-1][0])
    print('Surface: %.5f' % V01._Temp[-1][-1])
    print('')
    print('qgen=1, Too=0: Steady-State Temperatures')
    print('Core: %.5f' % V10._Temp[-1][0])
    print('Surface: %.5f' % V10._Temp[-1][-1])
    print('')
    print('qgen=1, Too=1: Steady-State Temperatures')
    print('Core: %.5f' % V11._Temp[-1][0])
    print('Surface: %.5f' % V11._Temp[-1][-1])
    print('')
    
    # Preparing data to be exported
    outputData1 = np.full((len(_xVector), (len(_tVector)+1)), np.nan)
    outputData2 = np.full((len(_xVector), (len(_tVector)+1)), np.nan)
    outputData3 = np.full((len(_xVector), (len(_tVector)+1)), np.nan)
    outputData4 = np.full((len(_xVector), (len(_tVector)+1)), np.nan)
    outputData1[:,0] = _xVector
    outputData2[:,0] = _xVector
    outputData3[:,0] = _xVector
    outputData4[:,0] = _xVector
    for i in range(len(_tVector)):
        outputData1[:,i+1] = V00._Temp[i]
        outputData2[:,i+1] = V01._Temp[i]
        outputData3[:,i+1] = V10._Temp[i]
        outputData4[:,i+1] = V11._Temp[i]
        
    # Exporting Data
    np.savetxt('verifyConduction1DOutput1.dat', outputData1)
    np.savetxt('verifyConduction1DOutput2.dat', outputData2)
    np.savetxt('verifyConduction1DOutput3.dat', outputData3)
    np.savetxt('verifyConduction1DOutput4.dat', outputData4)