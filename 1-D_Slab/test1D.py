###############################################################################
## Importing General Python Modules

import numpy as np
import matplotlib.pyplot as plt
# import PID as PID
# import sklearn.preprocessing as skl
# import sklearn.neural_network as NN
# from sklearn.model_selection import train_test_split
import time

###############################################################################
## Importing Files

from conduction1D import Conduction1D
# from piddatacreation import PIDDataCreation
# from neuralnetwork import NeuralNetwork
# from obtaintemperatures import ObtainTemperatures
 
###############################################################################

class TestConuction1D:
    
    def __init__(self, H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, ERROR_LIMIT,
                 _qVector, _TooVector, _tVector):
        
        # Calling and Initializing Conduction1D Object
        self.C1D = Conduction1D(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, 
                                ERROR_LIMIT, _TooVector, _tVector)
                
        self._qVector = _qVector
        
        # Putting Too and T Vectors into Object
        self._TooVector = _TooVector
        self._tVector = _tVector
        
        # Preallocating Space for Core and Surface Temperature Vectors
        self._coreTemp = np.full((len(_TooVector)),np.nan)
        self._surfTemp = np.full((len(_TooVector)),np.nan)
        
        # Calling getTempeatures Function
        self.getTemperatures()
        
    def getTemperatures(self):
        
        C1D = self.C1D
        
        L = C1D.L
        
        _qVector = self._qVector
        
        _TooVector = self._TooVector
        length = len(_TooVector)
        
        _coreTemp = self._coreTemp
        _surfTemp = self._surfTemp
        
        for i in range(length):
            
            # Get Temperature at Core
            _coreTemp[i] = C1D._getTemp(0)
            
            # Get Temperature at Surface
            _surfTemp[i] = C1D._getTemp(L)
            
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
    dt = 0.1
    endTime = 20+dt
    _tVector = np.arange(0,endTime,dt)
    N = len(_tVector)
    
    # Calling TestConduction1D for q=0, Too=0
    _qVector = np.zeros(N)
    _TooVector = np.zeros(N)
    _Too0_q0 = TestConuction1D(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES,
                               ERROR_LIMIT, _qVector, _TooVector, _tVector)
    
    # Calling TestConduction1D for q=1, Too=0
    _qVector = np.ones(N)
    _TooVector = np.zeros(N)
    _Too0_q1 = TestConuction1D(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES,
                               ERROR_LIMIT, _qVector, _TooVector, _tVector)
    
    # Calling TestConduction1D for q=0, Too=1
    _qVector = np.zeros(N)
    _TooVector = np.ones(N)
    _Too1_q0 = TestConuction1D(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES,
                               ERROR_LIMIT, _qVector, _TooVector, _tVector)
    
    # Calling TestConduction1D for q=1, Too=1
    _qVector = np.ones(N)
    _TooVector = np.ones(N)
    _Too1_q1 = TestConuction1D(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES,
                               ERROR_LIMIT, _qVector, _TooVector, _tVector)
    
#    plt.figure(0)
#    plt.title('Temperatures')
#    plt.plot(_Too0_q0._coreTemp, 'b-', label='qgen=0, Too=0, Core')
#    plt.plot(_Too0_q0._surfTemp, 'b--', label='qgen=0, Too=0, Surface')
#    plt.plot(_Too0_q1._coreTemp, 'g-', label='qgen=1, Too=0, Core')
#    plt.plot(_Too0_q1._surfTemp, 'g--', label='qgen=1, Too=0, Surface')
#    plt.plot(_Too1_q0._coreTemp, 'r-', label='qgen=0, Too=1, Core')
#    plt.plot(_Too1_q0._surfTemp, 'r--', label='qgen=0, Too=1, Surface')
#    plt.plot(_Too1_q1._coreTemp, 'k-', label='qgen=1, Too=1, Core')
#    plt.plot(_Too1_q1._surfTemp, 'k--', label='qgen=1, Too=1, Surface')
#    plt.xlabel('Time (seconds)')
#    plt.ylabel('Temperature (Celcius)')
#    plt.legend(loc = 'upper right')
#    plt.show()
#    
#    plt.figure(1)
#    plt.title('Generation Values')
#    plt.plot(_Too0_q0.C1D._q, 'b-', label='qgen=0, Too=0')
#    plt.plot(_Too0_q1.C1D._q, 'g-', label='qgen=1, Too=0')
#    plt.plot(_Too1_q0.C1D._q, 'r-', label='qgen=0, Too=1')
#    plt.plot(_Too1_q1.C1D._q, 'k-', label='qgen=1, Too=1')
#    plt.xlabel('Time (seconds)')
#    plt.ylabel('Generation Values (Watts)')
#    plt.legend()
#    plt.show()
#    
#    # Auxillary testing to verify result
#    _qVector = np.ones(N)
#    _TooVector = 5*np.ones(N)
#    _Too5_q1 = TestConuction1D(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES,
#                               ERROR_LIMIT, _qVector, _TooVector, _tVector)
#    _qVector = np.ones(N)
#    _TooVector = 10*np.ones(N)
#    _Too10_q1 = TestConuction1D(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES,
#                               ERROR_LIMIT, _qVector, _TooVector, _tVector)
#    _qVector = np.ones(N)
#    _TooVector = 15*np.ones(N)
#    _Too15_q1 = TestConuction1D(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES,
#                               ERROR_LIMIT, _qVector, _TooVector, _tVector)
#    
#    plt.figure(2)
#    plt.title('Temperatures, qgen=1')
#    plt.plot(_Too1_q1._coreTemp, 'k-', label='Too=1, Core')
#    plt.plot(_Too1_q1._surfTemp, 'k--', label='Too=1, Surface')
#    plt.plot(_Too5_q1._coreTemp, 'c-', label='Too=5, Core')
#    plt.plot(_Too5_q1._surfTemp, 'c--', label='Too=5, Surface')
#    plt.plot(_Too10_q1._coreTemp, 'm-', label='Too=10, Core')
#    plt.plot(_Too10_q1._surfTemp, 'm--', label='Too=10, Surface')
#    plt.plot(_Too15_q1._coreTemp, 'y-', label='Too=15, Core')
#    plt.plot(_Too15_q1._surfTemp, 'y--', label='Too=15, Surface')
#    plt.xlabel('Time (seconds)')
#    plt.ylabel('Temperature (Celcius)')
#    plt.legend(loc = 'upper right')
#    plt.show()
#    
#    plt.figure(3)
#    plt.title('Generation Values, qgen=1')
#    plt.plot(_Too1_q1.C1D._q, 'k-', label='Too=1')
#    plt.plot(_Too5_q1.C1D._q, 'c-', label='Too=5')
#    plt.plot(_Too10_q1.C1D._q, 'm-', label='Too=10')
#    plt.plot(_Too15_q1.C1D._q, 'y-', label='Too=15')
#    plt.xlabel('Time (seconds)')
#    plt.ylabel('Generation Values (Watts)')
#    plt.legend()
#    plt.show()
    
    # Eigenvalue testing
    
    eigenvaluesRange = np.arange(1,101,1)
    q0 = np.zeros(N)
    q1 = np.ones(N)
    Too0 = np.zeros(N)
    Too1 = np.ones(N)
    _t = np.arange(0,N,dt)
    
    steadyStateCore_q0Too1 = np.full(np.shape(eigenvaluesRange), np.nan)
    steadyStateSurf_q0Too1 = np.full(np.shape(eigenvaluesRange), np.nan)
    steadyStateCore_q1Too0 = np.full(np.shape(eigenvaluesRange), np.nan)
    steadyStateSurf_q1Too0 = np.full(np.shape(eigenvaluesRange), np.nan)
    steadyStateCore_q1Too1 = np.full(np.shape(eigenvaluesRange), np.nan)
    steadyStateSurf_q1Too1 = np.full(np.shape(eigenvaluesRange), np.nan)
    
    for i in range(len(eigenvaluesRange)):
        NUMBER_OF_EIGANVALUES = eigenvaluesRange[i]
        _q0Too1 = TestConuction1D(H, L, K, ALPHA, NUMBER_OF_EIGANVALUES,
                                  ERROR_LIMIT, q0, Too1, _t)
        _q1Too0 = TestConuction1D(H, L, K, ALPHA, NUMBER_OF_EIGANVALUES,
                                  ERROR_LIMIT, q1, Too0, _t)
        _q1Too1 = TestConuction1D(H, L, K, ALPHA, NUMBER_OF_EIGANVALUES,
                                  ERROR_LIMIT, q1, Too1, _t)
        steadyStateCore_q0Too1[i] = _q0Too1._coreTemp[-1]
        steadyStateSurf_q0Too1[i] = _q0Too1._surfTemp[-1]
        steadyStateCore_q1Too0[i] = _q1Too0._coreTemp[-1]
        steadyStateSurf_q1Too0[i] = _q1Too0._surfTemp[-1]
        steadyStateCore_q1Too1[i] = _q1Too1._coreTemp[-1]
        steadyStateSurf_q1Too1[i] = _q1Too1._surfTemp[-1]
    
    steadyStateCore_q0Too1_Error = (steadyStateCore_q0Too1)-1
    steadyStateSurf_q0Too1_Error = (steadyStateSurf_q0Too1)-1
    steadyStateCore_q1Too0_Error = (steadyStateCore_q1Too0)-1.5
    steadyStateSurf_q1Too0_Error = (steadyStateSurf_q1Too0)-1
    steadyStateCore_q1Too1_Error = (steadyStateCore_q1Too1)-2.5
    steadyStateSurf_q1Too1_Error = (steadyStateSurf_q1Too1)-2 
#    
#    plt.figure(4)
#    plt.title('Eigenvalue Testing Error')
#    plt.plot(steadyStateCore_q0Too1_Error)
#    plt.plot(steadyStateSurf_q0Too1_Error)
#    plt.plot(steadyStateCore_q1Too0_Error)
#    plt.plot(steadyStateSurf_q1Too0_Error)
#    plt.plot(steadyStateCore_q1Too1_Error)
#    plt.plot(steadyStateSurf_q1Too1_Error)
#    plt.ylim((-0.1, 0.1))
#    
#    plt.figure(5)
#    fig,axs = plt.subplots(2,1)
#    plt.suptitle('Eigenvalue Testing Error')
#    # Full
#    axs[0].plot(steadyStateCore_q0Too1_Error, label='qgen=0, Too=1, Core')
#    axs[0].plot(steadyStateSurf_q0Too1_Error, label='qgen=0, Too=1, Surf')
#    axs[0].plot(steadyStateCore_q1Too0_Error, label='qgen=1, Too=0, Core')
#    axs[0].plot(steadyStateSurf_q1Too0_Error, label='qgen=1, Too=0, Surf')
#    axs[0].plot(steadyStateCore_q1Too1_Error, label='qgen=1, Too=1, Core')
#    axs[0].plot(steadyStateSurf_q1Too1_Error, label='qgen=1, Too=1, Surf')
#    # First Subplot
#    axs[1].plot(steadyStateCore_q0Too1_Error)
#    axs[1].plot(steadyStateSurf_q0Too1_Error)
#    axs[1].plot(steadyStateCore_q1Too0_Error)
#    axs[1].plot(steadyStateSurf_q1Too0_Error)
#    axs[1].plot(steadyStateCore_q1Too1_Error)
#    axs[1].plot(steadyStateSurf_q1Too1_Error)
#    axs[1].set_ylim((-0.01, 0.01))
#    axs[0].legend()
#    plt.show()
#    
    # Data to save for testing 1D Conduction
    outputData = np.vstack((_tVector, 
                            _Too0_q0._coreTemp, _Too0_q0._surfTemp,
                            _Too0_q1._coreTemp, _Too0_q1._surfTemp,
                            _Too1_q0._coreTemp, _Too1_q0._surfTemp, 
                            _Too1_q1._coreTemp, _Too1_q1._surfTemp)).T
    np.savetxt('test1DOutput.dat', outputData)
    
    # Data to save for testing Eigenvalues needed for conduction solution
    outputData = np.vstack((eigenvaluesRange,
                            steadyStateCore_q0Too1_Error,
                            steadyStateSurf_q0Too1_Error,
                            steadyStateCore_q1Too0_Error,
                            steadyStateSurf_q1Too0_Error,
                            steadyStateCore_q1Too1_Error,
                            steadyStateSurf_q1Too1_Error)).T
    np.savetxt('test1DEigenvalueOutput.dat', outputData)