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

    # Plotting results for Too=0
    plt.figure(4)
    fig4, axs = plt.subplots(3, 3, sharex=True)
    fig4.suptitle('PID Controller Outputs: H=1, L=1, K=1, ALPHA=1, Too=0')
    axs[0, 0].set_title('P=%.2f - Core Temperature' % pVector[0])
    axs[1, 0].set_title('P=%.2f - Surface Temperature' % pVector[0])
    axs[2, 0].set_title('P=%.2f - Generation Values' % pVector[0])
    axs[0, 1].set_title('P=%.2f - Core Temperature' % pVector[1])
    axs[1, 1].set_title('P=%.2f - Surface Temperature' % pVector[1])
    axs[2, 1].set_title('P=%.2f - Generation Values' % pVector[1])
    axs[0, 2].set_title('P=%.2f - Core Temperature' % pVector[2])
    axs[1, 2].set_title('P=%.2f - Surface Temperature' % pVector[2])
    axs[2, 2].set_title('P=%.2f - Generation Values' % pVector[2])
    for i in range(len(pVector)):
        for j in range(len(iVector)):
            for k in range(len(dVector)):
                axs[0, i].plot(_tTrainingSeconds[:-1], result[i][j][k]._coreTemp,
                               color=colors[(3*j)+k], linestyle='-',
                               label=('I=%.2f, D=%.2f' % (iVector[j], dVector[k])))
                axs[1, i].plot(_tTrainingSeconds[:-1], result[i][j][k]._surfTemp,
                               color=colors[(3*j)+k], linestyle='-',
                               label=('I=%.2f, D=%.2f' % (iVector[j], dVector[k])))
                axs[2, i].plot(_tTrainingSeconds[:-1], (result[i][j][k].C1D._q),
                               color=colors[(3 * j) + k], linestyle='-',
                               label=('I=%.2f, D=%.2f' % (iVector[j], dVector[k])))
    axs[2, 0].set_xlabel('Time (Time Constants)')
    axs[2, 1].set_xlabel('Time (Time Constants)')
    axs[2, 2].set_xlabel('Time (Time Constants)')
    axs[0, 0].set_ylabel('Temperatures (Celcius)')
    axs[1, 0].set_ylabel('Temperatures (Celcius)')
    axs[2, 0].set_ylabel('Generation (Watts)')
    axs[0, 1].set_ylabel('Temperatures (Celcius)')
    axs[1, 1].set_ylabel('Temperatures (Celcius)')
    axs[2, 1].set_ylabel('Generation (Watts)')
    axs[0, 2].set_ylabel('Temperatures (Celcius)')
    axs[1, 2].set_ylabel('Temperatures (Celcius)')
    axs[2, 2].set_ylabel('Generation (Watts)')
    axs[0, 0].axhline(y=SET_POINT, color='k', linestyle='--')
    axs[0, 1].axhline(y=SET_POINT, color='k', linestyle='--')
    axs[0, 2].axhline(y=SET_POINT, color='k', linestyle='--')
    axs[2, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=3)
    fig4.subplots_adjust(bottom=0.15)
    plt.show()
    
    # Exporting Data for Too=0
    totalCases = len(pVector)*len(iVector)*len(dVector)
    pLen = len(pVector)
    iLen = len(iVector)
    outputData1 = np.full((N, (totalCases+1)), np.nan)
    outputData2 = np.full((N, (totalCases+1)), np.nan)
    outputData3 = np.full((N, (totalCases+1)), np.nan)
    outputData1[:,0] = _tTrainingSeconds[:-1]
    outputData2[:,0] = _tTrainingSeconds[:-1]
    outputData3[:,0] = _tTrainingSeconds[:-1]
    for i in range(len(pVector)):
        for j in range(len(iVector)):
            for k in range(len(dVector)):
                columnEntry = (i*(pLen**2))+(j*iLen)+k
                outputData1[:,columnEntry+1] = result[i][j][k]._coreTemp
                outputData2[:,columnEntry+1] = result[i][j][k]._surfTemp
                outputData3[:,columnEntry+1] = result[i][j][k].C1D._q      
    np.savetxt('PIDTuningOutput1.dat', outputData1)
    np.savetxt('PIDTuningOutput2.dat', outputData2)
    np.savetxt('PIDTuningOutput3.dat', outputData3)
    
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
    
    # Plotting results for SineWave
    plt.figure(4)
    fig4, axs = plt.subplots(3, 3, sharex=True)
    fig4.suptitle('PID Controller Outputs: H=1, L=1, K=1, ALPHA=1, Too=0.25sin(Pi*t)+0.25')
    axs[0, 0].set_title('P=%.2f - Core Temperature' % pVector[0])
    axs[1, 0].set_title('P=%.2f - Surface Temperature' % pVector[0])
    axs[2, 0].set_title('P=%.2f - Generation Values' % pVector[0])
    axs[0, 1].set_title('P=%.2f - Core Temperature' % pVector[1])
    axs[1, 1].set_title('P=%.2f - Surface Temperature' % pVector[1])
    axs[2, 1].set_title('P=%.2f - Generation Values' % pVector[1])
    axs[0, 2].set_title('P=%.2f - Core Temperature' % pVector[2])
    axs[1, 2].set_title('P=%.2f - Surface Temperature' % pVector[2])
    axs[2, 2].set_title('P=%.2f - Generation Values' % pVector[2])
    for i in range(len(pVector)):
        for j in range(len(iVector)):
            for k in range(len(dVector)):
                axs[0, i].plot(_tTrainingSeconds[:-1], result[i][j][k]._coreTemp,
                               color=colors[(3*j)+k], linestyle='-',
                               label=('I=%.2f, D=%.2f' % (iVector[j], dVector[k])))
                axs[1, i].plot(_tTrainingSeconds[:-1], result[i][j][k]._surfTemp,
                               color=colors[(3*j)+k], linestyle='-',
                               label=('I=%.2f, D=%.2f' % (iVector[j], dVector[k])))
                axs[2, i].plot(_tTrainingSeconds[:-1], (result[i][j][k].C1D._q),
                               color=colors[(3 * j) + k], linestyle='-',
                               label=('I=%.2f, D=%.2f' % (iVector[j], dVector[k])))
    axs[2, 0].set_xlabel('Time (Time Constants)')
    axs[2, 1].set_xlabel('Time (Time Constants)')
    axs[2, 2].set_xlabel('Time (Time Constants)')
    axs[0, 0].set_ylabel('Temperatures (Celcius)')
    axs[1, 0].set_ylabel('Temperatures (Celcius)')
    axs[2, 0].set_ylabel('Generation (Watts)')
    axs[0, 1].set_ylabel('Temperatures (Celcius)')
    axs[1, 1].set_ylabel('Temperatures (Celcius)')
    axs[2, 1].set_ylabel('Generation (Watts)')
    axs[0, 2].set_ylabel('Temperatures (Celcius)')
    axs[1, 2].set_ylabel('Temperatures (Celcius)')
    axs[2, 2].set_ylabel('Generation (Watts)')
    axs[0, 0].axhline(y=SET_POINT, color='k', linestyle='--')
    axs[0, 1].axhline(y=SET_POINT, color='k', linestyle='--')
    axs[0, 2].axhline(y=SET_POINT, color='k', linestyle='--')
    axs[2, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=3)
    fig4.subplots_adjust(bottom=0.15)
    plt.show()
    
    # Exporting SineWave Data
    totalCases = len(pVector)*len(iVector)*len(dVector)
    pLen = len(pVector)
    iLen = len(iVector)
    outputData1 = np.full((N, (totalCases+1)), np.nan)
    outputData2 = np.full((N, (totalCases+1)), np.nan)
    outputData3 = np.full((N, (totalCases+1)), np.nan)
    outputData1[:,0] = _tTrainingSeconds[:-1]
    outputData2[:,0] = _tTrainingSeconds[:-1]
    outputData3[:,0] = _tTrainingSeconds[:-1]
    for i in range(len(pVector)):
        for j in range(len(iVector)):
            for k in range(len(dVector)):
                columnEntry = (i*(pLen**2))+(j*iLen)+k
                outputData1[:,columnEntry+1] = result[i][j][k]._coreTemp
                outputData2[:,columnEntry+1] = result[i][j][k]._surfTemp
                outputData3[:,columnEntry+1] = result[i][j][k].C1D._q      
    np.savetxt('PIDTuningOutput4.dat', outputData1)
    np.savetxt('PIDTuningOutput5.dat', outputData2)
    np.savetxt('PIDTuningOutput6.dat', outputData3)