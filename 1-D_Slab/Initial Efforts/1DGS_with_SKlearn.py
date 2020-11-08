# Design and test NN to control generation in 1-D conduction slab with
# constant internal temeprature and varying external conditions

# Thermal Engineering Lab 2019

import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl
import sklearn.neural_network as NN
from sklearn.metrics import mean_squared_error
from math import sqrt

# conduction solution and PID
import forwardconduction  as fc
import PIDivmech as PID
from scipy import signal

# for file I/O
import csv
import pickle


################
# Tinf functions

# Square wave with a pulse width modulation
# N: Number of time steps
# period: the frequency of the pulses (the PWM spans half the entire array)
# tStart: delay before the function begins
def makePWMTinf(N=180, period=10, tStart=75):
    tInf = np.zeros(N) # allocate the array
    t = np.array(range(N)) # array of time steps
    
    # sig is the PWN signal---sweeps through two cycles in N time steps
    sig = np.sin(4*np.pi*t[tStart:-1]/(N-tStart))
    tInf[tStart:-1] = signal.square((t[tStart:-1]-tStart) * 2 * np.pi / period,
                                    duty=(sig+1)/2 )
    return tInf


# Constant (not very interesting but useful for comparisons)
# The arguments are not used but are retained here to conform to the
# other functions
def makeConstantTinf( N=180, period=10, tStart=75 ):
    return np.zeros(N)


# Simple sin wave
# N: Number of time steps
# period: the frequency of the oscillations
# tStart: delay before the function begins
def makeSinTinf(N=180, period=10, tStart=75):
    tInf = np.zeros(N)  # allocate the array

    for i in range(tStart, N):
        tInf[i] = np.sin((i-tStart) * 2 * np.pi / period) 

    return tInf


# Simple sin wave with higher (doubled) frequency (helper function so
# the frequency doesn't have to be changed manually to compare two sin
# waves)
# N: Number of time steps
# period: the frequency of the oscillations
# tStart: delay before the function begins
def makeHighSinTinf(N=180, period=10, tStart=75):
    return makeSinTinf(N, period/2, tStart)


# Square wave
# N: Number of time steps
# period: the frequency of the oscillations
# tStart: delay before the function begins
def makeSquareTinf(N=180, period=10, tStart=75):
    tInf = np.zeros(N) # allocate

    for i in range(tStart, N):
        tInf[i] = signal.square((i - tStart) / period * 2 * np.pi)

    return tInf


# Square wave
# N: Number of time steps
# period: the frequency of the oscillations
# tStart: delay before the function begins
def makeTriangleTinf(N=180, period=10, tStart=75):
    tInf = np.zeros(N)  # allocate

    for i in range(tStart, N):
        tInf[i] = signal.sawtooth((i-tStart+period/4) / period * 2 * np.pi,
                                  width=0.5)
    return tInf


# Other possible functions for Tinf:
# Ramp (up and down)
# Linear (1 to -1, 0 to 1, 0 to -1, -1 to 1)???
# ESin (not sure what this is or why it would be useful)

# End Tinf functions
####################

# makePIDData
# Uses PID to create training data and to create data for testing the
# NN
def makePIDData(tInf, pid, model, N=180):
    dt = 60  # timestamp

    coreTemp_list = np.zeros(N)  # Preallocate space
    surfTemp_list = np.zeros(N)  # Preallocate space
    qsets = np.zeros(N)  # Preallocate space

    # New Model and New PID, with inital qset values based on a PID value of 0
    # These declarations are the same as the ones in PIDStripChart
    # model = fc.X23_gToo_I(Bi, Fo, M=100)
    # pid = PID.PID(100.0, 10.0, 0.0, setpoint=37.0)
    pid.setSampleTime(dt)
    qset = pid.update(0.0)

    for i in range(N):
        qsets[i] = qset
        coreTemp_list[i], surfTemp_list[i] = model.getNextTemp(qset, tInf[i])
        qset = pid.update(coreTemp_list[i])

    return qsets, tInf, coreTemp_list, surfTemp_list



# SKlearn - does the NN stuff
# Uses makeDelT function to create matrix to train ANN with current
# and previous temperature
class SKlearn:

    def __init__( self, Nt=2 ):
        self._Nt = Nt

    def makeDelT(self, T):
        N = T.shape[0] - self._Nt + 1
        Tdeld = np.zeros((N, self._Nt))

        # The second column is the previous time's value
        for i in range( self._Nt ):
            Tdeld[:,i] = T[i:N+i]
            
        return Tdeld

    
    # Loads training data from 1DGS_surfT_train.dat, which includes
    # TCore and QTrain from the training function Test multiple times
    # against each set in the testing data in 1DGS_surfT_test.dat
    def trainAndTestAll(self):
        TTrain, QTrain = np.loadtxt('1DGS_surfT_train.dat')
        scaler = skl.StandardScaler(copy=False)
        scaler.fit(TTrain.reshape(-1, 1))
        scaler.transform(TTrain.reshape(-1, 1))

        # Do the previous time step magic
        T_train = self.makeDelT(TTrain)
        Q_train = QTrain[0:-(self._Nt-1)]

        # Define and train the NN
        mlp = NN.MLPRegressor(hidden_layer_sizes=(10,10,10),
                              max_iter=100000, solver='lbfgs')
        mlp.fit(T_train, Q_train)

        # Testing

        test_file = open("1DGS_surfT_test", "rb")
        test_file.seek(0)
        TTest = pickle.load(test_file)

        # Holds results for N runs
        QNet = {}

        for key in TTest:
             #Scale Test Function
             scaler = skl.StandardScaler(copy=False)
             scaler.fit(TTest[key].reshape(-1, 1))
             scaler.transform(TTest[key].reshape(-1, 1))
             T_test = self.makeDelT(TTest[key])

             # Test N times, store all data
             QNet[key]= mlp.predict(T_test)

        return QNet


# Peform the forward conduction solution given the qgenss and the Tinfs
class greensFromSKL:

    def __init__( self, fcmodel, Nt=2 ):
        self._Nt = Nt
        self._model = fcmodel
        
    def makeFCData(self, qgens, Tinfs ):

        N = len(qgens)

        coreTemp_list = np.zeros(N)  # Preallocate space
        surfTemp_list = np.zeros(N)  # Preallocate space

        self._model.reset()
        
        for i in range(N):
            coreTemp_list[i], surfTemp_list[i] = self._model.getNextTemp(qgens[i], Tinfs[i])

        return qgens, Tinfs, coreTemp_list, surfTemp_list


if __name__ == '__main__':
    Bi = 2.0
    Fo = 0.1

    N = 300
    Nt = 4   # number of time steps to train the ANN on

    numRuns = 1

    # Starting Timestamp for Tinf
    tStart = 10
    period = 20
    amp = 0.5
    To = 0

    # define the functions to train with and test
    trainFunctions = [makeHighSinTinf, makeSquareTinf, makePWMTinf]
    testFunctions = [makeSinTinf, makeHighSinTinf, makePWMTinf,
                     makeSquareTinf, makeTriangleTinf]

    # Create model to solve Greens, and the PID controller
    # FIXME: The conduction model should know this setpoint to get a
    # good initial tmeprature
    pid = PID.PID(0.5/Fo, 1/Fo, 0.0, setpoint=1.0)
    fcmodel = fc.X23_gToo_I( Bi, Fo, M=100 )

    # Generate Too values for testing and training
    # tInfTrain = makeESinTinf(N,0)
    # tInfTest = makeHighSinTinf(N,0)
    tInfTrain = {}
    tInfTest = {}

    rmsTrainTCoreVals = {}
    rmsTrainQVals = {}
    rmsTestTCoreVals = {}
    rmsTestQVals = {}

    testQset = {}
    testTInf = {}
    testCoreTemp = {}
    testSurfTemp = {}

    keyList = []

    # Generate List of Keys
    # Used to combine same combinations across multiple runs together
    for funcTrain in trainFunctions:
        for funcTest in testFunctions:
            keyList.append('Train-' + funcTrain.__name__ + ' Test-' + funcTest.__name__)

    # Create All training tinf data
    for func in trainFunctions:
        tInfTrain[func.__name__] = (To + amp * func(N, period, tStart))

    # Create all testing tinf data
    for func in testFunctions:
        tInfTest[func.__name__] = (To + amp * func(N, period, tStart))


    # Create all of the data from the PID that will be used for
    # comparison to the NN
    for test in tInfTest:
        fcmodel.reset()
        n = makePIDData(tInfTest[test], pid, fcmodel, N)
        testQset[test] = n[0]
        testTInf[test] = n[1]
        testCoreTemp[test] = n[2]
        testSurfTemp[test] = n[3]

    # Save TCore from the PID testing data with pickle
    test_file = open('1DGS_surfT_test', "wb")
    pickle.dump(testCoreTemp, test_file, pickle.HIGHEST_PROTOCOL)
    test_file.close()

    # Loop through each training function
    for train in tInfTrain:
        print("Training with " + train)

        # Calculate core/surface temp and q values from the PID
        fcmodel.reset()
        r = makePIDData(tInfTrain[train], pid, fcmodel, N)
        np.savetxt('1DGS_surfT_train.dat', (r[2], r[0])) 

        # Train numRuns of NN
        # Test each against all of the testing functions
        # Save results in yhat_test, which is a list of dictionaries
        yhat_test = []
        for r in range(numRuns):
            # This trains and tests the ANN
            SKL = SKlearn( Nt )
            yhat_test.append(SKL.trainAndTestAll())
            
        runNum = 1  # keeps track of which run, used to create keys

        # Loop through each run, which each has one set of data for
        # each testing function
        for run in yhat_test:

            # Loop through each testing function, grab data from the current run
            for test in tInfTest:

                GSK = greensFromSKL( fcmodel, Nt )  # Get temperature values from ANN output
                gskTestData = GSK.makeFCData(run[test], testTInf[test])  # Temp from Testing
                # RMS Calculations
                # FIXME: Why the offset?
                eoa = -(Nt-2)
                rmsTestTCore = sqrt(mean_squared_error(testCoreTemp[test][1:eoa], gskTestData[2]))

                rmsTestQ = sqrt(mean_squared_error(testQset[test][11:eoa], run[test][10:]))

                key = 'Run ' + str(runNum) +': Train-' + train + ' Test-' + test
                plotPath = './plots/Repeated/3Layer10-lbfgs/' + train + test + str(runNum) + '.png'
                dataPath = './data/Repeated/3Layer10-lbfgs/'

                rmsTestTCoreVals[key] = rmsTestTCore
                rmsTestQVals[key] = rmsTestQ

                fig, axs = plt.subplots(2, 1)
                fig.set_size_inches(18.5, 10.5)
                fig.suptitle(key)

                axs[0].plot(testCoreTemp[test], label='PID Core Temp')
                axs[0].plot(testSurfTemp[test], label='PID Surf Temp')
                axs[0].set_title('Test Temperatures')
                axs[0].plot(gskTestData[2], label='SKL Core Temp')
                axs[0].plot(gskTestData[3], label='SKL Surface Temp')
                axs[0].plot(testTInf[test], label='Too')
                axs[0].set(xlabel='Time (minutes)', ylabel='Temperature (C)')
                axs[0].text(.0, .2, "RMS TCore " + "{:.3f}".format(rmsTestTCore))
                axs[0].legend(loc=4)

                axs[1].set_title('Generation Values from Testing with New Data')
                axs[1].plot(testQset[test], label='q values (PID)')
                axs[1].plot(run[test], label='yhat (SKL)')
                axs[1].legend(loc=1)
                axs[1].text(10, 20, "RMS Q " + "{:.3f}".format(rmsTestQ))
                axs[1].set(xlabel='Time (minutes)', ylabel='Generation (W)')

                plt.tight_layout()
                plt.savefig(plotPath, pad_inches=.3)
                plt.close()

            runNum = runNum + 1



    # Save all data in excel files
    with open(dataPath + 'rmsTestTCore.csv', 'w', newline='') as csv_file3:
        writer = csv.writer(csv_file3)
        for key, value in rmsTestTCoreVals.items():
            writer.writerow([key, value])

    with open(dataPath + 'rmsTestQ.csv', 'w', newline='') as csv_file4:
        writer = csv.writer(csv_file4)
        for key, value in rmsTestQVals.items():
            writer.writerow([key, value])

    # Perform Average RMS and STD of RMS calculations
    tCoreAvg = {}
    QAvg = {}
    tCoreSTD = {}
    QSTD = {}

    # Use masterKey to group functions together across the runs
    for masterKey in keyList:
        tCore = []
        Q = []

        # Write data to the same file for TCore
        # ex. all runs for Train makeEsin Test HighSin will be in the same file
        with open(dataPath + '/functionsTCore/' + masterKey + '.csv', 'w', newline='') as csv_file5:
            writer = csv.writer(csv_file5)
            for key, value in rmsTestTCoreVals.items():
                if masterKey in key:
                    writer.writerow([key, value])
                    tCore.append(value)

        # Write data to the same file for Q
        # ex. all runs for Train makeEsin Test HighSin will be in the same file
        with open(dataPath + '/functionsQ/' + masterKey + '.csv', 'w', newline='') as csv_file6:
            writer = csv.writer(csv_file6)
            for key, value in rmsTestQVals.items():
                if masterKey in key:
                    writer.writerow([key, value])
                    Q.append(value)

        # Calculate averages and STD for all data across runs
        tCoreAvg[masterKey] = np.average(tCore)
        QAvg[masterKey] = np.average(Q)
        tCoreSTD[masterKey] = np.std(tCore)
        QSTD[masterKey] = np.std(Q)

    # Save the Average RMS, Average Q, STD Q, and STD RMS
    with open(dataPath + 'AverageRMSQ.csv', 'w', newline='') as csv_file7:
        writer = csv.writer(csv_file7)
        for key, value in QAvg.items():
            writer.writerow([key, value])

    with open(dataPath + 'AverageRMSTCore.csv', 'w', newline='') as csv_file8:
        writer = csv.writer(csv_file8)
        for key, value in tCoreAvg.items():
            writer.writerow([key, value])

    with open(dataPath + 'AverageSTDQ.csv', 'w', newline='') as csv_file9:
        writer = csv.writer(csv_file9)
        for key, value in QSTD.items():
            writer.writerow([key, value])

    with open(dataPath + 'AverageSTDTCore.csv', 'w', newline='') as csv_file10:
        writer = csv.writer(csv_file10)
        for key, value in tCoreSTD.items():
            writer.writerow([key, value])




