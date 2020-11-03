import pickle

import numpy as np
import matplotlib.pyplot as plt
# import PID as PID
import sklearn.preprocessing as skl
import sklearn.neural_network as NN
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split

# new conduction solution and PID
import forwardconduction  as fc
import PIDivmech as PID
from scipy import signal

# Old PID Import
# import PID as PID
import yaml
import csv

import json


###############################################################################

class SteppingClass:
    '''Class that takes discrete steps with Tinf and q arrays 
    and calculates temperature at a given x'''

    # def __init__(self,t,Tinf_array,q_array,L=0.02):
    def __init__(self, h, L, k, alpha):  # initialize class input values
        self.h = h
        self.L = L
        self.k = k
        self.alpha = alpha

    def eigenvalue(self, M, Bi):  # M is number of Eigenvalues
        b = np.zeros(M)
        tol = 1e-9
        for m in range(M):  # b[0]=np.min part...then to get
            if m == 0:
                b[m] = np.minimum(np.sqrt(Bi), np.pi / 2)
            else:
                b[m] = b[m - 1] + np.pi
            err = 1.0
            while np.abs(err) > tol:  # put all of this inside for loop
                f = b[m] * np.sin(b[m]) - Bi * np.cos(b[m])
                df = np.sin(b[m]) + b[m] * np.cos(b[m]) + Bi * np.sin(b[m])
                err = f / df
                b[m] -= err
        return b

    def greensStep(self, x, dt, Tinf_array, q_array):
        # length of Tinf_array and q_array is the number of timesteps
        h = self.h
        k = self.k
        alpha = self.alpha
        L = self.L  # length of slab

        b2 = h * L / k  # Biot Number
        n_iterations = 40  # Need about 40 for full convergence, going low can mess up the ANN
        term1 = 0
        term2 = 0
        n_timesteps = len(q_array)
        bm_arr = self.eigenvalue(n_iterations, b2)
        t = dt * n_timesteps

        for M in range(n_iterations):  # sum over eigenvalues
            bm = bm_arr[M]  # array of eigenvalues
            t2 = dt
            t1 = 0
            sum1 = 0
            sum2 = 0

            for i in range(n_timesteps):  # sum over time steps
                termA = 1 / (bm ** 2 * alpha) * np.exp(-bm ** 2 * alpha / L ** 2 * (t - t2))
                termB = 1 / (bm ** 2 * alpha) * np.exp(-bm ** 2 * alpha / L ** 2 * (t - t1))
                sum1 += (termA - termB) * q_array[i]
                sum2 += (termA - termB) * Tinf_array[i]
                t1 += dt
                t2 += dt

            Fm = (bm ** 2 + b2 ** 2) / (bm ** 2 + b2 ** 2 + b2) * np.cos(bm * x / L)
            term1 += 2 * alpha * L ** 2 / (k * bm) * Fm * np.sin(bm) * sum1
            term2 += 2 * alpha * L * Fm * np.cos(bm) * sum2 * h / k

        return (term1 + term2)


# %%
'''
def testSteppingClass( j ):

    if j == 0:
        # All parms unity and T_inf = 0.  The steady state solution for this
        # block of hard-coded inputs should be T_surf = 1 and T_core = 1.5.
        # Convergence reached at Nm = 20.
        N = 20
        dt = 1
        Tinf = np.zeros( N )
        qgen = np.ones( N )
        L = 1
        G = SteppingClass( 1, L, 1, 1 )
        
    elif j == 1:
        # All parms unity with an oscillating generation
        N = 80
        dt = 1
        times = np.linspace(0, N*dt, N)
        Tinf = np.zeros( N )
        qgen = 1 - np.cos( times/4 )
        L = 1
        G = SteppingClass( 1, L, 1, 1 )

    elif j == 2:
        # All parms unity and qgen = 0. Steady-state solution should be
        # T_surf = T_core = 1.  Convergence reached at Nm = 40
        N = 20
        dt = 1
        Tinf = np.ones( N )
        qgen = np.zeros( N )
        L = 1
        G = SteppingClass( 1, L, 1, 1 )

    elif j == 3:
        # Human params.  qgen = 5600 W/m. Too = 10
        N = 500
        dt = 60
        Tinf = 10 * np.ones( N )
        qgen = 5600 * np.ones( N )
        L = 0.035
        G = SteppingClass( 25, L, 0.613, 0.146e-6 )
       

    coreTemp = np.zeros( N )
    surfTemp = np.zeros( N )

    for i in range( 1, N ):
        coreTemp[i] = G.greensStep( 0, dt, Tinf[:i], qgen[:i] )
        surfTemp[i] = G.greensStep( L, dt, Tinf[:i], qgen[:i] )

    print( coreTemp[-1], surfTemp[-1] )

    return qgen, Tinf, coreTemp, surfTemp   
'''


# N: Number of time steps
# tInit: initial temperature value
def makeESinTinf(N=180, tInit=0, tStart=75):
    tInf = np.ones(N) * tInit  # Constant ambient temperature

    for i in range(N):
        amp = .5  # amplitude of sin wave
        if i >= tStart:
            j = (i - 75) / (N - 75)
            if i <= N - 50:
                tInf[i] = amp * np.exp(-2 * j) * np.sin(2 * 20 * np.pi * j ** 2) + tInit + (
                        i - 75) / N  # Best training data so far
            else:
                tInf[i] = tInf[i - 1] - 0.005

    return tInf


# N: Number of time steps
# tInit: initial temperature value
def makeSinTinf(N=180, tInit=0, tStart=75):
    tInf = np.ones(N) * tInit  # Constant ambient temperature

    for i in range(N):
        amp = .25
        if i >= tStart:
            tInf[i] = amp * np.sin((i - 75) / (N - 75) * 4 * np.pi) + tInit + amp  # simple sine wave for testing

    return tInf


# N: Number of time steps
# tInit: initial temperature value
def makeHighSinTinf(N=180, tInit=0, tStart=75):
    tInf = np.ones(N) * tInit  # Constant ambient temperature

    for i in range(N):
        amp = .25
        if i >= tStart:
            tInf[i] = amp * np.sin((i - 75) / (N - 75) * 6 * np.pi) + tInit + amp  # simple sine wave for testing

    return tInf


# N: Number of time steps
# tInit: initial temperature value
def makeSquareSinTinf(N=180, tInit=0, tStart=75):
    tInf = np.ones(N) * tInit  # Constant ambient temperature
    freq = 2 * np.pi

    for i in range(N):
        amp = .25
        if i >= tStart:
            tInf[i] = amp * (1 - signal.square((i - 75) / (N - 75) * 4 * np.pi))

    return tInf


# N: Number of time steps
# tInit: initial temperature value
def makeTriangleSinTinf(N=180, tInit=0, tStart=75):
    tInf = np.ones(N) * tInit  # Constant ambient temperature
    freq = 2 * np.pi

    for i in range(N):
        amp = .25
        if i >= tStart:
            tInf[i] = amp * (1 - signal.sawtooth((i - 75) / (N - 75) * 4 * np.pi))

    return tInf


# N: Number of time steps
# tInit: initial temperature value
def makeRampTinf(N=180, tInit=0, tStart=75):
    tInf = np.ones(N) * tInit  # Constant ambient temperature
    freq = 2 * np.pi

    for i in range(N):
        if i >= tStart:
            tInf[i] = -1 / 4 * ((i - 75) / (N - 75)) + .5

    return tInf


# makeData
# Uses PID to create training data
# Uses a sine wave that varies in frequency, amplitude, and mean
def makeData(tInf, pid, model, N=180):
    dt = 60  # timestamp

    coreTemp_list = np.zeros(N)  # Preallocate space
    surfTemp_list = np.zeros(N)  # Preallocate space
    qsets = np.zeros(N)  # Preallocate space

    # These numbers come from PIDStripChart
    # Bi = 1.4
    # Fo = .01

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


# makeData()
# create testing Tinf data, but also uses a PID to verify results of ANN against how PID would have performed
'''def makeData():
    qv = 0  # Initial qgen
    tinfv = 0  # Initial Too
    L = 0.035

    G = SteppingClass(25, L, 0.613, 0.146e-6)
    dt = 60
    N = 180

    coreTemp_list = np.zeros(N)  # Preallocate space
    surfTemp_list = np.zeros(N)  # Preallocate space

    Tinfs = np.ones(N) * tinfv  # Constant ambient temperature
    qsets = np.zeros(N)  # Preallocate space

    # New Model and New PID, with inital qset values based on a PID value of 0
    # These declarations are the same as the ones in PIDStripChart
    Bi = 1.4
    Fo = .01

    pid = PID.PID(0.5 / Fo, 0.8 / Fo, 0.0, setpoint=37.0)

    pid.setSampleTime(dt)
    qset = pid.update(0.0)

    model = fc.X23_gToo_I(Bi, Fo, M=100)

    # Set pid in the PID controller with the parameters in PIDparms.yaml
    # Although the PID is declared with certain values, this overwrites them
    with open("PIDparms.yaml", "r") as fobj:
        parms = yaml.load(fobj)
        pid.setKp(parms["Kp"])
        pid.setKi(parms["Ki"])
        pid.setKd(parms["Kd"])
        pid.printInputs()

    

    for i in range(N):
        amp = 12  # amplitude of sin wave
        if i >= 75:
            j = (i - 75) / (N - 75)
            if i <= 150:
                Tinfs[i] = amp * np.exp(-2 * j) * np.sin(2 * 20 * np.pi * j ** 2) + tinfv + (
                        i - 75) / 10  # Best training data so far
            else:
                Tinfs[i] = Tinfs[i - 1] - 0.05

        # Old Model
        #coreTemp_list[i] = G.greensStep(0, dt, Tinfs[:i], qsets[:i])
        #surfTemp_list[i] = G.greensStep(L, dt, Tinfs[:i], qsets[:i])

        qsets[i] = qset
        # New Model
        coreTemp_list[i], surfTemp_list[i] = model.getNextTemp(qset, Tinfs[i])

        #            if i > 25:
        qset = pid.update(coreTemp_list[i])
        #            else:
        #                qset = 4000000 #pid.update(coreTemp_list[i])


    return qsets, Tinfs, coreTemp_list, surfTemp_list'''

# makeNewData()
# create testing Tinf data, but also uses a PID to verify results of ANN against how PID would have performed
def makeNewData():
    qv = 0  # Initial qgen
    tinfv = 0  # Initial Too
    L = 0.035

    G = SteppingClass(25, L, 0.613, 0.146e-6)
    dt = 60
    N = 180

    coreTemp_list = np.zeros(N)  # Preallocate space
    surfTemp_list = np.zeros(N)  # Preallocate space

    Tinfs = np.ones(N) * tinfv  # Constant ambient temperature
    qsets = np.zeros(N)  # Preallocate space

    # New Model and New PID, with inital qset values based on a PID value of 0
    # These declarations are the same as the ones in PIDStripChart
    Bi = 1.4
    Fo = .01

    pid = PID.PID(0.5 / Fo, 0.8 / Fo, 0.0, setpoint=37.0)

    pid.setSampleTime(dt)
    qset = pid.update(0.0)

    model = fc.X23_gToo_I(Bi, Fo, M=100)

    # Set pid in the PID controller with the parameters in PIDparms.yaml
    # Although the PID is declared with certain values, this overwrites them
    with open("PIDparms.yaml", "r") as fobj:
        parms = yaml.load(fobj)
        pid.setKp(parms["Kp"])
        pid.setKi(parms["Ki"])
        pid.setKd(parms["Kd"])
        pid.printInputs()

    '''pid = PID.PID(P=35000, I=1, D=10)
    pid.SetPoint = 37
    pid.setSampleTime(dt)
    qsets[0] = pid.update(0)'''

    for i in range(N):
        amp = 5
        if i >= 75:
            Tinfs[i] = amp * np.sin((i - 75) / (N - 75) * 4 * np.pi) + tinfv + amp  # simple sine wave for testing

        # Old Model
        # coreTemp_list[i] = G.greensStep(0, dt, Tinfs[:i], qsets[:i])
        # surfTemp_list[i] = G.greensStep(L, dt, Tinfs[:i], qsets[:i])

        qsets[i] = qset
        # New Model
        coreTemp_list[i], surfTemp_list[i] = model.getNextTemp(qset, Tinfs[i])

        #            if i > 25:
        qset = pid.update(coreTemp_list[i])
        #            else:
        #                qset = 4000000 #pid.update(coreTemp_list[i])

    return qsets, Tinfs, coreTemp_list, surfTemp_list


# SKlearn
# uses makeDelT function to create matrix to train ANN with current and previous temperature
class SKlearn:

    def makeDelT(self, T):
        N = T.shape[0] - 1
        Tdeld = np.zeros((N, 2))
        #Tdeld = np.zeros((N, 2))

        # The second column is the previous time's value
        for i in range(N):
            #Tdeld[i, 2] = T[i]
            Tdeld[i, 1] = T[i]
            Tdeld[i, 0] = T[i + 1]

        return Tdeld


    # Loads training data from 1DGS_surfT_train.dat, which includes TCore and QTrain from the training function
    # Tests against the training data, and also the testing data in 1DGS_surfT_test.dat
    def trainAndTest(self):
        TTrain, QTrain = np.loadtxt('1DGS_surfT_train.dat')

        scaler = skl.StandardScaler(copy=False)
        scaler.fit(TTrain.reshape(-1, 1))
        scaler.transform(TTrain.reshape(-1, 1))
        # Do the previous time step magic

        T_train = self.makeDelT(TTrain)
        Q_train = QTrain[0:-1]

        # Define and train the NN
        mlp = NN.MLPRegressor(hidden_layer_sizes=(20), max_iter=100000, solver='lbfgs')  # 2,10,1
        mlp.fit(T_train, Q_train)

        ## Verify that the parameters actually give back the original training set
        yhat_train = mlp.predict(T_train)

        '''This begins the testing part.  This could be put into separate 
        functions in the future'''
        TTest = np.loadtxt('1DGS_surfT_test.dat')

        scaler = skl.StandardScaler(copy=False)
        scaler.fit(TTest.reshape(-1, 1))
        scaler.transform(TTest.reshape(-1, 1))
        T_test = self.makeDelT(TTest)
        yhat_test = mlp.predict(T_test)
        return yhat_train, yhat_test

    # Loads training data from 1DGS_surfT_train.dat, which includes TCore and QTrain from the training function
    # Tests 6 times against each set in the testing data in 1DGS_surfT_test.dat
    def trainAndTestAll(self):
        TTrain, QTrain = np.loadtxt('1DGS_surfT_train.dat')

        scaler = skl.StandardScaler(copy=False)
        scaler.fit(TTrain.reshape(-1, 1))
        scaler.transform(TTrain.reshape(-1, 1))

        # Do the previous time step magic
        T_train = self.makeDelT(TTrain)
        Q_train = QTrain[0:-1]

        # Define and train the NN
        mlp = NN.MLPRegressor(hidden_layer_sizes=(10,10,10), max_iter=100000, solver='lbfgs')  # 2,10,1
        mlp.fit(T_train, Q_train)

        # Testing
        #TTest = np.loadtxt('1DGS_surfT_test.dat')

        test_file = open("1DGS_surfT_test", "rb")
        test_file.seek(0)
        TTest = pickle.load(test_file)

        '''with open("1DGS_surfT_train.json", "r") as read_file:
            TTest = json.load(read_file)'''

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

    def Test(self):
        T1, Q1, Tinf1 = np.loadtxt('1DGS_surfT_test.dat')
        T2chop = T1
        Q2chop = Q1

        scaler = skl.StandardScaler(copy=False)
        scaler.fit(T2chop.reshape(-1, 1))
        scaler.transform(T2chop.reshape(-1, 1))
        T_test, Q_test = self.makeDelT(T2chop, Q2chop)
        yhat_test = mlp.predict(T_test)
        return yhat_train, yhat_test


class greensFromSKL:
    def makeData(self, yhat, Tinfs):
        qv = 0  # Initial qgen
        tinfv = 0  # Initial Too
        L = 0.035
        #        Tinfchop, Q0chop = np.loadtxt( 'SKLearnTestWithTrain.dat' )

        Q0chop = yhat  # no longer chopped here

        #    T2, Q2 = np.loadtxt('1dgs_surfT_test_chopped.dat')
        G = SteppingClass(25, L, 0.613, 0.146e-6)
        dt = 60
        N = 300

        coreTemp_list = np.zeros(N - 1)  # Preallocate space
        surfTemp_list = np.zeros(N - 1)  # Preallocate space

        # qsets = np.zeros(N)  # Preallocate space
        #        This is hardcoded for the currentfirst 9 values of PID
        #        to get to 37 degrees... should be changed to variable, along with its
        #        length for every time the values are chopped at 9:]
        # qsets[0:9] = [1295026.16666667, 647299.28262206, 323542.23219183,
        #              161717.44886003, 80832.74270437, 40411.63341844,
        #             20239.53790891, 10232.9286746, 5368.72489392]

        qsets = Q0chop
        #        Tinfs[9:] = Tinfchop

        Bi = 1.4
        Fo = .01
        model = fc.X23_gToo_I(Bi, Fo, M=100)
        for i in range(N - 1):
            # Old Model
            # coreTemp_list[i] = G.greensStep(0, dt, Tinfs[:i], qsets[:i])
            # surfTemp_list[i] = G.greensStep(L, dt, Tinfs[:i], qsets[:i])

            # New Model
            coreTemp_list[i], surfTemp_list[i] = model.getNextTemp(qsets[i], Tinfs[i])

        return qsets, Tinfs, coreTemp_list, surfTemp_list


if __name__ == '__main__':
    Bi = 1.4
    Fo = .01

    # Also have to change N in greensFromSkl around line 450
    N = 300

    numRuns = 6

    # Starting Timestamp for Tinf
    tStart = 10

    # makeSinTinf,makeHighSinTinf,makeESinTinf,makeRampTinf,makeSquareSinTinf,makeTriangleSinTinf
    # train with square and ESin
    trainFunctions = [makeHighSinTinf,makeSquareSinTinf, makeESinTinf]
    testFunctions = [makeSinTinf,makeHighSinTinf,makeESinTinf,makeRampTinf,makeSquareSinTinf,makeTriangleSinTinf]

    # Create model to solve Greens, and the PID controller
    # trainModel = fc.X23_gToo_I(Bi, Fo, M=100)
    # testModel = fc.X23_gToo_I(Bi, Fo, M=100)
    pid = PID.PID(100.0, 10.0, 0.0, setpoint=1.0)

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
        tInfTrain[func.__name__] = (func(N, 0, tStart))

    # Create all testing tinf data
    for func in testFunctions:
        tInfTest[func.__name__] = (func(N, 0, tStart))


    # Create all of the testing data for each testing function using makeData()
    for test in tInfTest:
        # Produce all testing data
        testModel = fc.X23_gToo_I(Bi, Fo, M=100)
        n = makeData(tInfTest[test], pid, testModel, N)
        testQset[test] = n[0]
        testTInf[test] = n[1]
        testCoreTemp[test] = n[2]
        testSurfTemp[test] = n[3]

    # Save All Testing Data
    # Switched to pickle because saving data as dictionary
    # Save TCore, its the only data set needed to test
    test_file = open('1DGS_surfT_test', "wb")
    pickle.dump(testCoreTemp, test_file, pickle.HIGHEST_PROTOCOL)
    test_file.close()
    # np.savetxt('1DGS_surfT_test.dat', testCoreTemp)

    # Loop through Each training function
    for train in tInfTrain:
        print("Training with " + train)

        # Calculate core/surface temp and q values for the PID
        # Save TCore and Q
        trainModel = fc.X23_gToo_I(Bi, Fo, M=100)
        r = makeData(tInfTrain[train], pid, trainModel,
                     N)  # Make the training data using PID controller (All temperatures and q)
        np.savetxt('1DGS_surfT_train.dat', (r[2], r[0]))  # Save the training data


        # Train numRuns of NN
        # Test each against all of the testing functions
        # Save results in yhat_test, which is a list of dictionaries
        yhat_test = []
        for r in range(numRuns):

            SKL = SKlearn()
            yhat_test.append(SKL.trainAndTestAll())  # This trains and tests the ANN

        runNum = 1  # keeps track of which run, used to create keys

        # Loop through each run, which each has one set of data for each testing function
        for run in yhat_test:

            # Loop through each testing function, grab data from the current run
            for test in tInfTest:

                GSK = greensFromSKL()  # Get temperature values from ANN output
                gskTestData = GSK.makeData(run[test], testTInf[test])  # Temp from Testing

                # RMS Calculations
                rmsTestTCore = sqrt(mean_squared_error(testCoreTemp[test][1:], gskTestData[2]))
                rmsTestQ = sqrt(mean_squared_error(testQset[test][11:], run[test][10:]))

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


        '''for test in tInfTest:
            print("Testing: " + test)
            
            testModel = fc.X23_gToo_I(Bi, Fo, M=100)
            ndata = makeData(tInfTest[test], pid, testModel,N)  # Make the testing data using PID controller (All termperatures and q)
            np.savetxt('1DGS_surfT_test.dat', ndata[2])  # Save the testing data
            
            SKL = SKlearn()
            yhat_train, yhat_test = SKL.trainAndTest()  # This trains and tests the ANN

            GSK = greensFromSKL()  # Get temperature values from ANN output
            gskTrainData = GSK.makeData(yhat_train, r[1])  # Temp from Training
            gskTestData = GSK.makeData(yhat_test, ndata[1])  # Temp from Testing

            rmsTrainTCore = sqrt(mean_squared_error(r[2][1:], gskTrainData[2]))
            rmsTrainQ = sqrt(mean_squared_error(r[0][11:], yhat_train[10:]))

            rmsTestTCore = sqrt(mean_squared_error(ndata[2][1:], gskTestData[2]))
            rmsTestQ = sqrt(mean_squared_error(ndata[0][11:], yhat_test[10:]))

            key = 'Train-' + train + ' Test-' + test
            filePath = './plots/lbfgs/1Layer10/' + train + test + '.png'
            dataPath = './data/Processed Files/lbfgs-1Layer10/'

            rmsTrainTCoreVals[key] = rmsTrainTCore
            rmsTrainQVals[key] = rmsTrainQ
            rmsTestTCoreVals[key] = rmsTestTCore
            rmsTestQVals[key] = rmsTestQ

            fig, axs = plt.subplots(2, 2)
            fig.set_size_inches(18.5, 10.5)
            fig.suptitle(key)
            axs[0, 0].plot(r[2], label='PID Core Temp')
            axs[0, 0].plot(r[3], label='PID Surf Temp')
            axs[0, 0].set_title('Train Temperatures')
            axs[0, 0].plot(gskTrainData[2], label='SKL Core Temp')
            axs[0, 0].plot(gskTrainData[3], label='SKL Surface Temp')
            axs[0, 0].plot(r[1], label='Too')
            axs[0, 0].set(xlabel='Time (minutes)', ylabel='Temperature (C)')
            axs[0, 0].text(0, .2, "RMS TCore " + "{:.3f}".format(rmsTrainTCore))
            axs[0, 0].legend(loc=4)

            axs[1, 0].set_title('Generation Values from Testing with Training')
            axs[1, 0].plot(r[0], label='q values (PID)')
            axs[1, 0].plot(yhat_train, label='yhat (SKL)')
            axs[1, 0].legend(loc=1)
            axs[1, 0].text(10, 20, "RMS Q " + "{:.3f}".format(rmsTrainQ))
            axs[1, 0].set(xlabel='Time (minutes)', ylabel='Generation (W)')

            axs[0, 1].plot(ndata[2], label='PID Core Temp')
            axs[0, 1].plot(ndata[3], label='PID Surf Temp')
            axs[0, 1].set_title('Test Temperatures')
            axs[0, 1].plot(gskTestData[2], label='SKL Core Temp')
            axs[0, 1].plot(gskTestData[3], label='SKL Surface Temp')
            axs[0, 1].plot(ndata[1], label='Too')
            axs[0, 1].set(xlabel='Time (minutes)', ylabel='Temperature (C)')
            axs[0, 1].text(.0, .2, "RMS TCore " + "{:.3f}".format(rmsTestTCore))
            axs[0, 1].legend(loc=4)

            axs[1, 1].set_title('Generation Values from Testing with New Data')
            axs[1, 1].plot(ndata[0], label='q values (PID)')
            axs[1, 1].plot(yhat_test, label='yhat (SKL)')
            axs[1, 1].legend(loc=1)
            axs[1, 1].text(10, 20, "RMS Q " + "{:.3f}".format(rmsTestQ))
            axs[1, 1].set(xlabel='Time (minutes)', ylabel='Generation (W)')
            plt.tight_layout()
            plt.savefig(filePath, pad_inches=.3)'''

    '''with open(dataPath + 'rmsTrainTCore.csv', 'w', newline='') as csv_file1:
        writer = csv.writer(csv_file1)
        for key, value in rmsTrainTCoreVals.items():
            writer.writerow([key, value])

    with open(dataPath + 'rmsTrainQ.csv', 'w', newline='') as csv_file2:
        writer = csv.writer(csv_file2)
        for key, value in rmsTrainQVals.items():
            writer.writerow([key, value])'''

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


    '''SKL = SKlearn()
    yhat_train, yhat_test = SKL.trainAndTest()  # This trains and tests the ANN
    GSK = greensFromSKL()  # Get temperature values from ANN output
    gskTrainData = GSK.makeData(yhat_train, r[1])  # Temp from Training
    gskTestData = GSK.makeData(yhat_test, ndata[1])  # Temp from Testing

    rmsTrainTCore = sqrt(mean_squared_error(r[2][1:],gskTrainData[2]))
    rmsTrainQ = sqrt(mean_squared_error(r[0][11:], yhat_train[10:]))

    rmsTestTCore = sqrt(mean_squared_error(ndata[2][1:], gskTestData[2]))
    rmsTestQ = sqrt(mean_squared_error(ndata[0][11:], yhat_test[10:]))

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(r[2], label='PID Core Temp')
    axs[0, 0].plot(r[3], label='PID Surf Temp')
    axs[0, 0].set_title('Train Temperatures')
    axs[0, 0].plot(gskTrainData[2], label='SKL Core Temp')
    axs[0, 0].plot(gskTrainData[3], label='SKL Surface Temp')
    axs[0, 0].plot(r[1], label='Too')
    axs[0, 0].set(xlabel='Time (minutes)', ylabel='y-label')
    axs[0, 0].text(0, .2, "RMS TCore " + "{:.3f}".format(rmsTrainTCore))
    axs[0, 0].legend()

    axs[1, 0].set_title('Generation Values from Testing with Training')
    axs[1, 0].plot(r[0], label='q values (PID)')
    axs[1, 0].plot(yhat_train, label='yhat (SKL)')
    axs[1, 0].legend()
    axs[1, 0].text(10, 20, "RMS Q " + "{:.3f}".format(rmsTrainQ))
    axs[1, 0].set(xlabel='Time (minutes)', ylabel='y-label')

    axs[0, 1].plot(ndata[2], label='PID Core Temp')
    axs[0, 1].plot(ndata[3], label='PID Surf Temp')
    axs[0, 1].set_title('Test Temperatures')
    axs[0, 1].plot(gskTestData[2], label='SKL Core Temp')
    axs[0, 1].plot(gskTestData[3], label='SKL Surface Temp')
    axs[0, 1].plot(ndata[1], label='Too')
    axs[0, 1].set(xlabel='Time (minutes)', ylabel='y-label')
    axs[0, 1].text(.0, .2, "RMS TCore " + "{:.3f}".format(rmsTestTCore))
    axs[0, 1].legend()

    axs[1, 1].set_title('Generation Values from Testing with New Data')
    axs[1, 1].plot(ndata[0], label='q values (PID)')
    axs[1, 1].plot(yhat_test, label='yhat (SKL)')
    axs[1, 1].legend()
    axs[1, 1].text(10, 20, "RMS Q " + "{:.3f}".format(rmsTestQ))
    axs[1, 1].set(xlabel='Time (minutes)', ylabel='y-label')
    plt.tight_layout()
    plt.show()'''

    ''' axs[0, 1].plot(x, y, 'tab:orange')
    axs[0, 1].set_title('Axis [0, 1]')
    axs[1, 0].plot(x, -y, 'tab:green')
    axs[1, 0].set_title('Axis [1, 0]')
    axs[1, 1].plot(x, -y, 'tab:red')
    axs[1, 1].set_title('Axis [1, 1]')'''

    '''plt.figure(0)
    plt.title('Temperatures from Testing with Training')
    plt.plot(r[2], label='PID Core Temp')
    plt.plot(r[3], label='PID Surf Temp')
    plt.plot(gskTrainData[2], label='SKL Core Temp')
    plt.plot(gskTrainData[3], label='SKL Surface Temp')
    plt.plot(r[1], label='Too')
    plt.xlabel('Time (minutes)')
    plt.figtext(.6,.8,"RMS " + "{:.3f}".format(rmsTrainTCore))
    plt.legend()

    plt.figure(1)
    plt.title('Generation Values from Testing with Training')
    plt.plot(r[0], label='q values (PID)')
    plt.plot(yhat_train, label='yhat (SKL)')
    plt.legend()
    plt.figtext(.75, .7, "RMS " + "{:.3f}".format(rmsTrainQ))
    plt.xlabel('Time (minutes)')

    plt.figure(2)
    plt.title('Temperatures from Testing with New Data')
    plt.plot(ndata[2], label='PID Core Temp')
    plt.plot(ndata[3], label='PID Surf Temp')
    plt.plot(gskTestData[2], label='SKL Core Temp')
    plt.plot(gskTestData[3], label='SKL Surface Temp')
    plt.plot(ndata[1], label='Too')
    plt.xlabel('Time (minutes)')
    plt.figtext(.6, .8,"RMS " + "{:.3f}".format(rmsTestTCore))
    plt.legend()

    plt.figure(3)
    plt.title('Generation Values from Testing with New Data')
    plt.plot(ndata[0], label='q values (PID)')
    plt.plot(yhat_test, label='yhat (SKL)')
    plt.xlabel('Time (minutes)')
    plt.legend()
    plt.figtext(.75, .7, "RMS " + "{:.3f}".format(rmsTestQ))
    plt.show()
'''
