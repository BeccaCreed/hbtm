import numpy as np
import matplotlib.pyplot as plt
import controller_pid as pid
import controller_neuralNetwork as nn
import plate as model
###############################################################################
if __name__ == '__main__':
    r = pid.makeData()  # Make the training data using PID controller (All termperatures and q)
    np.savetxt('1DGS_surfT_train.dat', (r[3], r[0], r[1]))  # Save the training data
    ndata = pid.makeNewData()  # Make the testing data using PID controller (All termperatures and q)
    np.savetxt('1DGS_surfT_test.dat', (ndata[3], ndata[0], ndata[1]))  # Save the testing data
    SKL = nn.SKlearn()
    yhat_train, yhat_test = SKL.trainAndTest()  # This trains and tests the ANN
    GSK = model.greensFromSKL()  # Get temperature values from ANN output
    gskTrainData = GSK.makeData(yhat_train, r[1])  # Temp from Training
    gskTestData = GSK.makeData(yhat_test, ndata[1])  # Temp from Testing

    plt.figure(0)
    plt.title('Temperatures from Testing with Training')
    plt.plot(r[2], label='PID Core Temp')
    plt.plot(r[3], label='PID Surf Temp')
    plt.plot(gskTrainData[2], label='SKL Core Temp')
    plt.plot(gskTrainData[3], label='SKL Surface Temp')
    plt.plot(r[1], label='Too')
    plt.xlabel('Time (minutes)')
    plt.legend()

    plt.figure(1)
    plt.title('Generation Values from Testing with Training')
    plt.plot(r[0][10:], label='q values (PID)')
    plt.plot(yhat_train, label='yhat (SKL)')
    plt.legend()
    plt.xlabel('Time (minutes)')
    plt.show()

    plt.figure(2)
    plt.title('Temperatures from Testing with New Data')
    plt.plot(ndata[2], label='PID Core Temp')
    plt.plot(ndata[3], label='PID Surf Temp')
    plt.plot(gskTestData[2], label='SKL Core Temp')
    plt.plot(gskTestData[3], label='SKL Surface Temp')
    plt.plot(ndata[1], label='Too')
    plt.xlabel('Time (minutes)')
    plt.legend()

    plt.figure(3)
    plt.title('Generation Values from Testing with New Data')
    plt.plot(ndata[0][10:], label='q values (PID)')
    plt.plot(yhat_test, label='yhat (SKL)')
    plt.xlabel('Time (minutes)')
    plt.legend()
    plt.show()