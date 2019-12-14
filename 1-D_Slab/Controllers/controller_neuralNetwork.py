import numpy as np
import sklearn.preprocessing as skl
import sklearn.neural_network as NN
###############################################################################
class SKlearn:
    def makeDelT(self, T, Q):
        N = T.shape[0] - 1
        Tdeld = np.zeros((N, 2))

        # The second column is the previous time's value
        for i in range(N):
            Tdeld[i, 1] = T[i]
            Tdeld[i, 0] = T[i + 1]

        Qdeld = np.zeros(N)
        Qdeld = Q[0:-1]

        return Tdeld, Qdeld

    def trainAndTest(self):
        T, Q, Tinf0 = np.loadtxt('1DGS_surfT_train.dat')#
        Tchop0 = T[9:]#T[9:] then 491 in either spot with :
        Qchop0 = Q[9:]
        Tinfchop0 = Tinf0[491:] #[3:] is last three and [:3] is first three
        #        np.savetxt('SKLearnTestWithTrain.dat',(Tinfchop0,Qchop0))

        scaler = skl.StandardScaler(copy=False)
        scaler.fit(Tchop0.reshape(-1, 1))
        scaler.transform(Tchop0.reshape(-1, 1))
        # Do the previous time step magic
        T_train, Q_train = self.makeDelT(Tchop0, Qchop0)

        # Define and train the NN
        mlp = NN.MLPRegressor(hidden_layer_sizes=(10), max_iter=100000)  # 2,10,1
        mlp.fit(T_train, Q_train)

        ## Verify that the parameters actually give back the original training set
        yhat_train = mlp.predict(T_train)

        '''This begins the testing part.  This could be put into separate 
        functions in the future'''
        T1, Q1, Tinf1 = np.loadtxt('1DGS_surfT_test.dat')#
        T2chop = T1[9:]#9
        Q2chop = Q1[9:] #why does this change?

        scaler = skl.StandardScaler(copy=False)
        scaler.fit(T2chop.reshape(-1, 1))
        scaler.transform(T2chop.reshape(-1, 1))
        T_test, Q_test = self.makeDelT(T2chop, Q2chop)
        yhat_test = mlp.predict(T_test)
        return yhat_train, yhat_test