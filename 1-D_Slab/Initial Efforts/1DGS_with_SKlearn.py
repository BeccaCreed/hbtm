import numpy as np
import matplotlib.pyplot as plt
import PID as PID
import sklearn.preprocessing as skl
import sklearn.neural_network as NN
from sklearn.model_selection import train_test_split
###############################################################################
    
class SteppingClass:
    '''Class that takes discrete steps with Tinf and q arrays 
    and calculates temperature at a given x'''
    #def __init__(self,t,Tinf_array,q_array,L=0.02):
    def __init__( self, h, L, k, alpha ): #initialize class input values
        self.h = h
        self.L = L
        self.k = k
        self.alpha = alpha
        
    def eigenvalue(self, M, Bi ): #M is number of Eigenvalues
        b = np.zeros( M )
        tol = 1e-9
        for m in range( M ): #b[0]=np.min part...then to get 
            if m == 0:
                b[m] = np.minimum(np.sqrt(Bi), np.pi/2)
            else:
                b[m] = b[m-1] + np.pi
            err = 1.0
            while np.abs( err ) > tol: #put all of this inside for loop
                f = b[m]*np.sin(b[m]) - Bi * np.cos(b[m])
                df = np.sin(b[m]) + b[m] * np.cos(b[m]) + Bi*np.sin(b[m])
                err = f / df
                b[m] -= err
        return b

    def greensStep( self, x, dt, Tinf_array, q_array ): 
        #length of Tinf_array and q_array is the number of timesteps
        h = self.h
        k = self.k
        alpha = self.alpha
        L = self.L #length of slab
        
        b2 = h*L/k #Biot Number
        n_iterations = 20 #Need about 40 for full convergence, going low can mess up the ANN
        term1 = 0
        term2 = 0
        n_timesteps = len(q_array)
        bm_arr = self.eigenvalue(n_iterations,b2)
        t = dt * n_timesteps
        
        for M in range(n_iterations): # sum over eigenvalues
            bm = bm_arr[M] #array of eigenvalues
            t2 = dt
            t1 = 0
            sum1 = 0
            sum2 = 0
            
            for i in range( n_timesteps ): # sum over time steps
                termA = 1/(bm**2*alpha)*np.exp(-bm**2 * alpha/L**2*(t-t2))
                termB = 1/(bm**2*alpha)*np.exp(-bm**2 * alpha/L**2*(t-t1))
                sum1 += (termA - termB)*q_array[i]
                sum2 += (termA - termB)*Tinf_array[i]
                t1 += dt
                t2 += dt

            Fm = (bm**2+b2**2)/(bm**2+b2**2+b2) * np.cos(bm*x/L)
            term1 += 2*alpha*L**2/(k*bm) * Fm * np.sin(bm) * sum1 
            term2 += 2*alpha*L * Fm * np.cos(bm) * sum2*h/k
            
        return (term1+term2)

#%%
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
    
def makeData():
    qv = 0 # Initial qgen
    tinfv = 0 # Initial Too
    L = 0.035
        
    G = SteppingClass( 25, L, 0.613, 0.146e-6 )
    dt = 60
    N = 180
    
    coreTemp_list = np.zeros(N) #Preallocate space
    surfTemp_list = np.zeros(N) #Preallocate space
        
    Tinfs = np.ones(N)*tinfv #Constant ambient temperature
    qsets = np.zeros(N) #Preallocate space
    
    for i in range(N):
        amp = 12
        if i >= 75:
            j = (i-75)/(N-75)
            if i <= 150:
                Tinfs[i] = amp*np.exp(-2*j)*np.sin(2*20*np.pi*j**2) + tinfv + (i - 75)/10 #Best training data so far
            else:
                Tinfs[i] = Tinfs[i-1] - 0.05
        
        coreTemp_list[i] = G.greensStep( 0, dt, Tinfs[:i], qsets[:i] ) #Get temp at core
        surfTemp_list[i] = G.greensStep( L, dt, Tinfs[:i], qsets[:i] ) #Get temp at skin
            
        pid = PID.PID(P=35000, I=1, D=10) 
    '''#These values can be adjusted, but if they are, 
    the values hardcoded in later on to get temperature 
    to 37* need to be adjusted'''
        pid.SetPoint = 37
        pid.setSampleTime(dt)
#            if i > 25:
        qset = pid.update(coreTemp_list[i])
#            else:
#                qset = 4000000 #pid.update(coreTemp_list[i])
        qsets[i] = qset
        
            
    return qsets, Tinfs, coreTemp_list, surfTemp_list

def makeNewData():
    qv = 0 # Initial qgen
    tinfv = 0 # Initial Too
    L = 0.035
        
    G = SteppingClass( 25, L, 0.613, 0.146e-6 )
    dt = 60
    N = 180
    
    coreTemp_list = np.zeros(N) #Preallocate space
    surfTemp_list = np.zeros(N) #Preallocate space
        
    Tinfs = np.ones(N)*tinfv #Constant ambient temperature
    qsets = np.zeros(N) #Preallocate space
    
    for i in range(N):
        amp = 5
        if i >= 75:
            Tinfs[i] = amp*np.sin((i-75)/(N-75)*4*np.pi) + tinfv + amp #simple sine wave for testing
        coreTemp_list[i] = G.greensStep( 0, dt, Tinfs[:i], qsets[:i] )
        surfTemp_list[i] = G.greensStep( L, dt, Tinfs[:i], qsets[:i] )
            
        pid = PID.PID(P=35000, I=1, D=10)
        pid.SetPoint = 37
        pid.setSampleTime(dt)
#            if i > 25:
        qset = pid.update(coreTemp_list[i])
#            else:
#                qset = 4000000 #pid.update(coreTemp_list[i])
        qsets[i] = qset
        
    return qsets, Tinfs, coreTemp_list, surfTemp_list

class SKlearn:
    def makeDelT( self, T, Q ):
        N = T.shape[0] - 1
        Tdeld = np.zeros( (N, 2) )
    
        # The second column is the previous time's value
        for i in range(N):
            Tdeld[i,1] = T[i]
            Tdeld[i,0] = T[i+1]
            
        Qdeld = np.zeros( N )
        Qdeld = Q[0:-1]
    
        return Tdeld, Qdeld
    def trainAndTest(self):
        T,Q,Tinf0 = np.loadtxt( '1DGS_surfT_train.dat')
        Tchop0 = T[9:]
        Qchop0 = Q[9:]
        Tinfchop0 = Tinf0[9:]
#        np.savetxt('SKLearnTestWithTrain.dat',(Tinfchop0,Qchop0))

        scaler = skl.StandardScaler( copy=False )
        scaler.fit( Tchop0.reshape(-1,1) )
        scaler.transform( Tchop0.reshape(-1,1) )
        # Do the previous time step magic
        T_train, Q_train = self.makeDelT( Tchop0, Qchop0 )
        
        # Define and train the NN
        mlp = NN.MLPRegressor( hidden_layer_sizes=(10), max_iter=100000 ) #2,10,1
        mlp.fit( T_train, Q_train )
        
        ## Verify that the parameters actually give back the original training set
        yhat_train = mlp.predict( T_train )

        '''This begins the testing part.  This could be put into separate 
        functions in the future'''
        T1, Q1, Tinf1 = np.loadtxt('1DGS_surfT_test.dat')
        T2chop = T1[9:]
        Q2chop = Q1[9:]
        
        scaler = skl.StandardScaler( copy=False )
        scaler.fit( T2chop.reshape(-1,1) )
        scaler.transform( T2chop.reshape(-1,1) )
        T_test, Q_test = self.makeDelT(T2chop,Q2chop)
        yhat_test = mlp.predict(T_test)
        return yhat_train, yhat_test

class greensFromSKL:
    def makeData(self,yhat,Tinfs):
        qv = 0 # Initial qgen
        tinfv = 0 # Initial Too
        L = 0.035
#        Tinfchop, Q0chop = np.loadtxt( 'SKLearnTestWithTrain.dat' )
        Q0chop = yhat
    #    T2, Q2 = np.loadtxt('1dgs_surfT_test_chopped.dat')
        G = SteppingClass( 25, L, 0.613, 0.146e-6 )
        dt = 60
        N = 180
        
        coreTemp_list = np.zeros(N-1) #Preallocate space
        surfTemp_list = np.zeros(N-1) #Preallocate space
            
        qsets = np.zeros(N) #Preallocate space
#        This is hardcoded for the currentfirst 9 values of PID 
#        to get to 37 degrees... should be changed to variable, along with its 
#        length for every time the values are chopped at 9:]
        qsets[0:9] = [1295026.16666667,  647299.28262206,  323542.23219183,
            161717.44886003,   80832.74270437,   40411.63341844,
             20239.53790891,   10232.9286746 ,    5368.72489392] 
        
        qsets[10:] = Q0chop
#        Tinfs[9:] = Tinfchop
        for i in range(N-1):
            coreTemp_list[i] = G.greensStep( 0, dt, Tinfs[:i], qsets[:i] )
            surfTemp_list[i] = G.greensStep( L, dt, Tinfs[:i], qsets[:i] )
            
        return qsets, Tinfs, coreTemp_list, surfTemp_list



if __name__ == '__main__':
    
    r = makeData() #Make the training data using PID controller (All termperatures and q)
    np.savetxt( '1DGS_surfT_train.dat', (r[3], r[0], r[1]) ) #Save the training data
    ndata = makeNewData() #Make the testing data using PID controller (All termperatures and q)
    np.savetxt('1DGS_surfT_test.dat', (ndata[3], ndata[0], ndata[1])) #Save the testing data
    SKL = SKlearn()
    yhat_train, yhat_test = SKL.trainAndTest() #This trains and tests the ANN
    GSK = greensFromSKL() #Get temperature values from ANN output
    gskTrainData = GSK.makeData(yhat_train,r[1]) #Temp from Training
    gskTestData = GSK.makeData(yhat_test,ndata[1]) #Temp from Testing
    
    

    
    
    plt.figure(0)
    plt.title('Temperatures from Testing with Training')
    plt.plot(r[2],label='PID Core Temp')
    plt.plot(r[3],label='PID Surf Temp')
    plt.plot(gskTrainData[2],label='SKL Core Temp')
    plt.plot(gskTrainData[3],label='SKL Surface Temp')
    plt.plot(r[1],label='Too')
    plt.xlabel('Time (minutes)')
    plt.legend()
    
    plt.figure(1)
    plt.title('Generation Values from Testing with Training')
    plt.plot(r[0][10:],label='q values (PID)')
    plt.plot(yhat_train,label='yhat (SKL)')
    plt.legend()
    plt.xlabel('Time (minutes)')
    plt.show()
    
    plt.figure(2)
    plt.title('Temperatures from Testing with New Data')
    plt.plot(ndata[2],label='PID Core Temp')
    plt.plot(ndata[3],label='PID Surf Temp')
    plt.plot(gskTestData[2],label='SKL Core Temp')
    plt.plot(gskTestData[3],label='SKL Surface Temp')
    plt.plot(ndata[1],label='Too')
    plt.xlabel('Time (minutes)')
    plt.legend()
    
    plt.figure(3)
    plt.title('Generation Values from Testing with New Data')
    plt.plot(ndata[0][10:],label='q values (PID)')
    plt.plot(yhat_test,label='yhat (SKL)')
    plt.xlabel('Time (minutes)')
    plt.legend()
    plt.show()
