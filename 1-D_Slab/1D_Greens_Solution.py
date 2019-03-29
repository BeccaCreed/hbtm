import numpy as np
import matplotlib.pyplot as plt
import PID as PID

class Variables:
    '''This class's sol function returns the solution to the minimized equation, good for long term solution'''
    def __init__(self):
        self.alpha = 0.146e-6
        self.K = 0.5918
        self.h = 5
        self.q = 2.315
        self.Tinf = 10 #283.15 #10*C
        self.L = 0.1913
        self.B2 = self.h*self.L/self.K
    def sol(self,x):
        a = -self.q/2/self.K*x**2 + self.q*self.L/self.h + self.q*self.L**2/2/self.K + self.Tinf
        return a
        
###############################################################################
    
class SteppingClass:
    '''Class that takes discrete steps with Tinf and q arrays 
    and calculates temperature at a given x'''
    #def __init__(self,t,Tinf_array,q_array,L=0.02):
    def __init__( self, h, L, k, alpha ):
        self.h = h
        self.L = L
        self.k = k
        self.alpha = alpha
        
    def eigenvalue(self, M, Bi ): #fix this
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
        #Tinf_array = self.Tinf_array
        #q_array = self.q_array
        
        h = self.h
        k = self.k
        alpha = self.alpha
        L = self.L #length of slab
        
        b2 = h*L/k #Biot Number
        n_iterations = 40 # This determines how many eigenvalues, currently low
                          # for speed, but can be increased for accuracy later
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

if __name__ == '__main__':
    
    r = makeData()
#    r = testSteppingClass( 3 )

    plt.figure(0)
    plt.plot(r[2],label='Core Temp')
    plt.plot(r[3],label='Surf Temp')
    plt.legend()
    plt.figure(1)
    plt.plot(r[0],label='q values')
    plt.legend()
    plt.show()

