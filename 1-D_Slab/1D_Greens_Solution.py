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
    def __init__(self,t,Tinf_array,q_array,L=0.02):
        self.q_array = q_array
        self.n_timesteps = len(q_array)
        self.Tinf_array = Tinf_array #Ambient Temp
        self.L=L #Peripheral Temp
        self.t=t
        self.dt=t/len(q_array)
        
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

    def greensStep(self,x):
        Tinf_array = self.Tinf_array
        q_array = self.q_array
        K=0.5918 #W/mK
        h = 70 #should actually be more like 18 W/(m^2 K)
        alpha = 0.146e-6 #m^2/s
        L = self.L #length of slab
        b2 = h*L/K #Biot Number
        dt = self.dt
        n_iterations = 10 #This determines how many eigenvalues, currently low for speed, but can be increased for accuracy later
        term1 = 0
        term2 = 0
        n_timesteps = len(q_array)
        bm_arr = self.eigenvalue(n_iterations,b2)
        t = self.t
        for M in range(n_iterations): #Overall Sum
            bm = bm_arr[M] #array of eigenvalues
            t2 = dt
            t1 = 0 #fix this to be current time
            sum1 = 0
            for i in range(n_timesteps): #inner sums
                termA = 1/(bm**2*alpha)*np.exp(-bm**2 * alpha/L*(t-t2)) #first term in first sum
                termB = 1/(bm**2*alpha)*np.exp(-bm**2 * alpha/L*(t-t1)) #second term in first sum
                sum1 += (termA - termB)
                t1 += dt
                t2 += dt
            term1 += 2*alpha*L**2/(K*bm)*((bm**2+b2**2)/(bm**2+b2**2+b2))*np.cos(bm*x/L)*np.sin(bm)*sum1*q_array[i] #This should be M not i, I think
            term2 += 2*alpha*L/(1)*((bm**2+b2**2)/(bm**2+b2**2+b2))*np.cos(bm*x/L)*np.cos(bm)*sum1*Tinf_array[i]*h/K
        return (term1+term2)
        
if __name__ == '__main__':
    def makeData():
        qv = 0#400000 #This is to set q high to force the temperature up to a reasonable value
        tinfv = 0
        G=SteppingClass(t=60,Tinf_array=[tinfv],q_array=[qv]) #this t becomes dt in stepping function
        t_initial = G.t
        dt = G.dt
        N=180
        coreTemp_list = np.zeros(N) #Preallocate space
        periphTemp_list = np.zeros(N) #Preallocate space
        Tinfs = np.ones(N)*tinfv #Constant ambient temperature
        qsets = np.zeros(N) #Preallocate space
        for i in range(N):
            Tinf_set = Tinfs[0:i+1]           
            G.Tinf_array = Tinf_set
            coreTemp_list[i] = G.greensStep(x=0)
            periphTemp_list[i] = G.greensStep(x=G.L)
            pid = PID.PID(P=35000, I=1, D=10)
            pid.SetPoint = 37
            pid.setSampleTime(dt)
#            if i > 25:
            qset = pid.update(coreTemp_list[i])
#            else:
#                qset = 4000000 #pid.update(coreTemp_list[i])
            qsets[i] = qset
            G.q_array = qsets[0:i+1]
            G.t += t_initial
        return Tinf_set, coreTemp_list, periphTemp_list, qsets
    r = makeData()

    plt.figure(0)
    plt.plot(r[1],label='Core Temp')
#    plt.plot(r[2],label='Peripheral Temp')
    plt.legend()
    plt.figure(1)
    plt.plot(r[3],label='q values')
    plt.legend()

