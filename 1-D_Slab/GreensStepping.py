import numpy as np
import matplotlib.pyplot as plt

class GreensEquation:
    def __init__(self,t,Tinf_array,q_array,L=1):#0.1913):
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
                b[m] = np.minimum( np.sqrt(Bi), np.pi/2 )
            else:
                b[m] = b[m-1] + np.pi
        
            err = 1.0
            while np.abs( err ) > tol: #put all of this inside for loop
                f = b[m] * np.sin( b[m] ) - Bi * np.cos( b[m] )
                df = np.sin( b[m] ) + b[m] * np.cos( b[m] ) + Bi * np.sin( b[m] )
            
                err = f / df
                b[m] -= err
        
        return b

    def greensStep(self,x):
        Tinf_array = self.Tinf_array
        q_array = self.q_array
        K=1#0.5918 #W/mK
        h =1# 5 #W/(m^2 K)
        alpha =1# 0.146e-6 #m^2/s
        L = self.L
        b2 = h*L/K #Biot Number
        dt = self.dt
        n_iterations = 5
        term1 = 0
        term2 = 0
        n_timesteps = len(q_array)
        bm_arr = self.eigenvalue(n_iterations,b2)
        for M in range(n_iterations):
            bm = bm_arr[M] #array of eigenvalues
            t = self.t
            n_timesteps = len(q_array)
            t2 = dt
            t1 = 0
            sum1 = 0
            sum2 = 0
            for i in range(n_timesteps):  #iterate through qs, Tinfs
                termA = 1/(bm**2*alpha)*np.exp(-bm**2 * alpha/L*(t-t2)) #first term in first sum
                termB = 1/(bm**2*alpha)*np.exp(-bm**2 * alpha/L*(t-t1)) #second term in first sum
                sum1 += (termA - termB)*q_array[i]
                termC = 1/(bm**2*alpha)*np.exp(-bm**2 * alpha/L*(t-t2)) #first term in second sum
                termD = 1/(bm**2*alpha)*np.exp(-bm**2 * alpha/L*(t-t1)) #second term in second sum
                sum2 += (termC - termD)*Tinf_array[i]
                t2 += dt #for each time step, t2 advances
                t1 += dt
            term1 += 2*alpha*L**2/(K*bm)*((bm**2+b2**2)/(bm**2+b2**2+b2))*np.cos(bm*x/L)*np.sin(bm)*sum1
            term2 += 2*alpha*L/(1)*((bm**2+b2**2)/(bm**2+b2**2+b2))*np.cos(bm*x/L)*np.cos(bm)*sum2*h/K
        return (term1+term2)
        
if __name__ == '__main__':
    #This is ramp function and sine function
    
    def makeRampData():
        tinfv = 1
        G=GreensEquation(t=1e7,Tinf_array=[tinfv],q_array=[0]) #t is t from Greens Equation...stepping through
        N=100
        t_initial = G.t
        coreTemp_list = np.zeros(N)
        periphTemp_list = np.zeros(N)
        
        qset = [0]
        Tinf_set = [tinfv]
        for i in range(N):
            qv = i/N
            G.q_array = qset
            G.Tinf_array = Tinf_set
 
            coreTemp_list[i] = G.greensStep(x=0)
            periphTemp_list[i] = G.greensStep(x=G.L)
            qset.append(qv)
            Tinf_set.append(tinfv)
            G.q_array = qset
            G.Tinf_array = Tinf_set
            G.t += t_initial

        return Tinf_set, coreTemp_list, periphTemp_list, qset
            #change Tinf here?
    r = makeRampData()
    
    def makeSineWaveData():
        tinfv = 10

        GE=GreensEquation(t=1e7,Tinf_array=[tinfv],q_array=[0.5])
        N = 100
        t_initial = GE.t
        coreTemp_list = np.ones(N)
        periphTemp_list = np.ones(N)
        qset = [0.5]
        Tinf_set = [tinfv]
        for i in range(N):
            qv = np.sin(6*np.pi*i/N)*0.5 + 0.5
            GE.q_array = qset
            GE.Tinf_array = Tinf_set
            coreTemp_list[i] = GE.greensStep(x=0)
            periphTemp_list[i] = GE.greensStep(x=GE.L)
            qset.append(qv)
            Tinf_set.append(tinfv)
            GE.q_array = qset
            GE.Tinf_array = Tinf_set
            GE.t += t_initial

        return Tinf_set, coreTemp_list, periphTemp_list, qset
    s = makeSineWaveData()

    plt.figure(1)
    plt.plot(r[1],label='Core Temp')
    plt.plot(r[2],label='Peripheral Temp')
    plt.legend()
    plt.plot(r[3],label='Q')
    plt.title('Ramp')

    plt.figure(2)
    plt.plot(s[1],label='Core Temp')
    plt.plot(s[2],label='Peripheral Temp')
    plt.plot(s[3],label='Q')
    plt.title('Sine Wave')
    plt.legend()    
    
    
    
    def makeJumpData():
        tinfv = 5

        GE1=GreensEquation(t=1e3,Tinf_array=[5],q_array=[0])

        N = 100
        t_initial = GE1.t
        coreTemp_list = np.ones(N)
        periphTemp_list = np.ones(N)
        qset = [0]
        Tinf_set = [tinfv]
        for i in range(N):
            if i < 40:
                qv = i
            else:
                qv = i -35
            coreTemp_list[i] = GE1.greensStep(x=0)
            periphTemp_list[i] = GE1.greensStep(x=GE1.L)
            qset.append(qv)
            Tinf_set.append(tinfv)
            GE1.q_array = qset
            GE1.Tinf_array = Tinf_set
            GE1.t += t_initial
        return Tinf_set, coreTemp_list, periphTemp_list, qset
    j = makeJumpData()
    
    plt.figure(3)
    plt.plot(j[1],label='Core Temp')
    plt.plot(j[2],label='Peripheral Temp')
    plt.plot(j[3],label='Q')
    plt.title('Jump')
    plt.legend()   