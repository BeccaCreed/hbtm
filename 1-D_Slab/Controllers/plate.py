import numpy as np
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

class greensFromSKL:
    def makeData(self, yhat, Tinfs):
        qv = 0  # Initial qgen
        tinfv = 0  # Initial Too
        L = 0.035
        #        Tinfchop, Q0chop = np.loadtxt( 'SKLearnTestWithTrain.dat' )
        Q0chop = yhat #[9:] #this was just Q0chop = yhat
        #print(yhat)
        #    T2, Q2 = np.loadtxt('1dgs_surfT_test_chopped.dat')
        G = SteppingClass(25, L, 0.613, 0.146e-6)
        dt = 60
        N = 500#180

        coreTemp_list = np.zeros(N - 1)  # Preallocate space
        surfTemp_list = np.zeros(N - 1)  # Preallocate space

        qsets = np.zeros(N)  # Preallocate space
        #        This is hardcoded for the currentfirst 9 values of PID
        #        to get to 37 degrees... should be changed to variable, along with its
        #        length for every time the values are chopped at 9:]
        qsets[0:9] = [1295026.16666667, 647299.28262206, 323542.23219183,
                      161717.44886003, 80832.74270437, 40411.63341844,
                      20239.53790891, 10232.9286746, 5368.72489392]

        qsets[10:] = Q0chop #10
        #print(Q0chop)
        #print(qsets)
        #        Tinfs[9:] = Tinfchop
        for i in range(N - 1):
            coreTemp_list[i] = G.greensStep(0, dt, Tinfs[:i], qsets[:i])
            surfTemp_list[i] = G.greensStep(L, dt, Tinfs[:i], qsets[:i])

        return qsets, Tinfs, coreTemp_list, surfTemp_list