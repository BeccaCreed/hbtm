from __future__ import print_function
from __future__ import absolute_import
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
import numpy.linalg as la
from pycuda.compiler import SourceModule
import numpy as np

def eigenvalue(M, Bi ):
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

#Still need to figure out n_iterations, inputting qarray and Tinfarray, doubles instead of floats?, fix values, 
mod = SourceModule("""
__global__ void equation(double *q_array, double *Tinf_array, double *Fm, double *bm, double *term1, double *term2, double *temps)
{
  int i = threadIdx.x;
  const double x = 0.035;
  const int n_timesteps = 100000;
  const double L = 0.035;
  const int h = 25;
  const double k = 0.613;
  const double alpha = 0.146E-6;
  const double b2 = h * L / k;
  const double bmsq = pow(bm[i], 2);
  const double b2sq = pow(b2, 2);
  const double Lsq = pow(L, 2);
  const int term = 1/(bmsq*alpha);
  
  int dt = 60;
  int t2 = dt;
  int t1 = 0;
  int j;
  int t = dt*n_timesteps;
  double termA, termB;
  double sum1 = 0;
  double sum2 = 0;
      for (j=0; j < n_timesteps; j++) {
          termA = term*exp(-bmsq * alpha/Lsq*(t-t2));
          termB = term*exp(-bmsq * alpha/Lsq*(t-t1));
          sum1 += (termA - termB) * q_array[j];
          sum2 += (termA - termB) * Tinf_array[j];
          t1 += dt;
          t2 += dt;
          //printf("termA-termB: %f ", termA-termB);
      }
  Fm[i] = (bmsq+b2sq)/(bmsq+b2sq+b2)*cos(bm[i]*x/L);
  
  term1[i] = 2*alpha*Lsq/k/bm[i] * Fm[i] * sin(bm[i]) * sum1;
  term2[i] = 2*alpha*L * Fm[i] * cos(bm[i]) * sum2*h/k;
  temps[i] = term1[i] + term2[i];
}
""")

M = 40 # Number of threads being used

gpu_eq= mod.get_function("equation")
h = 25
L = 0.035
k = 0.613
b2 = h * L / k
b_m = eigenvalue(M,b2).astype(numpy.float64)
#print("b_m: {0}".format(b_m))
#print("cos(b_m[1]): {0}, cos(b_m[3]): {1}".format(np.cos(b_m[1]), np.cos(b_m[3])))
n_timesteps = 100000
Fm = numpy.zeros_like(b_m).astype(numpy.float64)
temp = numpy.zeros_like(b_m).astype(numpy.float64)
term1 = numpy.zeros_like(b_m).astype(numpy.float64)
term2 = numpy.zeros_like(b_m).astype(numpy.float64)
q_array = (np.ones(n_timesteps)*1000).astype(numpy.float64)
Tinf_array = (np.ones(n_timesteps)*15).astype(numpy.float64)
temps = np.zeros_like(b_m).astype(numpy.float64)
x = L
gpu_eq(
       drv.In(q_array), drv.In(Tinf_array), drv.In(Fm), drv.In(b_m), drv.In(term1), drv.In(term2), drv.Out(temps),
       block=(M,1,1))
      


print(temps)
print(sum(temps))
