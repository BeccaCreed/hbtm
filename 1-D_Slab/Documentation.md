Documentation/Guide for 1DGSwithSKLearn.py
Patrick Mudge
•	The class SteppingClass is for a single 1D slab, needs to be initialized with its parameters
o	Contains function that will return array of Eigenvalues for each use of the equation
o	Number of Eigenvalues needs to be >~ 40 to get proper convergence.  Lower number of iterations can be used but sometimes mess up the ANN by allowing overshoot from rounding.
•	The function GreensStep takes x, dt, array of Tinf, and array of q.  This will return the temperature up to this point.  This function is called each time step to preserve the initial conditions.  This essentially creates a triple nested for loop.  This could be improved.  The GPU version of the code is one way of helping with this.
o	TermA and B and sum1 and 2 are ways of reducing the equation to avoid repeating floating point operations.
•	MakeData is a function to use PID to create training data.  It uses a sine wave that varies in frequency, amplitude, and mean.  This seems to do well but can probably be improved
•	MakeNewData is a function to mainly create testing Tinf data, but also uses a PID to verify results of ANN against how PID would have performed.
•	SKLearn class uses makeDelT function to create matrix to train ANN with current and previous temperature.  Future work could be to train it with 2 previous temperatures by using function twice.
•	TrainAndTest function outputs core and skin temperatures.  It could be improved by avoiding saving text files and just saving variables.
•	Main function primarily just plots the data.
Documentation/Guide for GPU alteration
•	First time, you need to install pycuda and save all of the modules needed into a larger module
o	module load GCC/6.4.0-2.28 CUDA/9.0.176 Python/3.6.3 OpenMPI/2.1.1 numpy/1.13.1-Python-3.6.3
o	module save pycuda
o	pip install --user pycuda
•	Then in the future, you just need to get onto a GPU node and restore your folder
o	salloc --partition=pascal --gres=gpu:1 --account=sc3260_acc --time=2:00:00 --mem=10G
	This is the command I used since our class used a reservation, not sure how to do it without the reservation
o	module restore pycuda
o	module load GCC CUDA
	This compiler is necessary for compiling the C code.
o	python GreensGPUversion.py
•	The file is essentially just a rewrite of the Stepping function in C, and each eigenvalue (element of M) is calculated on a different GPU simultaneously.  This creates linear speedup with number of eigenvalues.  This should let you use as many eigenvalues as you have GPUs (I ran it with 80) to create more accurate results while developing.
