###############################################################################
## Importing General Python Modules

import numpy as np

###############################################################################

class Conduction1D:
    
    def __init__(self, H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, ERROR_LIMIT,
                 _TooVector, _tVector):
        
        # Initializing class input values
        self.H = H
        self.L = L
        self.K = K
        self.ALPHA = ALPHA
        self.Bi= self._Bi(H, L, K)
        self._m = self._getEigenvalues(NUMBER_OF_EIGENVALUES, ERROR_LIMIT)

        # Inputting Too and t vectors into class objects
        self._Too = _TooVector
        self._t = _tVector
        
        # Preallocating space for q vectors
        self._q = np.full((len(_TooVector)),np.nan)
        
    def _getTemp(self, x):
        
        H = self.H
        L = self.L
        K = self.K
        ALPHA = self.ALPHA
        Bi = self.Bi
        
        _m = self._m
        
        # New Preallocated Vectors
        _Too = self._Too
        _q = self._q
        '''
        This figures out the length of how many _q entries are currently in
        the Conduction1D by determing how many are not NaN.
        '''
        length = np.sum(~np.isnan(_q))
        '''
        The t-vector length used later in this calculation is one longer than
        the length need in other parts of the problem.
        '''
        _tVecLength = (length+1)
        
        _t = self._t
        '''
        Creates a variable that referneces the times needed for calcuations
        later in this function based on the result above for_tVecLength
        '''
        _tVec = _t[:_tVecLength]
        
        '''
        This is calculating the term infront of the summation term that
        that includes the eigenvalue terms
        '''
        Fm_arr = ((_m**2+Bi**2)/(_m**2+Bi**2+Bi)*np.cos(_m*x/L))
        Coeff = (1/(_m**2*ALPHA))
        
        # Calculating terms associated with the summation
        arr = np.arange(length)[np.newaxis].T
        termA = np.exp(-((_m**2)*ALPHA/(L**2))*(_tVec[-1]-_tVec[(arr+1)]))
        termB = np.exp(-((_m**2)*ALPHA/(L**2))*(_tVec[-1]-_tVec[arr]))
        sigmaResult = Coeff[np.newaxis]*Fm_arr[np.newaxis]*(termA-termB)
        
        # Multiplying summation matrix from step above to vector of q's
        qArray = _q[:length][np.newaxis].T
        termAlpha = (2*ALPHA*L**2/(K*_m))*np.sin(_m)*sigmaResult*qArray
        
        # Multiplying summation matrix from step above to vector of Too's
        TooArray = _Too[:length][np.newaxis].T
        termBeta = 2*ALPHA*L*np.cos(_m)*(H/K)*sigmaResult*TooArray
        
        '''
        Summing all of the terms in the previous iteration to return a
        temperuature for a given location and time
        '''
        return(np.sum(np.sum(termAlpha+termBeta)))
        

    def _getEigenvalues(self, NUMBER_OF_EIGANVALUES, ERROR_LIMIT):
        
        Bi = self.Bi
        
        # Preallocating space for number of eigenvalues
        _m = np.zeros(NUMBER_OF_EIGANVALUES)
        
        for i in range(NUMBER_OF_EIGANVALUES): 
            
            #b[0]=np.min at the beginning...then rest goes to else step
            if i == 0:
                _m[i] = np.minimum(np.sqrt(Bi), np.pi/2)
            else:
                _m[i] = _m[i-1]+np.pi
                
            error = 1.0
            
            # Iterating throgh until convergence is achived
            while (np.abs(error))>(ERROR_LIMIT):
                f = (_m[i]*np.sin(_m[i]))-(Bi*np.cos(_m[i]))
                df = (np.sin(_m[i]))+(_m[i]*np.cos(_m[i]))+(Bi*np.sin(_m[i]))
                error = f/df
                _m[i] -= error
                
        return _m
        
    def _Bi(self, H, L, K):
        
        # Calculating Biot Number
        
        return(H*L/K)