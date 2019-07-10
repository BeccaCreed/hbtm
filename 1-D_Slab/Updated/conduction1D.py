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
        
    def _calcTempScaled(self, x, i):
        
        L = self.L
        ALPHA = self.ALPHA
        Bi = self.Bi
        _m = self._m
        
        _q = self._q
        
        '''
        This figures out the length of how many _q entries are currently in
        the Conduction1D by determing how many are not NaN, and then adds one
        to this value to set the number of entries from the time vector that
        will be used later in this function.
        '''
        _tVecLength = (len(_q[~np.isnan(_q)])+1)
        
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
        
        # Term is later applied to both parts of the summation
        Coeff = 1/(_m**2*ALPHA)
        
        # Term under summation that contains t2 in the exponent
        term1 =  Coeff*np.exp(-((_m**2)*ALPHA/(L**2))*(_tVec[-1]-_tVec[i+1]))
        
        # Term under the summation that contains t1 in the summation
        term2 =  Coeff*np.exp(-((_m**2)*ALPHA/(L**2))*(_tVec[-1]-_tVec[i]))
        
        return(Fm_arr*(term1-term2))
        
    def _getTemp(self, x):
        
        H = self.H
        L = self.L
        K = self.K
        ALPHA = self.ALPHA
        
        _m = self._m
        
        # New Preallocated Vectors
        _Too = self._Too
        _q = self._q
        length = np.sum(~np.isnan(_q))
        
        # Preallocating space for iteration below        
        term1Vec = np.zeros(length)
        term2Vec = np.zeros(length)
        
        for i in range(length):
                        
            sumStep = self._calcTempScaled(x, i)
            
            # Calculating the term that includes the generation for a  step
            term1Vec[i] = np.sum((2*ALPHA*L**2/(K*_m))*np.sin(_m)*sumStep*(_q[i]))
            # Calculating the term that includes the _Too term for a step
            term2Vec[i] = np.sum(2*ALPHA*L*np.cos(_m)*(H/K)*sumStep*(_Too[i]))
        
        '''
        Summing all of the terms in the previous iteration to return a
        temperuature for a given location and time
        '''
        return(np.sum(term1Vec+term2Vec))
        

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
