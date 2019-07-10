###############################################################################
## Importing General Python Modules

import numpy as np

###############################################################################
## Importing Files
import PID as PID
from conduction1D import Conduction1D

###############################################################################

class PIDDataCreation:
    
    def __init__(self, H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, ERROR_LIMIT, 
                 PROPORTIONAL, INTERGRAL, DERIVATIVE, SET_POINT, _TooVector, 
                 _tVector):
        
        # Calling and Initializing Conduction1D Object
        self.C1D = Conduction1D(H, L, K, ALPHA, NUMBER_OF_EIGENVALUES, 
                                ERROR_LIMIT, _TooVector, _tVector)
        
        # Initalizing PID Controller Parameters
        self.PROPORTIONAL = PROPORTIONAL
        self.INTERGRAL = INTERGRAL
        self.DERIVATIVE = DERIVATIVE
        self.SET_POINT = SET_POINT
        
        # Putting Too and T Vectors into PIDDataCreation Object
        self._TooVector = _TooVector
        self._tVector = _tVector
        
        # Preallocating Space for Core and Surface Temperature Vectors
        self._coreTemp = np.full((len(_TooVector)),np.nan)
        self._surfTemp = np.full((len(_TooVector)),np.nan)
        
        # Calling makeData Function
        self.makeData()
        
    def makeData(self):
        
        C1D = self.C1D
        
        PROPORTIONAL = self.PROPORTIONAL
        INTERGRAL = self.INTERGRAL
        DERIVATIVE = self.DERIVATIVE
        SET_POINT = self.SET_POINT
        
        L = C1D.L
        
        _TooVector = self._TooVector
        length = len(_TooVector)
        _tVector = self._tVector
        
        _coreTemp = self._coreTemp
        _surfTemp = self._surfTemp
        
        for i in range(length):
            
            #Get Temperature at Core
            _coreTemp[i] = C1D._getTemp(0)
            
            #Get Temperature at Surface
            _surfTemp[i] = C1D._getTemp(L)
                
            '''
            Calling PID Controller
            First, it updates it with the current temperature and time.
            The controller then returns a generation value.
            '''
            pid = PID.PID(P=PROPORTIONAL, I=INTERGRAL, D=DERIVATIVE) 
            pid.SetPoint = SET_POINT
            pid.setSampleTime(_tVector[i+1]-_tVector[i])
            qSetPoint = pid.update(_coreTemp[i])
            
            '''
            The generation value outputed from the PID Controller is then
            inputted to the _q vector object in the Conduction1D object
            '''
            C1D._q[i] = qSetPoint