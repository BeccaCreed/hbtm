
import numpy as np
import math

# Solves the forward conduction problem for arbitrary generation and
# arbitrary free stream temperature using Green's functions with a
# piecewise integration of the generation term and the convection term
# (postfix _D).  The second class does the same except it also
# integrates the initial conditions and caches the solution.  This
# means that the second version (postfix _I) integrates only over one
# time step instead starting from t=0 for every step.

# A word about units:  The solution assumes that the scaled generation
# and scaled connvection are provided as inputs.  The output is a real
# temperature difference.

# Bi = hL/k
# Fo = alpha dt / L^2
# Q = q'''L^2 / k
# H = h L Too / k

# Greg Walker, July 2020

# This class will calculate the center (x=0) temperature and the
# external (x=1) temperature from a generation and convection term.
class X23_gToo_D():
    
    def __init__( self, Bi, Fo, M=10, tol=1e-3 ):
        self._Bi = Bi
        self._M = M
        self._Fo = Fo

        self._gens = []
        self._Toos = []
        
        self._bms = self._getEigenvalues( self._Bi, self._M, tol )
        self._Fms = self._getPrefactors( self._bms, self._Bi )

        self._Gpfs = self._Fms * np.sin( self._bms ) / self._bms**3
        self._Hpfs = self._Fms * np.cos( self._bms ) / self._bms**2

        #print( self._bms )
        #print( self._bms * np.tan( self._bms ) )
        #print( self._Fms )

    def reset( self ):
        self._gens = []
        self._Toos = []


    def _getEigenvalues( self, Bi, M, tol ):
         
        # Preallocating space for number of eigenvalues
        m = np.zeros( M )
        
        for i in range( M ): 
            
            #b[0]=np.min at the beginning...then rest goes to else step
            if i == 0:
                m[i] = np.minimum(np.sqrt(Bi), np.pi/2)
            else:
                m[i] = m[i-1] + np.pi
                
            error = 1.0
            
            # Iterating throgh until convergence is achived
            while np.abs(error) > tol:
                f = (m[i]*np.sin(m[i]))-(Bi*np.cos(m[i]))
                df = (np.sin(m[i]))+(m[i]*np.cos(m[i]))+(Bi*np.sin(m[i]))
                error = f/df
                m[i] -= error
                
        return m
 
    def _getPrefactors( self, bm, Bi ):
        return (bm**2 + Bi**2)/(bm**2 + Bi**2 + Bi)

    # The arguments are the next piecewise constant generation and
    # convection terms in the series. Assume dt is constant
    
    # G = q''' * L^2 / k
    # H = h * Too * L / k = Bi Too
    
    def getNextTemp( self, G, H ):
        self._gens.append( G )
        self._Toos.append( H )

        N = len( self._gens )
        M = self._M
        Bi = self._Bi
        Fo = self._Fo
        bms = self._bms

        Gs = self._gens
        Hs = self._Toos

        T0 = 0 # temperature at the centerline
        T1 = 0 # temperature at the surface
        
        for i in range( N ):
            for m in range ( M ):
                trans = np.exp(-bms[m]**2*Fo*(N-i-1)) -\
                        np.exp(-bms[m]**2*Fo*(N-i))
                T0 += self._Gpfs[m] * Gs[i] * trans
                T0 += self._Hpfs[m] * Hs[i] * trans
                T1 += self._Gpfs[m] * np.cos( bms[m] ) * Gs[i] * trans
                T1 += self._Hpfs[m] * np.cos( bms[m] ) * Hs[i] * trans

        return 2 * T0, 2 * T1

# This version of the forward conduction solution caches the most
# recent temperature distribution and integrates that over one time
# step so we don't have to start over at t=0 every time.
class X23_gToo_I():
    
    def __init__( self, Bi, Fo, M=10, tol=1e-3, init=(0.0, 0.0) ):
        self._Bi = Bi
        self._M = M
        self._Fo = Fo

        # from trial and error this seems like enough discretization
        self._Nx = 41
        self._SS( init )
        
        self._bms = self._getEigenvalues( self._Bi, self._M, tol )
        self._Fms = self._getPrefactors( self._bms, self._Bi )

        trans = np.exp(-self._bms**2 * Fo)
        self._Gpfs = self._Fms * np.sin( self._bms ) *(1-trans)/ self._bms**3
        self._Hpfs = self._Fms * np.cos( self._bms ) *(1-trans)/ self._bms**2
        self._Tpfs = self._Fms * trans / self._bms


    # This provides a steady solution as an initial condition provided
    # Q and H and known.  If not it starts with Ti = 0.0
    def _SS( self, init ):
        Q = init[0]
        H = init[1]
        x = np.linspace( 0, 1, self._Nx )
        self._Tx = Q/2.0*(1-x**2) + Q/self._Bi + H/self._Bi

    # Reinitilize the temperature distribution keeping the properties
    # the same.
    def reset( self ):
        self._Tx = np.zeros( self._Nx )

    def printInputs( self ):
        fmtstr = "# {} = {}" 
        print( fmtstr.format( "Bi", self._Bi ) )
        print( fmtstr.format( "Fo", self._Fo ) )
        print( fmtstr.format( "M", self._M ) )
        #print( fmtstr.format( "tol", self._tol ) )
        print( fmtstr.format( "Nx", self._Nx) )
               
    def _getEigenvalues( self, Bi, M, tol ):
         
        # Preallocating space for number of eigenvalues
        m = np.zeros( M )
        
        for i in range( M ): 
            
            #b[0]=np.min at the beginning...then rest goes to else step
            if i == 0:
                m[i] = np.minimum(np.sqrt(Bi), np.pi/2)
            else:
                m[i] = m[i-1] + np.pi
                
            error = 1.0
            
            # Iterating throgh until convergence is achived
            while np.abs(error) > tol:
                f = (m[i]*np.sin(m[i]))-(Bi*np.cos(m[i]))
                df = (np.sin(m[i]))+(m[i]*np.cos(m[i]))+(Bi*np.sin(m[i]))
                error = f/df
                m[i] -= error
                
        return m
 
    def _getPrefactors( self, bm, Bi ):
        return (bm**2 + Bi**2)/(bm**2 + Bi**2 + Bi)

    # The arguments are the next piecewise constant generation and
    # convection terms in the series. Assume dT is constant
    
    # G = q''' * L^2 / k
    # H = h * d(Too) * L / k = Bi d(Too)
    
    def getNextTemp( self, G, H ):

        M = self._M
        Nx = self._Nx
        Bi = self._Bi
        Fo = self._Fo
        bms = self._bms

        Ta = (self._Tx[1:] + self._Tx[:-1]) / 2.0
        Tx = np.zeros( Nx )

        dx = 1/(Nx - 1)
        
        for m in range ( M ):
            #trans = 1 - np.exp(-bms[m]**2 * Fo)
            #Tm = np.sin( bms[m] * np.linspace(0, 1, Nx) )
            #Ti = np.sum( Ta * (Tm[1:] - Tm[:-1]) )

            Tm = 0
            for j in range( Nx-1 ):
                Tm += Ta[j] * (np.sin( bms[m]*(j+1)*dx ) - np.sin( bms[m]*j*dx ))
            Tf = 0
            #for j in range( Nx ):
            Tf += self._Tpfs[m] * Tm #* np.exp( -bms[m]**2 * Fo )
            Tf += self._Gpfs[m] * G #* trans
            Tf += self._Hpfs[m] * H #* trans
            for j in range( Nx ):
                Tx[j] += 2 * Tf * np.cos( bms[m] * j*dx )


        self._Tx = np.copy( Tx )
        return Tx[0], Tx[-1]
            

# equation for the semi-infinite slab solution
def SIS( x, Fo, Bi ):
    return math.erfc( x/math.sqrt(4*Fo) ) - math.exp( Bi*x + Bi**2*Fo ) * math.erfc( x/math.sqrt(4*Fo) + Bi*math.sqrt(Fo) )

# compare to SS and SIS solutions
def verify():

    # steady state solutions (large Fo)
    Bis = [1, 10, 100]
    Fo = 20
    Nt = 2

    GTs = [0, 1]
    HTs = [0, 1]
    
    print("# G, H, Bi, T_ss(0), T_ss(1), T_GF(0), T_GF(1), %dT(0), %dT(1)" )
    for Bi in Bis:
        for GT in GTs:
            for HT in HTs:
                model = X23_gToo_I( Bi, Fo/Nt, M=100, init=(GT,HT) )
                for i in range ( Nt ):
                    TGs = model.getNextTemp( GT, HT )
                TSs = [GT/2.0 + GT/Bi + HT/Bi, GT/Bi + HT/Bi]
                print( f"{GT} {HT} {Bi:3}\t{TSs[0]:.3}\t{TSs[1]:.3}\t", \
                       f"{TGs[0]:.3}\t{TGs[1]:.3}\t", \
                       f"{(TSs[0]-TGs[0])/TSs[0]:.3}\t{(TSs[1]-TGs[1])/TSs[1]:.3}" )
    print( "---" )


    # compare to SIS for transient component (no generation). Only
    # valid for Fo<0.25 and not too big Bi.
    
    Bis = [1, 10]
    Fos = [0.001, 0.01, 0.1]
    Nt = 2
    GT = 0
    HT = 1

    print( "Fo, Bi, T_sis(0), T_sis(1), T_GF(0), T_GF(1)" )
    for Bi in Bis:
        for Fo in Fos:
            model = X23_gToo_I( Bi, Fo/Nt, M=1000, tol=1e-3 )
            for i in range( Nt ):
                TGs = model.getNextTemp( GT, HT )
            TSs = [SIS(1,Fo,Bi)/Bi, SIS(0,Fo,Bi)/Bi]
            print( f"{Fo}\t{Bi:3}\t{TSs[0]:.3}\t{TSs[1]:.3}\t", \
                   f"{TGs[0]:.3}\t{TGs[1]:.3}\t" )

    
if __name__ == "__main__":
    verify()
