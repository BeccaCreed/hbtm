
# Create a scrolling strip chart of the temperatures and forcing
# functions used for tuning a PID to the 1D slab problem.

# Greg Walker, July 2020

from scipy import signal
import forwardconduction as fc
import PIDivmech as PID

import sys, getopt
import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import yaml

# The class handles the looping through the animate function and the
# incrementing of the PID control.

class PIDStripChart():
    
    def __init__( self, tuning_mode=False, Bi=1.0, Fo=1.0 ):

        self.model = fc.X23_gToo_I( Bi, Fo, M=100 )
        self.pid = PID.PID( 0.5/Fo, 0.8/Fo, 0.0, setpoint=10.0 )
        self.Q = self.pid.update( 0 )
        
        self._Bi = Bi
        
        self.ts = []
        self.y_Qs = []
        self.y_Ts = []
        self.y_Es = []
        self.y_Hs = []

        self.stops = []

        # In self_tuning mode, the file "PIDparms.yaml" can be edited
        # by hand and the changes (once saved) are immediately
        # reflected in the PID output
        self.tuning_mode = tuning_mode
        if self.tuning_mode:
            self.setPIDparms()
            
        self.model.printInputs()

    def setPIDparms( self ):
        with open("PIDparms.yaml", "r") as fobj:
            parms = yaml.load( fobj )
            self.pid.setKp( parms["Kp"] )
            self.pid.setKi( parms["Ki"] )
            self.pid.setKd( parms["Kd"] )

    # Provide different forcing functions (manually comment all except
    # the one you want) to see how the PID responds
    def externalforce( self, i ):

        offset = 0
        mag = -20

        if not self.tuning_mode:
            t = i / 500
            sig = np.sin(np.pi/2 + 2 * np.pi * t)
            Too = 0.5*(1 - signal.square(2 * np.pi * 30 * t, duty=(sig + 1)/2))
        else:
            freq = math.pi / 20
            offset = -20
            mag = 5     #try for large and small values
            
            # constant
            #Too = 0
        
            # cos wave
            Too = 0.5*(1 - math.cos(i*freq))

            # square wave
            #Too = 0.5*(1 - signal.square( i*freq ))

            # triangle wave
            #Too = signal.sawtooth( i*freq, 0.5 )

        return self._Bi * (offset + mag * Too)

    # Used by FuncAnimate to loop over the model and PID
    def animate( self, i ):

        # Q = q''' L^2 / k
        # H = h L dToo / k
        
        H = self.externalforce( i )


        print(self.Q,H)
        v = self.model.getNextTemp( self.Q, H )
        print(v[0],v[1])

        self.ts.append( i )
        self.y_Qs.append( self.Q )
        self.y_Ts.append( v[0] )
        self.y_Es.append( v[1] )
        self.y_Hs.append( H/self._Bi )

        # output training data
        if not self.tuning_mode:
            print( self.Q, v[1] )

        # The size of the chart is 100 (hardcoded)
        if i > 99:
            self.y_Qs.pop( 0 )
            self.y_Ts.pop( 0 )
            self.y_Es.pop( 0 )
            self.y_Hs.pop( 0 )
            self.ts.pop( 0 )
            ax.set_xlim( [self.ts[0],self.ts[-1]] )

        line_Q.set_data( self.ts, self.y_Qs )
        line_T.set_data( self.ts, self.y_Ts )
        line_E.set_data( self.ts, self.y_Es )
        line_H.set_data( self.ts, self.y_Hs )

        if self.tuning_mode:
            self.setPIDparms()
    
        self.Q = self.pid.update( v[0] )
    
        return line_Q, line_T, line_E, line_H

    
tuning_mode = True
opts, args = getopt.getopt(sys.argv[1:],"t",["tuning_mode_off"])
for opt, arg in opts:
    if opt in ("-t","--tuning_mode_off"):
        tuning_mode = False
        
# Define the fig for animation
fig, ax = plt.subplots()
ax.set_xlim( [0, 100] )
ax.set_ylim( [-30, 30] )

line_Q, = ax.plot( [], [], label="TQ" )
line_T, = ax.plot( [], [], label="Tc" )
line_E, = ax.plot( [], [], label="Ts" )
line_H, = ax.plot( [], [], label="Too" )
ax.legend( loc="upper left" )
ax.grid()

strip = PIDStripChart( Bi=1.4, Fo=0.01, tuning_mode=tuning_mode )
if strip.tuning_mode:
    frames = None
else:
    frames = range( 500 )

ani = animation.FuncAnimation( fig, strip.animate, frames=frames, repeat=False )
plt.show()
