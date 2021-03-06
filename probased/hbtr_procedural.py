#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:24:15 2018

@author: Matthew Gerboth

Based on the method by Stolwijk and the code listing for his model in the 
NASA technical report, I am going to implement our model which is going to be
simlar to his, execpt

1) The model will lump all nodes in a segment into a single node
2) The model will attempt to treat blood flow through a segment
3) The model will use SI units

Additionally I am going to try to work to update the coding 
conventions of his model to use some functions to encapsulate things 

TODO add repriation losses to the torso ! 

Parameters ar a mix of tanabe and stolwijk
"""

import numpy as np
from hbtr_pressure import satpress, emax
import collections

# Model Layout 
# Ids to make node names to the place in various places in the 
# Vectors of parameters 
nodeidx = {'Torso' : 0, 'Head': 1, 
           'LUArm' : 2,  'LLArm' : 3,  'LHand' : 4,
           'RUArm' : 5,  'RLArm' : 6,  'RHand' : 7,
           'LULeg' : 8, ' LLLeg' : 9, 'LFoot' : 10,
           'RULeg' : 11, 'RLLeg' : 12, 'RFoot' : 13
           }

# blood flow to distal portion of body
blood_out = {'Torso': ['Head', 'LUArm', 'RUArm', 'LULeg', 'RULeg'],
             'LUArm': ['LLArm'], 'LLArm': ['LHand'],
             'RUArm': ['RLArm'], 'RLArm': ['RHand'],
             'LULeg': ['LLLeg'], 'LLLeg': ['LFoot'],
             'RULeg': ['RLLeg'], 'RLLeg': ['RFoot'],
             'Head' : [], 'LHand' : [], 'RHand' : [],
             'LFoot' : [], 'RFoot' : []
             }

# blood flow from proximal portion of the body
blood_in = {'Torso': [], 'Head' : ['Torso'],
            'LUArm': ['Torso'], 'LLArm': ['LUArm'], 'LHand' : ['LLArm'],
            'RUArm': ['Torso'], 'RLArm': ['RUArm'], 'RHand' : ['RLArm'],
            'LULeg': ['Torso'], 'LLLeg': ['LULeg'], 'LFoot' : ['LLLeg'],
            'RULeg': ['Torso'], 'RLLeg': ['RULeg'], 'RFoot' : ['RLLeg']
             }
# These three dictionaries are going to be used to define blood flows between
# different nodes

# Links as a flat vector # Note that we need to have a 
# no loops in this set of links, and I should build some programatic 
# method to assure that 
links = [[1,2,5,8,11], [], 
         [3], [4],  [], [6],  [7],  [],
         [9], [10], [],  [12], [13], []]

bbf = np.array([0, 48.45,
                2.620, 1.365, 1.121, 2.620, 1.365, 1.121,
                1.749, 0.270, 0.528, 1.749, 0.270, 0.528]) # blood flows in [L/h]


def total_bf(node, lnks, baseflows):
    if lnks[node] == []:
        return baseflows[node]
    else:
        return baseflows[node] \
               + sum([total_bf(n, lnks, baseflows) for n in lnks[node]])

def build_bfcm(links, baseflows):
    mat = np.zeros((len(links),len(links)))
    # for each set of links, create the correct blood flow matrix
    for i, lnks in enumerate(links):
        for j in lnks:
            bf = total_bf(j, links, baseflows)
            mat[i,j], mat[j,i] = bf, bf
    return mat

# matrix of base blood flow coefficents
# rows are sources, collumns are destinatios, matrix should be symetric
bbfcm = np.zeros((14,14))
# and then set all the correct values
bbfcm[0,1],   bbfcm[1,0]    = 51.7, 51.7 # Torso to and from head
bbfcm[0,2],   bbfcm[2,0]    = 5.45, 5.45 # Torso to and form LUArm
bbfcm[2,3],   bbfcm[3,2]    = 2.65, 2.65 # LUArm to and from LLArm
bbfcm[3,4],   bbfcm[4,3]    = 1.20, 1.20 # LLArm to and form LHand
bbfcm[0,5],   bbfcm[5,0]    = 5.45, 5.45 # Torso to and form RUArm
bbfcm[5,6],   bbfcm[6,5]    = 2.65, 2.65 # RUArm to and from RLArm
bbfcm[6,7],   bbfcm[7,6]    = 1.20, 1.20 # RLArm to and form RHand
bbfcm[0,8],   bbfcm[8,0]    = 2.72, 2.72 # Torso to and from LULeg
bbfcm[8,9],   bbfcm[9,8]    = 0.85, 0.85 # LULeg to and from LLLeg
bbfcm[9,10],  bbfcm[10,9]   = 0.56, 0.56 # LLLeg to LFoot
bbfcm[0,11],  bbfcm[11,0]   = 2.72, 2.72 # Torso to and from RULeg
bbfcm[11,12], bbfcm[12,11]  = 0.85, 0.85 # RULeg to and from RLLeg
bbfcm[12,13], bbfcm[13, 12] = 0.56, 0.56 # RLLeg to RFoot

# NODE TEMP
temp = np.array([35.0, 36.1, 
                 34.4, 34.9, 35.3, 34.4, 34.9, 35.3,
                 34.8, 34.3, 34.6, 34.8, 34.3, 34.6]) # temperature of node [C]

# paramters for controled
cap   = np.array([171380.0, 17556.0, 
                  9041.34, 5739.14, 1400.3, 9041.34, 5739.14, 1400.3, 
                  29314.3, 13973.7, 2006.4, 29314.3, 13973.7, 2006.4]) # Heat capacitance  [J/K]
qbase = np.array([59.539, 17.3, 
                  1.264, 0.371, 0.140, 1.264, 0.371, 0.140,
                  1.440, 0.380, 0.313, 1.440, 0.380, 0.313]) # Base Metabolic rate [W]
ebase = np.array([3.753, 0.732,
                  1.394, 1.394, 0.523, 1.394, 1.394, 0.523,
                  3.312, 3.312, 0.720, 3.312, 3.312, 0.720]) # Base sweating rate [W]
sarea = np.array([0.557, 0.140, 
                  0.096, 0.063, 0.050, 0.096, 0.063, 0.050, 
                  0.209, 0.112, 0.056, 0.209, 0.112, 0.056]) # Surface Area [m^2]
radh  = np.array([5.229, 6.391,
                  4.997, 4.997, 3.486, 4.997, 4.997, 3.486,
                  4.648, 4.648, 4.648, 4.648, 4.648, 4.648]) # radiant heat transfer coefficent [W/m^2K]
covh  = np.array([2.498, 3.196, 
                  3.486, 3.486, 3.893, 3.486, 3.486, 3.893,
                  3.196, 3.196, 3.486, 3.196, 3.196, 3.486]) # convective heat transfer coefficent [W/m^2K]

# parameters for controler
tset  = np.array([35.0, 36.1, 
                  34.4, 34.9, 35.3, 34.4, 34.9, 35.3,
                  34.8, 34.3, 34.6, 34.8, 34.3, 34.6]) # set temperature [C]
skinr = np.array([0.493, 0.070,
                  0.023, 0.012, 0.092, 0.023, 0.012, 0.092,
                  0.050, 0.025, 0.017, 0.050, 0.025, 0.017]) # response sensativity fraction
skins = np.array([0.481, 0.081, 
                  0.051, 0.026, 0.016, 0.051, 0.026, 0.016,
                  0.073, 0.036, 0.018, 0.073, 0.036, 0.018]) # Sweating sensativity fraction
skinv = np.array([0.322, 0.320,
                  0.031, 0.016, 0.061, 0.031, 0.016, 0.061,
                  0.092, 0.023, 0.050, 0.092, 0.023, 0.050]) # vasoconstriction sensitivity fraction
skinc = np.array([0.195, 0.022,
                  0.022, 0.022, 0.152, 0.022, 0.022, 0.152,
                  0.022, 0.022, 0.152, 0.022, 0.022, 0.152]) # vasodialation sensitivity fraction
workm = np.array([0.300, 0.000,
                  0.026, 0.014, 0.005, 0.026, 0.014, 0.005,
                  0.201, 0.099, 0.005, 0.201, 0.099, 0.005]) # Work fraction
chilm = np.array([0.850, 0.020,
                  0.004, 0.026, 0.000, 0.004, 0.026, 0.000,
                  0.023, 0.012, 0.000, 0.023, 0.012, 0.000]) # shivering sensitivity fraction

# Single number parameters for controlers
csw, ssw, psw = 371.2, 33.6, 0.0 # sweating controlers
cdl, sdl, pdl = 0.0, 0.0, 24.4 # dialation controlers
cst, sst, pst = 117.0, 7.5, 0.0 # constrition controlers
csh, ssh, psh = 11.5, 11.5, 0 # shiver contorlers
local_temp_coeff = 10

# Here I am going to fake the input and output for the moment 
# TODO: Fake input and output by hard coding things for the moment

# and seting up the exparimental conditions
tair = 20.0 # degrees [C] for the air pressure
rhumid = 0.30 # 30 % relative humidity 
lr = 8.8 # lewis ratio, TODO: MAKE A REAL NUMBER, this is a guess 
pc = 1.067 # heat capacity of blood in Wh/l

# Compute the combined heat transfer coeffs
# at the moment I am going to make the airspead ~ 0 at all points 
combh = (radh + covh) * sarea # combined heat transfer coeff in W/K

# get the parial pressure of the air 
pair = rhumid * satpress(tair)

# Signal calculations
def signals(temp, tset, skinr):
    """Use the parameters 
    : temp, tset, and skinr to calculate
    : error (per segment) 
    : warm  (per segment)
    : cold  (per segment)
    : wrms  (1 per body)
    : clds  (1 per body)
    """
    error = temp - tset
    cold = -np.clip(error, None, 0)
    warm = np.clip(error, 0, None)
    clds = np.sum(skinr * cold)
    wrms = np.sum(skinr * warm)
    return error, cold, warm, clds, wrms 
    
# Comand calculations
def commands(error, cold, warm, clds, wrms):
    """calulates the commnds using the values calculate above
    and a number of sensativity parameters. Commands will update a number
    of things
    """
    # most equation of form a * head_error + b * (wrms - clds) + c * head * body,
    # all are positive or zero
    sweat = np.clip(csw * error[0] + ssw * (wrms - clds) + psw * warm[0] * wrms, 0, None)
    dialt = np.clip(cdl * error[0] + sdl * (wrms - clds) + pdl * warm[0] * wrms, 0, None)
    stric = np.clip(-cst * error[0] - sst * (wrms - clds) + pst * cold[0] * clds, 0, None)
    chill = np.clip(-csh * error[0] - ssh * (wrms - clds) + psh * cold[0] * clds, 0, None)
    return sweat, dialt, stric, chill

# calcualte effected ouputs
# such as blood flow, sweating, and metabolism
def calculate_effected_outputs(error, sweat, dialt, stric, chill):
    """Calculate the effected outputs using the parts of model
    I.e. bloodflow, persperation, and shivering (later add in BAT)
    """
    local_effects = 2.0 ** (error / local_temp_coeff)
    qs = qbase + chill * chilm
    bfs = bbf.copy() # so we dont overwrite the base file 
    # now update based on the dialt and stric signals
    bfs = (bfs + skinv * dialt) / (1 + skinc * stric) * local_effects
    bfm = build_bfcm(links, bfs)
    # now make a new 
    esweat = ebase + sweat * skins * local_effects
    emaxs = emax(temp, pair, lr, covh, sarea) 
    eclip = np.clip(esweat, None, emaxs)
    return bfm, eclip, qs

# now calclate the heat flows for all of the different compoents
def heat_flows(bfs, esweat, qs, temp, tair):
    # all portions not invloving the blood flows
    hfs = qs - esweat - combh * (temp - tair)
    # now for each of the blood flows. Since in and out are morally equivelent
    # and we should use other - self. to get the value 
    for i in range(len(hfs)):
        bfc = bfs[i] # ith row of the matrix
        tdffs = temp - temp[i] # difference between other temp and this temp
        hfs[i] += pc * np.dot(bfc, tdffs) # update the heat flow for all the bloods
        
    return hfs

# printing method for temperatures 
def print_temperatures(step, temps):
    """Print out a line of temperatures as a formated list
    """
    ntemps = list(temps)
    fstring = ["{:.5f}" for t in ntemps]
    pstring = "{:d}\t" + '  '.join(fstring)
    print(pstring.format(step, *ntemps))

# now setup the simulatin
dt = 1 # second

# initial conditions 
print_temperatures(0, temp)

def log_append(log, *args):
    """flatten and append all the args itno the log as a single line
    """
    logline = []
    for logp in args:
        if isinstance(logp, collections.Iterable):
            # we must flatten this into the log line
            for p in logp:
                logline.append(p)
        else:
            logline.append(logp)
    
    log.append(logline)

# logging
templog = []
controllog = []
signallog = []
effectslog = []

for i in range(120000):
    # signals
    error, cold, warm, clds, wrms = signals(temp, tset, skinr)
    # controls
    sweat, dialt, stric, chill = commands(error, cold, warm, clds, wrms)
    # effector outputs
    bfs, esweat, qs = calculate_effected_outputs(error, sweat, dialt, stric, chill)
    # heat flows
    hfs = heat_flows(bfs, esweat, qs, temp, tair)
    dtemp = hfs / cap * dt
    # TODO: adaptive timesteping?, better order integrator? 
    temp = temp + dtemp
    
    log_append(signallog, error, cold, warm, clds, wrms)
    log_append(controllog, sweat, dialt, stric, chill)
    log_append(effectslog, esweat, qs, hfs)
    log_append(templog, temp)
    
    print_temperatures(i + 1, temp) # perhaps come up with some better output
    
# Afterwards save out matlab matrxes if true
matsave = True

if matsave:
    tempmat = np.array(templog)
    ctrlmat = np.array(controllog)
    sgnlmat = np.array(signallog)
    efctmat = np.array(effectslog)
    
    np.savez("test.npz", temps=tempmat, ctrl=ctrlmat, 
             sgnl=sgnlmat, efcts=efctmat)
    
