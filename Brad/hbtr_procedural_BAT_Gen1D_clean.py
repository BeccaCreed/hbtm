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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from hbtr_pressure import satpress, emax
import collections

# Model Layout
# Ids to make node names to the place in various places in the
# Vectors of parameters
nodeidx = {'Torso': 0, 'Head': 1,
           'LUArm': 2, 'LLArm': 3, 'LHand': 4,
           'RUArm': 5, 'RLArm': 6, 'RHand': 7,
           'LULeg': 8, ' LLLeg': 9, 'LFoot': 10,
           'RULeg': 11, 'RLLeg': 12, 'RFoot': 13, 'BAT': 14
           }
#'BAT' is an approximation of a neck for now
# These three dictionaries are going to be used to define blood flows between
# different nodes

# Links as a flat vector # Note that we need to have a
# no loops in this set of links, and I should build some programatic
# method to assure that
links = [[8, 11, 14], [],
         [3], [4], [], [6], [7], [],
         [9], [10], [], [12], [13], [], [1,2,5]]

bbf = np.array([0, 48.45,
                2.620, 1.365, 1.121, 2.620, 1.365, 1.121,
                1.749, 0.270, 0.528, 1.749, 0.270, 0.528, 53.69])  # blood flows in [L/h] #value for BAT not adjusted for veins because head and arm value already included BAT


def total_bf(node, lnks, baseflows):
    if lnks[node] == []:
        return baseflows[node]
    else:
        return baseflows[node] \
               + sum([total_bf(n, lnks, baseflows) for n in lnks[node]])


def build_bfcm(links, baseflows):
    mat = np.zeros((len(links), len(links)))
    # for each set of links, create the correct blood flow matrix
    for i, lnks in enumerate(links):
        for j in lnks:
            bf = total_bf(j, links, baseflows)
            mat[i, j], mat[j, i] = bf, bf
    return mat

# NODE TEMP
temp = np.array([35.0, 36.1,
                 34.4, 34.9, 35.3, 34.4, 34.9, 35.3,
                 34.8, 34.3, 34.6, 34.8, 34.3, 34.6, 36.1])  # temperature of node [C] #BAT temp? just copied head

# paramters for controled
cap = np.array([171380.0, 17556.0,
                9041.34, 5739.14, 1400.3, 9041.34, 5739.14, 1400.3,
                29314.3, 13973.7, 2006.4, 29314.3, 13973.7, 2006.4, 172000.0])  # Heat capacitance  [J/K] #made up BAT, or should I split from torso?
qbase = np.array([50, 17.3,
                  1.264, 0.371, 0.140, 1.264, 0.371, 0.140,
                  1.440, 0.380, 0.313, 1.440, 0.380, 0.313, 9.539])  # Base Metabolic rate [W] #Torso was 59.539 then divided with BAT
ebase = np.array([3.753, 0.732,
                  1.394, 1.394, 0.523, 1.394, 1.394, 0.523,
                  3.312, 3.312, 0.720, 3.312, 3.312, 0.720, 0.732])  # Base sweating rate [W] #Copied Torso for BAT
sarea = np.array([.55, 0.14,
                  0.096, 0.063, 0.05, 0.096, 0.063, 0.050,
                  0.209, 0.112, 0.056, 0.209, 0.112, 0.056, 0.007])  # Surface Area [m^2] #Took .007 off Torso for BAT
radh = np.array([5.229, 6.391,
                 4.997, 4.997, 3.486, 4.997, 4.997, 3.486,
                 4.648, 4.648, 4.648, 4.648, 4.648, 4.648, 5.229])  # radiant heat transfer coefficent [W/m^2K] #Copied Torso for BAT
covh = np.array([2.498, 3.196,
                 3.486, 3.486, 3.893, 3.486, 3.486, 3.893,
                 3.196, 3.196, 3.486, 3.196, 3.196, 3.486, 2.498])  # convective heat transfer coefficent [W/m^2K] #Copied Torso for BAT

# parameters for controler
tset = np.array([35.0, 36.1,
                 34.4, 34.9, 35.3, 34.4, 34.9, 35.3,
                 34.8, 34.3, 34.6, 34.8, 34.3, 34.6, 36.0])  # set temperature [C] #Guesstimated from BAT Group memory
skinr = np.array([0.4428, 0.070,
                  0.023, 0.012, 0.092, 0.023, 0.012, 0.092,
                  0.050, 0.025, 0.017, 0.050, 0.025, 0.017, 0.0492])  # response sensativity fraction #For BAT, divided Torso
skins = np.array([0.481, 0.081,
                  0.051, 0.026, 0.016, 0.051, 0.026, 0.016,
                  0.073, 0.036, 0.018, 0.073, 0.036, 0.018, 0.081])  # Sweating sensativity fraction #Copied Head for BAT
skinv = np.array([0.322, 0.320,
                  0.031, 0.016, 0.061, 0.031, 0.016, 0.061,
                  0.092, 0.023, 0.050, 0.092, 0.023, 0.050, 0.321])  # vasoconstriction sensitivity fraction #BAT in between head and torso value
skinc = np.array([0.195, 0.022,
                  0.022, 0.022, 0.152, 0.022, 0.022, 0.152,
                  0.022, 0.022, 0.152, 0.022, 0.022, 0.152, 0.195])  # vasodialation sensitivity fraction #Copied Torso for BAT
chilm = np.array([0.850, 0.020,
                  0.004, 0.026, 0.000, 0.004, 0.026, 0.000,
                  0.023, 0.012, 0.000, 0.023, 0.012, 0.000, 0.000])  # shivering sensitivity fraction #0 for BAT? but 0.02 for BATness

# Single number parameters for controlers
csw, ssw, psw = 371.2, 33.6, 0.0  # sweating controlers
cdl, sdl, pdl = 0.0, 0.0, 24.4  # dialation controlers
cst, sst, pst = 117.0, 7.5, 0.0 #50, 2, 0 # # constrition controlers
csh, ssh, psh = 11.5, 11.5, 0  # shiver contorlers
local_temp_coeff = 10

# Here I am going to fake the input and output for the moment
# TODO: Fake input and output by hard coding things for the moment

# and seting up the exparimental conditions
rhumid = 0.30 # 30 % relative humidity
lr = 8.8  # lewis ratio, TODO: MAKE A REAL NUMBER, this is a guess
pc = 1.067  # heat capacity of blood in Wh/l

# Compute the combined heat transfer coeffs
# at the moment I am going to make the airspead ~ 0 at all points
combh = (radh + covh) * sarea  # combined heat transfer coeff in W/K

# get the parial pressure of the air, further down in code now, can use this way if one temp for entire run
#ttair=20
#pair = rhumid * satpress(ttair)

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
    # most equation of form a * head_error + b * (wrms - clds) + c * head * body, changed by Becca 7/20
    # all are positive or zero
    sweat = np.clip(csw * error + ssw * (wrms - clds) + psw * warm * wrms, 0, None)
    dialt = np.clip(cdl * error + sdl * (wrms - clds) + pdl * warm * wrms, 0, None)
    stric = np.clip(-cst * error - sst * (wrms - clds) + pst * cold * clds, 0, None)
    chill = np.clip(-csh * error - ssh * (wrms - clds) + psh * cold * clds, 0, None)

    return sweat, dialt, stric, chill


# calcualte effected ouputs
# such as blood flow, sweating, and metabolism, !bat guesses by Becca
def calculate_effected_outputs(error, sweat, dialt, stric, chill):
    """Calculate the effected outputs using the parts of model
    I.e. bloodflow, persperation, and shivering (later add in BAT)
    """
    local_effects = 2.0 ** (error / local_temp_coeff)
    qs = qbase + chill * chilm
    bfs = bbf.copy()  # so we dont overwrite the base file
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
        bfc = bfs[i]  # ith row of the matrix
        tdffs = temp - temp[i]  # difference between other temp and this temp
        hfs[i] += pc * np.dot(bfc, tdffs)  # update the heat flow for all the bloods
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
dt = 1  # second

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
sweatlog = []
shiverlog = []
constrictionlog = []
dilationlog = []
signallog = []
effectslog = []
airTemplog = []

tair = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(15000):
    #Room temperature
    if i<400: #10800
        for j in range(len(temp)):
            tair[j] = 50
            pair = rhumid * satpress(tair[j])
    #elif i<800:
    #   tair = 35.9
    #elif i<1200:
    #    tair = 35.9
    #elif i<1600:
    #    tair = 35.9
    #else:
    #    tair = 35.9
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
    log_append(sweatlog, sweat)
    log_append(shiverlog, chill)
    log_append(dilationlog, dialt)
    log_append(constrictionlog, stric)
    log_append(effectslog, esweat, qs, hfs)
    log_append(templog, temp)
    log_append(airTemplog, tair)

    print_temperatures(i + 1, temp)  # perhaps come up with some better output
np.save('Air_oneHour.npy', airTemplog)
# Afterwards save out matlab matrxes if true
matsave = True

if matsave:
    tempmat = np.array(templog)
    sweatmat = np.array(sweatlog)
    shivermat = np.array(shiverlog)
    constrictionmat = np.array(constrictionlog)
    dilationmat = np.array(dilationlog)
    sgnlmat = np.array(signallog)
    efctmat = np.array(effectslog)
    tairmat = np.array(airTemplog)

    #np.savez("test.npz", temps=tempmat, sweat=sweatmat, shiver=shivermat, constriction=constrictionmat, dilation=dilationmat,
             #bat=batmat, sgnl=sgnlmat, efcts=efctmat)

head= [item[1] for item in tempmat]
luarm = [item[2] for item in tempmat]
llarm= [item[3] for item in tempmat]
lhand= [item[4] for item in tempmat]
luleg= [item[8] for item in tempmat]
llleg= [item[9] for item in tempmat]
lfoot= [item[10] for item in tempmat]
neck = [item[14] for item in tempmat]
torso = [item[0] for item in tempmat]
tair = [item[0] for item in tairmat]
plt.plot (luarm, label="luarm")
plt.plot(head, label="head")
plt.plot (torso, label="torso")
plt.plot(llarm, label="llarm")
plt.plot (lhand, label="lhand")
plt.plot(luleg, label="luleg")
plt.plot (llleg, label="llleg")
plt.plot(lfoot, label="lfoot")
plt.plot(neck,label="neck")
plt.plot(tair, label="Tair")
plt.xlabel('Iterations')
plt.ylabel('Temperature')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
