import numpy as np
import PID as PID
import plate as model
###############################################################################
def makeData():
    qv = 0  # Initial qgen
    tinfv = 0  # Initial Too
    L = 0.035

    G = model.SteppingClass(25, L, 0.613, 0.146e-6)
    dt = 60
    N = 500#180

    coreTemp_list = np.zeros(N)  # Preallocate space
    surfTemp_list = np.zeros(N)  # Preallocate space

    Tinfs = np.ones(N) * tinfv  # Constant ambient temperature
    qsets = np.zeros(N)  # Preallocate space

    for i in range(N):
        amp = 5
        if i >= 75:
            Tinfs[i] = amp * np.sin((i - 75) / (N - 75) * 4 * np.pi) + tinfv + amp  # simple sine wave for testing
        #amp = 12
        #if i >= 75:
        #    j = (i - 75) / (N - 75)
        #    if i <= 150:
        #        Tinfs[i] = amp * np.exp(-2 * j) * np.sin(2 * 20 * np.pi * j ** 2) + tinfv + (
        #                                                                                    i - 75) / 10  # Best training data so far
        #    else:
        #        Tinfs[i] = Tinfs[i - 1] - 0.05

        coreTemp_list[i] = G.greensStep(0, dt, Tinfs[:i], qsets[:i])  # Get temp at core
        surfTemp_list[i] = G.greensStep(L, dt, Tinfs[:i], qsets[:i])  # Get temp at skin

        pid = PID.PID(P=35000, I=1, D=10)
    '''#These values can be adjusted, but if they are, 
    the values hardcoded in later on to get temperature 
    to 37* need to be adjusted'''
    pid.SetPoint = 37
    pid.setSampleTime(dt)
    #            if i > 25:
    qset = pid.update(coreTemp_list[i])
    #            else:
    #                qset = 4000000 #pid.update(coreTemp_list[i])
    qsets[i] = qset


    return qsets, Tinfs, coreTemp_list, surfTemp_list


def makeNewData():
    qv = 0  # Initial qgen
    tinfv = 0  # Initial Too
    L = 0.035

    G = model.SteppingClass(25, L, 0.613, 0.146e-6)
    dt = 60
    N = 500#180

    coreTemp_list = np.zeros(N)  # Preallocate space
    surfTemp_list = np.zeros(N)  # Preallocate space

    Tinfs = np.ones(N) * tinfv  # Constant ambient temperature
    qsets = np.zeros(N)  # Preallocate space

    for i in range(N):
        amp = 12
        if i >= 75:
            j = (i - 75) / (N - 75)
            if i <= 150:
                Tinfs[i] = amp * np.exp(-2 * j) * np.sin(2 * 20 * np.pi * j ** 2) + tinfv + (
                                                                                                i - 75) / 10  # Best training data so far
            else:
                Tinfs[i] = Tinfs[i - 1] - 0.05
        #amp = 5
        #if i >= 75:
        #    Tinfs[i] = amp * np.sin((i - 75) / (N - 75) * 4 * np.pi) + tinfv + amp  # simple sine wave for testing
        coreTemp_list[i] = G.greensStep(0, dt, Tinfs[:i], qsets[:i])
        surfTemp_list[i] = G.greensStep(L, dt, Tinfs[:i], qsets[:i])

        pid = PID.PID(P=35000, I=1, D=10)
        pid.SetPoint = 37
        pid.setSampleTime(dt)
        #            if i > 25:
        qset = pid.update(coreTemp_list[i])
        #            else:
        #                qset = 4000000 #pid.update(coreTemp_list[i])
        qsets[i] = qset

    return qsets, Tinfs, coreTemp_list, surfTemp_list