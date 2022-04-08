from Particle import *
from Particles_Properties import Proton, AntiProton

from pathlib import Path
import math as m
import pandas as pd
import numpy as np
import copy


"""
Will start a proton and antiproton at a distance on the x axis,
The proton will start stationary at [0,0,0] while the antiproton
will have a velocity in the y direction
"""

Proton_Xposition = []        #List that will store all the x-coords for the Proton
Proton_Yposition = []        #
Proton_Zposition = []

AntiProton_Xposition = []        #List that will store all the x-coords for the Antiproton
AntiProton_Yposition = []        #
AntiProton_Zposition = []


Amount = 1000     #How long the simulation will run for (in seconds)
Ttime = 1                 #How many seconds will the algorithms use for the approximations as time-stamps
Data = []                   #List that will store all the information after every update


for i in range(Amount):
    Proton.updateCoulombAcceleration(AntiProton)
    AntiProton.updateCoulombAcceleration(Proton)

    Proton.update(Ttime)
    AntiProton.update(Ttime)

    if i % 1 == 0:
        time = Ttime * (i + 1)
        Data.append([time, copy.deepcopy(Proton), copy.deepcopy(AntiProton)])

        Proton_Xposition.append(Proton.position[0])
        Proton_Yposition.append(Proton.position[1])
        Proton_Zposition.append(Proton.position[2])

        AntiProton_Xposition.append(AntiProton.position[0])
        AntiProton_Yposition.append(AntiProton.position[1])
        AntiProton_Zposition.append(AntiProton.position[2])

np.save("Proton_x", Proton_Xposition, allow_pickle=True)
np.save("Proton_y", Proton_Yposition, allow_pickle=True)
np.save("Proton_z", Proton_Zposition, allow_pickle=True)

np.save("AntiProton_x", AntiProton_Xposition, allow_pickle=True)
np.save("AntiProton_y", AntiProton_Yposition, allow_pickle=True)
np.save("AntiProton_z", AntiProton_Zposition, allow_pickle=True)


np.save("TwoBodyTest", Data, allow_pickle=True)

