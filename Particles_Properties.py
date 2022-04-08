from Particle import Particle
from pathlib import Path
import math as m
import pandas as pd
import numpy as np
import copy


protonMass = 1.673 * m.pow(10,-27)
protonCharge = 1.6 * m.pow(10,-19)

electronMass = 9.109 * m.pow(10,-31)
electronCharge = 1.602 * m.pow(10,-19)

Proton = Particle(
    position=np.array([0, 0, 0]),
    velocity=np.array([0, 0, 0]),
    acceleration=np.array([0, 0, 0]),
    name="Proton",
    mass=protonMass,
    charge=+1
)

AntiProton = Particle(
    position=np.array([1*m.pow(10,-9), 0, 0]),
    velocity=np.array([0, 1, 0]),
    acceleration=np.array([0, 0, 0]),
    name="AntiProton",
    mass=protonMass,
    charge=-1
)

Electron = Particle(
    position=[0, 0, 0],
    velocity=[0, 0, 0],
    acceleration=np.array([0, 0, 0]),
    name="Electron",
    mass=electronMass,
    charge=-1
)

Positron = Particle(
    position=[0, 0, 0],
    velocity=[0, 0, 0],
    acceleration=np.array([0, 0, 0]),
    name="Positron",
    mass=electronMass,
    charge=+1
)



