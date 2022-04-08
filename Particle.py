import numpy as np
import math as m
import pandas as pd
import plotly as plt



class Particle:
    """
        Class to define particles and how
        the electrostatic forces between
        them makes them move in space.

        Parameters
        ----------
        Name
        Mass
        Initial values for the planet
    """

    def __init__(
            self,
            position = np.array([0,0,0], dtype=float),
            velocity = np.array([0, 0, 0], dtype=float),
            acceleration = np.array([0, 0, 0], dtype=float),
            name = 'Electron',
            mass = 1.0,
            charge = -1,
            ):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.array(acceleration, dtype=float)
        self.name = name
        self.mass = mass
        self.charge = charge

    def __str__(self):
        return "Particle: {0}, Mass: {1:.3e}, Charge: {2}, Position: {3}, Velocity: {4}, Acceleration: {5}".format(
            self.name,
            self.mass,
            self.charge,
            self.position,
            self.velocity,
            self.acceleration
        )

    def update(self, deltaT):
        """
        Uses the Euler method to approximate the position, velocity & acceleration
        """

        self.deltaT = deltaT

        self.position = self.position + self.velocity * self.deltaT
        self.velocity = self.velocity + self.acceleration * self.deltaT

    def updateCoulombAcceleration(self, body):
        """
        Updates the acceleration between two bodies
        based on Coulomb's Law
        """

        K = 8.988 * m.pow(10,9)

        self.body = body
        self.acceleration = K * self.charge * body.charge * ((self.position[0]-body.position[0])**2 + (self.position[1]-body.position[1])**2 + (self.position[2]-body.position[2])**2)**(-3/2) * (self.position - body.position) / self.mass


    def kineticEnergy(self):
        """
        Calculates the Kinetic Energy of the Planet
        """

        v_squared = (np.linalg.norm(self.velocity))**2
        KEnergy = 0.5*self.mass*v_squared
        return KEnergy


