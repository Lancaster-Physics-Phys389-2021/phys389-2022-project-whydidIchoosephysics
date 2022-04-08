#!/usr/bin/env python
# coding: utf-8

# In[1]:


from cmath import cos, pi, sin
from tkinter import OptionMenu
import numpy as np
import math
import copy
import matplotlib.pyplot as plt

"""This class aims to set methods to calculate the kinetics using Euler's Method"""
class Particle:

    G = 6.67408E-11

    def __init__(
    self,
    position=np.array([0, 0, 0], dtype=float),
    velocity=np.array([0, 0, 0], dtype=float),
    acceleration=np.array([0, -10, 0], dtype=float),
    name='Ball',
    mass=1.0):

        self.position = np.array(position, dtype= float)
        self.velocity = np.array(velocity, dtype= float)
        self.acceleration = np.array(acceleration, dtype= float)
        self.name = name
        self.mass = mass
    
    

    def __str__(self):
        return "Particle: {0}, Mass: {1:.3e}, Position: {2}, Velocity: {3}, Acceleration: {4}".format(
        self.name, self.mass,self.position, self.velocity, self.acceleration )

    """A function that updates, velocity, and position knowing the acceleration"""
    def update(self, deltaT):
        initial_velocity = self.velocity
        initial_position = self.position
        self.acceleration = self.acceleration
        self.position = initial_position + (deltaT * self.velocity)
        self.velocity = initial_velocity + (deltaT * self.acceleration)

    """A function that calculates the gravitational acceleration"""
    def n_bodies_acceleration (self, bodies):
        self.acceleration = np.array([0, 0, 0], dtype=float)
        for nbody in bodies:
            r = self.position-nbody.position
            r_d =np.linalg.norm(r)
            r_unit = r/r_d
            acceleration = -self.G*nbody.mass*r_unit/((r_d)**2)
            self.acceleration += acceleration
        return self.acceleration

    """A function that calculates the gravitational force """
    def force(self,bodies):
        force = np.array([0, 0, 0], dtype=float)
        for nbody in bodies:
            r = self.position -nbody.position
            r_d =np.linalg.norm(r)
            r_unit = r/r_d
            force += (-self.G*nbody.mass*self.mass*r_unit/((r_d)**2))
        return force

    """A function that calculates Potential energy  """
    def Potential_energy (self,bodies):
        Potential_E = 0
        for nbody in bodies:
            r = self.position - nbody.position
            r_d = np.linalg.norm(r)
            potential_es = -self.G*self.mass*nbody.mass/r_d
            Potential_E += potential_es
        return Potential_E

    """A function that calculates Kinetic Energy  """
    def kineticEnergy(self):
        v_v = (np.linalg.norm(self.velocity))**2
        Kinetic = 0.5*self.mass*v_v
        return Kinetic

    """A function that calculates Angular Momentum  """
    def angular_momentum(self):
        angular_momentum = np.array([0, 0, 0], dtype=float)
        momentum = self.velocity*self.mass
        angular_momentum = np.cross(self.position,momentum)
        return angular_momentum




# In[2]:


class Parcticle_in_fields(Particle):
    def __init__(
    self,
    position=np.array([0, 0, 0], dtype=float),
    velocity=np.array([0, 0, 0], dtype=float),
    acceleration=np.array([0, 0, 0], dtype=float),
    name='Ball',
    mass=1.0,
    c = 3*10**8,
    charge = 1.602*10**(-19)):
        Particle.__init__(self,position,velocity,acceleration,name,mass)
        self.charge= charge
        self.c=c

    def magneticField(self):
        if np.linalg.norm(self.position) >= self.cyclotronRadius:
            self.magneticField = np.array([0, 0, 0])
        return self.magneticField
    
    def update_Cromer(self, deltaT):
        initial_velocity = self.velocity
        initial_position = self.position
        self.acceleration = self.acceleration
        self.velocity = initial_velocity + (deltaT * self.acceleration)
        self.position = initial_position + (deltaT * self.velocity)

    def update_Verlet(self, deltaT):
        initial_velocity = self.velocity
        initial_position = self.position
        initial_acceleration = self.acceleration
       
        force=self.charge*np.cross(self.velocity,self.magneticField)
        new_acceleration =force/self.mass
        
        self.position = initial_position + (initial_velocity*deltaT) +(initial_acceleration*deltaT**2)
        self.velocity = initial_velocity + (deltaT/2) * (initial_acceleration+new_acceleration) 
    

    def acceleration_due_to_magnetic_field(self):

        self.magneticField=np.array ([0, 0, 20])

        force=self.charge*np.cross(self.velocity,self.magneticField)
        accEM =force/self.mass
        self.acceleration = accEM
        return self.acceleration


# In[3]:


particle_constant_magnetic=Parcticle_in_fields(mass=9.109e-31,velocity=[80,0,0])


# In[4]:


time=0
Times=[]
deltaT=1e-15
Data=[]
for j in range(0, 3000):
    for i in range(0, 10):
        Times.append(time)
        time += deltaT
        particle_constant_magnetic.acceleration_due_to_magnetic_field()
#         particle_constant_magnetic.update_Cromer(deltaT)#
        particle_constant_magnetic.update(deltaT)
#         particle_constant_magnetic.update_Verlet(deltaT)
        if i == 0:
            period = [time, copy.deepcopy(particle_constant_magnetic)]
            Data.append(period)


# In[5]:


x = []
y = []
z = []
norm=[]
time=[]
velocity=[]
energy=[]
for i in Data:
        x.append(i[1].position[0])
        y.append(i[1].position[1])
        z.append(i[1].position[2])
        norm.append(np.linalg.norm(i[1].position))
        time.append(i[0])
        v=i[1].velocity
        velocity.append(np.linalg.norm(v))

    


# In[6]:


plt.plot(time, norm)
plt.xlabel('$Time (s)$', fontsize = 15)
plt.ylabel('Norm of the trajectory (m)', fontsize =15)
plt.show()


# In[7]:


plt.plot(time, velocity)
plt.xlabel('$Time (s)$', fontsize = 15)
plt.ylabel('speed of the particle (m/s)', fontsize =15)
plt.show()


# In[8]:


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z, label="Proton trajectory")
ax.legend()
ax.set_xlabel('$x$', fontsize=15)
ax.set_ylabel('$y$', fontsize=15)
ax.set_zlabel("$z$", fontsize=15)
plt.show()


# In[9]:


plt.plot(x, y)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize =15)
plt.show()


# In[ ]:




