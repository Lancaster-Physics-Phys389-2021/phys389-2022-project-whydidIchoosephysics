#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import copy
import matplotlib.pyplot as plt
import numpy as np

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
    k=8.98755e9*100,
    charge = 1.602*10**(-19)):
        Particle.__init__(self,position,velocity,acceleration,name,mass)
        self.charge= charge
        self.c=c
        self.k=k

    def magneticField(self):
        if np.linalg.norm(self.position) >= self.cyclotronRadius:
            self.magneticField = np.array([0, 0, 0])
        return self.magneticField

    def acceleration_due_to_couloumb_force(self,bodies):
        self.acceleration = np.array([0, 0, 0], dtype=float)
        for nbody in bodies:
            r = self.position-nbody.position
            r_d =np.linalg.norm(r)
            r_unit = r/r_d
            force = self.k*self.charge*nbody.charge*r_unit/((r_d)**2)
            acceleration=force/self.mass
            
            self.acceleration += acceleration
        return self.acceleration


# In[3]:


particle_A=Parcticle_in_fields(mass=1.673e-27,position=[0.000005,0,0])
particle_B=Parcticle_in_fields(mass=1.673e-27,position=[0.00000,0,0])
bodies_A=[particle_A]
bodies_B=[particle_B]


# In[5]:


time=0
Times=[]
deltaT=1e-15
Data=[]
for j in range(0, 30000):
    for i in range(0, 100):
        Times.append(time)
        time += deltaT
        particle_A.acceleration_due_to_couloumb_force(bodies_B)
        particle_B.acceleration_due_to_couloumb_force(bodies_A)
        particle_A.update(deltaT)
        particle_B.update(deltaT)
        if i == 0:
            period = [time, copy.deepcopy(particle_A),copy.deepcopy(particle_B)]
            Data.append(period)


# In[ ]:


x = []
y = []
z = []
x_B=[]
y_B = []
z_B = []
for i in Data:
        x.append(i[1].position[0])
        y.append(i[1].position[1])
        z.append(i[1].position[2])
        x_B.append(i[2].position[0])
        y_B.append(i[2].position[1])
        z_B.append(i[2].position[2])
    


# In[ ]:


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z, label="Proton A trajectory")
ax.plot(x_B, y_B, z_B, label="Proton B trajectory")
ax.plot(x[0], y[0], z[0], marker="o",markersize=5)
ax.plot(y[0], y[0], z[0], marker="o",markersize=5)
ax.legend()
ax.set_xlabel('$X(m)$', fontsize=15)
ax.set_ylabel('$Y(m)$', fontsize=15)
ax.set_zlabel("$Z (m)$", fontsize=15)
plt.show()


# In[ ]:


plt.figure()
plt.plot(x, y)
plt.plot(x_B, y_B)
plt.plot([0.000005],[0], marker="o",markersize=5,label="Initial position particle A")
plt.plot([0],[0], marker="o",markersize=5,label="Initial position particle B")
plt.xlabel('$X(m)$', fontsize = 15)
plt.ylabel('$Y (m)$', fontsize =15)
plt.legend()
plt.show()


# In[ ]:


print(x)


# In[ ]:




