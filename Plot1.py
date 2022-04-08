from Proton_Positron_Simulation1 import *

import matplotlib.pyplot as plt

ax = plt.axes(projection='3d')
# Data for a three-dimensional line
xline = np.load("Proton_x.npy")
zline = np.load("Proton_y.npy")
yline = np.load("Proton_z.npy")

xlineS = np.load("AntiProton_x.npy")
zlineS = np.load("AntiProton_y.npy")
ylineS = np.load("AntiProton_z.npy")

ax.plot3D(xline, yline, zline, label = 'Proton', color = 'green', marker = 'o')
ax.plot3D(xlineS, ylineS, zlineS, label = 'Antiproton', color = 'red')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.legend(["Proton", "Antiproton"])

# giving a title to my graph
plt.title('Euler - 13.8 Days')

plt.show()
