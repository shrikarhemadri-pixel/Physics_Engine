import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from class_method import *

"""
Start of Main Script/Code
"""

"""
Attempting Eliptical Orbit with the earth, sun, and moon system
"""

earth_sun_distance = 1.495978707e11
earth_moon_distance =  3.844e8
sun = Particle(sun_mass, [0,0])
earth = Particle(earth_mass, [earth_sun_distance, 0], [0, 2.98e4])
moon = Particle(moon_mass, [(earth_sun_distance+earth_moon_distance), 0], [0, 2.98e4 + 1.022e3 ])

#Attempting to make a system that contains three objects
#Testing Secnairos where Gravity does have a massive between two moons

"""
Position Points for the earth, sun, and moon
"""


total_system = [earth, sun, moon]

#X values
earth_x = [earth.position[0]]
sun_x = [sun.position[0]]
moon_x = [moon.position[0]]


#Y values
earth_y = [earth.position[1]]
sun_y = [sun.position[1]]
moon_y = [moon.position[1]]


#Distances to cross check error
distance_mag, distance_vec = Distance(earth, moon)
difference = [distance_mag]




#Getting time values
dt_list = [0]

#Values When Gravity is affecting the attribuites
for i in range(8640*22):

    for k in range(len(total_system)):
        update_attributes(total_system[k], netAcceleration(total_system, k))

    #Update moons  position
    moon_x.append(moon.position[0])
    moon_y.append(moon.position[1])

    #Update planet 1's position
    earth_x.append(earth.position[0])
    earth_y.append(earth.position[1])

    #Update planet 2's position
    sun_x.append(sun.position[0])
    sun_y.append(sun.position[1])

    #Time plots
    dt_list.append(dt*(i+1))

    #Distance Updates
    distance_mag, distance_vec = Distance(earth, moon)
    difference.append(distance_mag)



"""
plt.title('Earth, Sun, Moon System in the x direction')
plt.xlabel('Time (sec)')
plt.ylabel('X (m)')
plt.xlim(min(dt_list), max(dt_list))
plt.ylim(min(moon_x), max(moon_y))
plt.quiver(dt_list[:-1], earth_x[:-1], np.diff(dt_list), np.diff(earth_x), color='r', angles='xy', scale_units='xy', scale=1, label='Earth X motion')
plt.quiver(dt_list[:-1], moon_x[:-1], np.diff(dt_list), np.diff(moon_x), color='b', angles='xy', scale_units='xy', scale=1, label='Moon X motion')
plt.legend()
plt.show()



"""
plt.title('Earth, Sun, Moon System')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.xlim(min(earth_x), max(earth_x))
plt.ylim(min(earth_y), max(earth_y))

plt.quiver(earth_x[:-1], earth_y[:-1], np.diff(earth_x), np.diff(earth_y), color='r', angles='xy', scale_units='xy', scale=1, label='Earth')
plt.quiver(moon_x[:-1], moon_y[:-1], np.diff(moon_x), np.diff(moon_y),color='b', angles='xy', scale_units='xy', scale=1,label='Moon')
plt.plot(sun_x, sun_y, markersize=10)
plt.legend()

plt.show()


#Error

#Twin moons operating a planet
    #Accounting for Gravity?
#- Wednesday Next meeting


#Non-point like objects
    #Should a particle be point like?
    #How to plot a non point like object (object that has density)
    #should the particle have a density distribution



#Rotating Stars


