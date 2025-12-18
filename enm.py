import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from class_method import *

p1 = Particle(1, [1e4, 0], [0, 0], 10)
p2 = Particle(1, [-1e4, 0], [0, 0], 10)

total_system = [p1, p2]

p1_x = [p1.position[0]]
p1_y = [p1.position[1]]

p2_x = [p2.position[0]]
p2_y = [p2.position[1]]

dt_list = [0]

distance_mag, distance_vec = Distance(p1, p2)
difference = [distance_mag]

for i in range(100):

    for k in range(len(total_system)):
        update_attributes(total_system[k], netAcceleration(total_system, k))

    #Update moons  position
    p1_x.append(p1.position[0])
    p1_y.append(p1.position[1])

    #Update planet 1's position
    p2_x.append(p2.position[0])
    p2_y.append(p2.position[1])

    #Time plots
    dt_list.append(dt*(i+1))

    #Distance Updates
    distance_mag, distance_vec = Distance(p1, p2)
    difference.append(distance_mag)




plt.quiver(p1_x[:-1], p1_y[:-1], np.diff(p1_x), np.diff(p1_y), color='r',  angles='xy', scale_units='xy', scale=1, label='p1')
plt.quiver(p2_x[:-1], p2_y[:-1], np.diff(p2_x), np.diff(p2_y), color='b',  angles='xy', scale_units='xy', scale=1, label='p2')
plt.xlim(min(p2_x), max(p1_x))
plt.legend()
plt.show()
