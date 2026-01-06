from simulator_particle_core import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


#Attempting to simulate 2 colliding particles moving towards each other
p1 = Particle(1, [-1, 0], [0.5, 0])
p2 = Particle(1, [1, 0], [-0.5, 0])

#Simulate method found in the class_method files 
Simulate(4000, [p1, p2])

# Basic Matplotlib Animations to help with visualization
fig, ax = plt.subplots()
fig.set_facecolor('black')
ax.set_facecolor('black')
scat1 = ax.scatter(part_dictionary_x[0][0], part_dictionary_y[0][0], c='b', s=20, label='particle1')
scat2 = ax.scatter(part_dictionary_x[1][0], part_dictionary_y[1][0], c='white', s=20, label='particle2')


"""Frame increment function- currently working on putting it in the "simulator_particle_core" file to 
make the interface when simulating functions even more simple"""


def update(frame):
    x = part_dictionary_x[0][frame]
    y = part_dictionary_y[0][frame]

    data = np.stack([x, y]).T
    scat1.set_offsets(data)

    x2 = part_dictionary_x[1][frame]
    y2= part_dictionary_y[1][frame]

    data2 = np.stack([x2, y2]).T
    scat2.set_offsets(data2)

    return scat1, scat2


ani = animation.FuncAnimation(fig=fig, func=update, frames=4000, interval=1, blit=False)

plt.show()
