from class_method import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

p1 = Particle(1, [-1, 0], [0.5, 0])
p2 = Particle(1, [1, 0], [-0.5, 0])

Simulate(300, [p1, p2])


fig, ax = plt.subplots()

scat1 = ax.scatter(part_dictionary_x[0][0], part_dictionary_y[0][0], c='b', s=20, label='particle1')
scat2 = ax.scatter(part_dictionary_x[1][0], part_dictionary_y[1][0], c='b', s=20, label='particle2')

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

ani = animation.FuncAnimation(fig=fig, func=update, frames=300, interval=30, blit=False)

plt.show()
