from class_method import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


fig, ax = plt.subplots()

ax.set_facecolor('black')
fig.set_facecolor('black')
earth = Particle(earth_mass, [0, 0], [0, 0])
moon = Particle(moon_mass, [3.84e8, 0], [0, 1.022e3])



Simulate(100000, [earth, moon])




scat = ax.scatter(part_dictionary_x[0][0], part_dictionary_y[0][0], c='b', s=20, label='earth')
scat2 = ax.scatter(part_dictionary_x[1][0], part_dictionary_y[1][0], c='gray', s=20, label='moon')
ax.set_xlim(-5e8, 5e8)
ax.set_ylim(-5e8, 5e8)


def update(frame):
    x = part_dictionary_x[0][frame]
    y = part_dictionary_y[0][frame]

    data = np.stack([x, y]).T
    scat.set_offsets(data)

    x2 = part_dictionary_x[1][frame]
    y2 = part_dictionary_y[1][frame]

    data2 = np.stack([x2, y2]).T
    scat2.set_offsets(data2)


    return scat, scat2


ani = animation.FuncAnimation(
    fig=fig,
    func=update,
    frames=100000,
    interval=30,
    blit=False
)
plt.show(block=True)


