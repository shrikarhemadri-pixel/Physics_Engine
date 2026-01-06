from simulator_particle_core import * 
import matplotlib.pyplot as plt
from simulator_particle_core import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter



#Testcase scenario 
# 1) two body motion G is set to 1 
# 2) M is set to 1 for both
# 3) at exactly t = 2pi the position must be equal to the initial one

p1 = Particle(1, [0, 0], [0, 0.5])
p2 = Particle(1, [1, 0], [0, -0.5])

Simulate(10000, [p1, p2])

fig, ax = plt.subplots()


ax.set_facecolor('black')
fig.set_facecolor('black')


min_x = min(min(values) for values in part_dictionary_x.values())
max_x = max(max(values) for values in part_dictionary_x.values())

min_y = min(min(values) for values in part_dictionary_y.values())
max_y = max(max(values) for values in part_dictionary_x.values())



ax.set_xlim(min_x-0.2, max_x + 0.2)
ax.set_ylim(min_y - 0.2, max_y + 0.2)



scat1 = ax.scatter(part_dictionary_x[0][0], part_dictionary_y[0][0], c='blue', s=20, label='particle1')
scat2 = ax.scatter(part_dictionary_x[1][0], part_dictionary_y[1][0], c='red', s=20, label='particle2')

def update(frame):
    x = part_dictionary_x[0][frame]
    y = part_dictionary_y[0][frame]

    data = np.stack([x, y]).T
    scat1.set_offsets(data)

    x2 = part_dictionary_x[1][frame]
    y2 = part_dictionary_y[1][frame]

    data2 = np.stack([x2, y2]).T
    scat2.set_offsets(data2)



ani = animation.FuncAnimation(fig=fig, func=update, frames=10000, interval=1)
plt.show()





