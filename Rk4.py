from simulator_particle_core import * 
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

# p1 = Particle(1,[-0.9700436, 0.24308753], [0.466203685, 0.43236753])
# p2 = Particle(1, [0, 0], [-0.466203685*2, -0.43236753*2])
# p3 = Particle(1, [0.9700436, -0.24308753], [0.466203685, 0.43236753])

R = 1/np.sqrt(3)

p1 = Particle(
    1,
    np.array([ 0.5773502692,  0.0 ]),
    np.array([ 0.0,  0.7598356857 ])
)

p2 = Particle(
    1,
    np.array([ -0.2886751346,  0.5 ]),
    np.array([ -0.6580370065, -0.3799178429 ])
)

p3 = Particle(
    1,
    np.array([ -0.2886751346, -0.5 ]),
    np.array([  0.6580370065, -0.3799178429 ])
)


Simulate(10000, [p1, p2, p3])
# fig, ax = plt.subplots()
# ax.set_facecolor('black')
# fig.set_facecolor('black')




# scat1 = ax.scatter(part_dictionary_x[0][0], part_dictionary_y[0][0], c='blue', s=20, label='particle1')
# scat2 = ax.scatter(part_dictionary_x[1][0], part_dictionary_y[1][0], c='white', s=20, label='particle2')
# scat3 = ax.scatter(part_dictionary_x[2][0], part_dictionary_y[2][0], c='red', s=20, label='particle3')


# min_x = min(min(values) for values in part_dictionary_x.values())
# max_x = max(max(values) for values in part_dictionary_x.values())

# min_y = min(min(values) for values in part_dictionary_y.values())
# max_y = max(max(values) for values in part_dictionary_x.values())



# ax.set_xlim(min_x-0.2, max_x + 0.2)
# ax.set_ylim(min_y - 0.2, max_y + 0.2)

# line1 = []
# line2 = []
# line3 = []



# trail1 = []
# trail2 = []
# trail3 = []
# trail_length = 100

# def update(frame):
#     # Particle 1
#     x = part_dictionary_x[0][frame]
#     y = part_dictionary_y[0][frame]
#     trail1.append([x, y])
#     if len(trail1) > trail_length:
#         trail1.pop(0)
#     data = np.array(trail1)
#     alphas = np.linspace(0.1, 1, len(data))[:, None]  # fading
#     scat1.set_offsets(data)
#     scat1.set_alpha(alphas)

#     # Particle 2
#     x2 = part_dictionary_x[1][frame]
#     y2 = part_dictionary_y[1][frame]
#     trail2.append([x2, y2])
#     if len(trail2) > trail_length:
#         trail2.pop(0)
#     data2 = np.array(trail2)
#     alphas2 = np.linspace(0.1, 1, len(data2))[:, None]
#     scat2.set_offsets(data2)
#     scat2.set_alpha(alphas2)

#     # Particle 3
#     x3 = part_dictionary_x[2][frame]
#     y3 = part_dictionary_y[2][frame]
#     trail3.append([x3, y3])
#     if len(trail3) > trail_length:
#         trail3.pop(0)
#     data3 = np.array(trail3)
#     alphas3 = np.linspace(0.1, 1, len(data3))[:, None]
#     scat3.set_offsets(data3)
#     scat3.set_alpha(alphas3)

#     return scat1, scat2, scat3

# ani = animation.FuncAnimation(fig=fig, func=update, frames=10000, interval=1, blit=False)
# plt.show()


plt.plot(part_dictionary_x[0], part_dictionary_y[0])
plt.plot(part_dictionary_x[1], part_dictionary_y[1])
plt.plot(part_dictionary_x[2], part_dictionary_y[2])

plt.show()
