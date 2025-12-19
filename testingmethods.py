import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from class_method import *
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

p1 = Particle(10, [-1,  -1], [0,  0], -1e-7)
p2 = Particle(10, [ 1, 1], [0, 0], -1e-7)
p3 = Particle(10, [1, -1], [0, 0], -1e-7)
p4 = Particle(10, [-1, 1], [0, 0], -1e-7)
p5 = Particle(10, [0, 0], [0, 0], 1e-7)
p6 = Particle(10, [-0.5, -0.5], [0, 0], 1e-7)
p7 = Particle(10, [0.5, 0.5], [0, 0], 1e-7)
p8 = Particle(10, [0.5, -0.5], [0, 0], 1e-7)
p9 = Particle(10, [-0.5, 0.5], [0, 0], 1e-7)
p10 = Particle(10, [1, 0], [0, 0], -1e-7)
p11 = Particle(10, [0, 1], [0, 0], -1e-7)
p12 = Particle(10, [0, -1], [0, 0], -1e-7)
p13 = Particle(10, [-1, 0], [0, 0], -1e-7)
p14 = Particle(10, [-10, 0], [1, 0], 1)
p15 = Particle(10, [10, 0], [-1, 0])
p16 = Particle(10, [0, 10], [0, -1], 1)


Simulate(3000, [p1,p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15])

fig, ax  = plt.subplots()

ax.set_facecolor('black')
fig.set_facecolor('black')

t = np.linspace(0, dt*600, 600)


scat = ax.scatter(part_dictionary_x[0][0], part_dictionary_y[0][0], c='red', s=20, label='particle1')
scat2 = ax.scatter(part_dictionary_x[1][0], part_dictionary_y[1][0], c='red', s=20, label='particle2')
scat3 = ax.scatter(part_dictionary_x[2][0], part_dictionary_y[2][0], c='red', s=20, label='particle3')
scat4 = ax.scatter(part_dictionary_x[3][0], part_dictionary_y[3][0], c='red', s=20, label='particle4' )
scat5 = ax.scatter(part_dictionary_x[4][0], part_dictionary_y[4][0], c='blue', s=20, label='particle5' )
scat6 = ax.scatter(part_dictionary_x[5][0], part_dictionary_y[5][0], c='blue', s=20, label='particle5' )
scat7 = ax.scatter(part_dictionary_x[6][0], part_dictionary_y[6][0], c='blue', s=20, label='particle5' )
scat8 = ax.scatter(part_dictionary_x[7][0], part_dictionary_y[7][0], c='blue', s=20, label='particle5' )
scat9 = ax.scatter(part_dictionary_x[8][0], part_dictionary_y[8][0], c='blue', s=20, label='particle5' )
scat10 = ax.scatter(part_dictionary_x[9][0], part_dictionary_y[9][0], c='blue', s=20, label='particle5' )
scat11 = ax.scatter(part_dictionary_x[10][0], part_dictionary_y[10][0], c='blue', s=20, label='particle5' )
scat12 = ax.scatter(part_dictionary_x[11][0], part_dictionary_y[11][0], c='blue', s=20, label='particle5' )
scat13 = ax.scatter(part_dictionary_x[12][0], part_dictionary_y[12][0], c='blue', s=20, label='particle5' )
scat14 = ax.scatter(part_dictionary_x[13][0], part_dictionary_y[13][0], c='white', s=20, label='particle14')
scat15 = ax.scatter(part_dictionary_x[14][0], part_dictionary_y[14][0], c='white', s=20, label='particle14')

def update(frame):
    x = part_dictionary_x[0][frame]
    y = part_dictionary_y[0][frame]

    data = np.stack([x, y]).T
    scat.set_offsets(data)

    x2 = part_dictionary_x[1][frame]
    y2 = part_dictionary_y[1][frame]

    data2 = np.stack([x2, y2]).T
    scat2.set_offsets(data2)

    x3 = part_dictionary_x[2][frame]
    y3 = part_dictionary_y[2][frame]

    data3 = np.stack([x3, y3]).T
    scat3.set_offsets(data3)

    x4 = part_dictionary_x[3][frame]
    y4 = part_dictionary_y[3][frame]

    data4 = np.stack([x4, y4]).T
    scat4.set_offsets(data4)

    x5 = part_dictionary_x[4][frame]
    y5 = part_dictionary_y[4][frame]

    data5 = np.stack([x5, y5]).T
    scat5.set_offsets(data5)

    x6 = part_dictionary_x[5][frame]
    y6 = part_dictionary_y[5][frame]

    data6 = np.stack([x6, y6]).T
    scat6.set_offsets(data6)

    x7 = part_dictionary_x[6][frame]
    y7 = part_dictionary_y[6][frame]

    data7 = np.stack([x7, y7]).T
    scat7.set_offsets(data7)

    x8 = part_dictionary_x[7][frame]
    y8 = part_dictionary_y[7][frame]

    data8 = np.stack([x8, y8]).T
    scat8.set_offsets(data8)

    x9 = part_dictionary_x[8][frame]
    y9 = part_dictionary_y[8][frame]

    data9 = np.stack([x9, y9]).T
    scat9.set_offsets(data9)

    x10 = part_dictionary_x[9][frame]
    y10 = part_dictionary_y[9][frame]

    data10 = np.stack([x10, y10]).T
    scat10.set_offsets(data10)

    x11 = part_dictionary_x[10][frame]
    y11 = part_dictionary_y[10][frame]

    data11 = np.stack([x11, y11]).T
    scat11.set_offsets(data11)

    x12 = part_dictionary_x[11][frame]
    y12 = part_dictionary_y[11][frame]

    data12 = np.stack([x12, y12]).T
    scat12.set_offsets(data12)

    x13 = part_dictionary_x[12][frame]
    y13 = part_dictionary_y[12][frame]

    data13 = np.stack([x13, y13]).T
    scat13.set_offsets(data13)

    x14 = part_dictionary_x[13][frame]
    y14 = part_dictionary_y[13][frame]

    data14 = np.stack([x14, y14]).T
    scat14.set_offsets(data14)


    x15 = part_dictionary_x[14][frame]
    y15 = part_dictionary_y[14][frame]

    data15 = np.stack([x15, y15]).T
    scat15.set_offsets(data15)

    x16 = part_dictionary_x[15][frame]
    y16 = part_dictionary_y[15][frame]

    data16 = np.stack([x16, y16]).T
    scat16.set_offsets(data16)

    return scat, scat2, scat3, scat4, scat5, scat6, scat7, scat8, scat9, scat10, scat11, scat12, scat13, scat14, scat15


ani = animation.FuncAnimation(fig=fig, func=update, frames=3000, interval=30)
plt.show()
