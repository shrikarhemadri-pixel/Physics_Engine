import matplotlib
from class_method import *
from testingmethods import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product
from IPython.display import HTML

part1 = Particle(0.1, [-1, 0], [0, 0])
part2 = Particle(0.1, [1, 0], [0, 0])

Simulate(1000, [part1, part2])

fig, ax = plt.subplots()
t = np.linspace(0, )