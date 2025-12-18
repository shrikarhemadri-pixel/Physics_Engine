from class_method import *
import numpy as np
import vispy 


from vispy import app
import sys
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Polygon, Ellipse, Rectangle, RegularPolygon
from vispy.color import Color
from vispy import gloo
import time


#Color Variables
white = Color("#ecf0f1")
gray = Color("#756666")
red = Color("#e74c3c")
blue = Color("#2980b9")
orange = Color("#e88834")
black = Color("#090202")


#Canvas initialization
canvas = SceneCanvas(keys='interactive', title='Polygon Example',
                     show=True)
v = canvas.central_widget.add_view()
v.bgcolor = black
v.camera = 'panzoom'

#Particle Initialization- parameters(mass, position, velocity, charge, volume)
part1 = Particle(0.1, [0.1, 0.1], [0, 0], -1e-1, 0, 0.01)
part2 = Particle(0.1, [-0.1, -0.1], [0,0], 1e-1, 0, 0.1 )
Simulate(700, [part1, part2])


#Setting the camera zoom to the minimum values

all_pos = np.vstack()
cx, cy = (part1.position[0], part1.position[1])
halfx, halfy = (0.1, 0.1)

poly_coords = (cx, cy)


ellipse = Ellipse(center=(part1.position[0], part1.position[1]), radius=(part1.radius, part1.radius),
                  color=blue, border_width=2, border_color=black,
                  num_segments=1,
                  parent=v.scene)

ellipse.num_segments = 100
ellipse.start_angle = 0
ellipse.span_angle = 360


if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()