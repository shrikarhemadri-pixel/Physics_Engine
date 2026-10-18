import fluidsim
import numpy as np
import constants
"""
Start of Atherosclerosis Modeling

Step 1: simulate laminar flow through a pipe

Step 2: Introduce obstructions that resist the flow (models the atherosclerosis plaque)

Step 3: Experiement with "pipe" morphology 

Step 4: Fine tune parameters
"""

FLOW_VELOCITY = np.array([0., 0.])


#Step 1--> Laminar Flow

#Getting Particles to move in a certain direction

def update_fluid_flow():

    fluidsim.positions += FLOW_VELOCITY * constants.DT

def loop_fluid(index):
    if fluidsim.positions[index][0] == constants.XLIMITS:
        fluidsim.positions[index][0] = constants.XLIMITS * -1
    if fluidsim.positions[index][1] == constants.YLIMITS:
        fluidsim.positions[index][1] = constants.YLIMITS * -1