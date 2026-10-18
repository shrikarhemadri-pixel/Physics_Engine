import numpy as np
import constants as C

"""
Atherosclerosis modeling roadmap

Step 1: simulate laminar flow through a pipe          <- driven periodic tube below
Step 2: introduce obstructions that resist the flow   <- STENOSIS_SEVERITY (lumen narrowing)
Step 3: experiment with "pipe" morphology             <- edit wall_radius()
Step 4: fine-tune parameters
"""

# Pressure-gradient-equivalent body acceleration along +x. In a periodic tube
# this drives the flow the way a pressure drop would (it is not a position hack:
# it enters through the momentum equation, so viscosity and walls resist it).
FLOW_ACCELERATION = np.zeros(C.DIM)
if C.SCENE == "pipe":
    FLOW_ACCELERATION[0] = 0.2


def driving_acceleration():
    return FLOW_ACCELERATION


def wall_radius(x):
    """Local lumen radius R(x). Straight pipe unless STENOSIS_SEVERITY > 0."""
    bump = np.exp(-0.5 * ((np.asarray(x) - C.STENOSIS_CENTER) / C.STENOSIS_WIDTH) ** 2)
    return C.PIPE_RADIUS * (1.0 - C.STENOSIS_SEVERITY * bump)


def wall_radius_slope(x):
    """dR/dx, needed so the wall normal tilts correctly on a narrowing."""
    bump = np.exp(-0.5 * ((np.asarray(x) - C.STENOSIS_CENTER) / C.STENOSIS_WIDTH) ** 2)
    return C.PIPE_RADIUS * C.STENOSIS_SEVERITY * bump * (np.asarray(x) - C.STENOSIS_CENTER) / C.STENOSIS_WIDTH ** 2


def wrap_periodic(positions):
    """Re-insert particles that left through +x/-x on the opposite side."""
    if C.PERIODIC_X:
        length = 2 * C.XLIMITS
        positions[:, 0] = (positions[:, 0] + C.XLIMITS) % length - C.XLIMITS


def radial_velocity_profile(positions, velocities, nbins=8):
    """Mean axial velocity vs. distance from the axis (pipe scene).
    Laminar flow should give a parabola: fastest at the centre, ~0 at the wall."""
    rho = np.linalg.norm(positions[:, 1:], axis=1)
    edges = np.linspace(0, C.PIPE_RADIUS, nbins + 1)
    which = np.clip(np.digitize(rho, edges) - 1, 0, nbins - 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    prof = np.array([velocities[which == b, 0].mean() if np.any(which == b) else np.nan for b in range(nbins)])
    return centres, prof