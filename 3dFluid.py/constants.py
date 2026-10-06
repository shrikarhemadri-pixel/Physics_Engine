import numpy as np

# ------------------------------------------------------------------
# Dimensionality: 2 or 3. Every module reads this, so the same code
# runs the 2D solver (for checking against your old results) and 3D.
# ------------------------------------------------------------------
DIM = 3

# ---- window / drawing ----
WIDTH, HEIGHT = 1100, 650
RADIUS = 1                      # draw radius scale

# ---- domain (half-extents, simulation units). Flow runs along +x ----
XLIMITS = 60.0
YLIMITS = 36.0
ZLIMITS = 36.0
HALF_EXTENTS = np.array([XLIMITS, YLIMITS, ZLIMITS])[:DIM]

SCENE = "box"                  # "pipe" (driven periodic tube) or "box" (gravity test)
PIPE_RADIUS = 36.0              # lumen radius (half-width in 2D)
PERIODIC_X = (SCENE == "pipe")  # particles leaving +x re-enter at -x

# Gaussian narrowing of the lumen = a simple plaque/stenosis.
# 0.0 = straight pipe, 0.5 = 50 % diameter reduction at the throat.
STENOSIS_SEVERITY = 0.0
STENOSIS_CENTER = 0.0
STENOSIS_WIDTH = 15.0

BOX_FILL_FRACTION = 0.5         # "box" scene: fraction of height filled with fluid

# ---- particles ----
PARTICLE_SPACING = 6.0
SMOOTHING_RADIUS = 12.0         # = 2 x spacing  -> ~33 neighbours in 3D, ~12 in 2D

# ---- physics ----
REST_DENSITY = 1.0              # density is normalised so the starting lattice = 1
PRESSURE_MULTIPLIER = 1000.0    # stiffness k in P = k (rho - rho0); sound speed ~ sqrt(k)
NEAR_PRESSURE_MULTIPLIER = 400.0
VISCOSITY_STRENGTH = 4.0        # relaxation rate (1/time) toward the local mean velocity
CLAMP_NEGATIVE_PRESSURE = True  # no tension: avoids clumping at walls / free surface

COLLISION_DAMPING = 0.3         # restitution: fraction of wall-normal speed kept
WALL_FRICTION = 0.15            # fraction of tangential speed removed per wall contact

DT = 0.02
GRAVITY = np.array([0.0, -9.8, 0.0])[:DIM] if SCENE == "box" else np.zeros(DIM)