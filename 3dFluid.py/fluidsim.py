from constants import *
import numpy as np
import spatialHash
import fluidflow

# State arrays are (re)created by create_grid(); other modules read them as fluidsim.positions etc.
positions = np.zeros((0, DIM))
velocities = np.zeros((0, DIM))
densities = np.zeros((0, 2))        # columns: [density, near density]

PARTICLE_MASS = 1.0                 # set by calibrate_rest_state()
REST_NEAR_DENSITY = 0.0


"""
Smoothing kernels, normalised so that the integral over the support = 1
in the chosen dimension.  h = SMOOTHING_RADIUS.

                      2D                  3D
 (h-r)^2   : 6/(pi h^4)         15/(2 pi h^5)
 (h-r)^3   : 10/(pi h^5)        15/(pi h^6)        ("spiky", near-density)
 (h^2-r^2)^3: 4/(pi h^8)        315/(64 pi h^9)    (viscosity)

Everything accepts scalars or numpy arrays.
"""
_h = SMOOTHING_RADIUS
if DIM == 2:
    K_POW2 = 6.0 / (np.pi * _h**4)
    K_POW3 = 10.0 / (np.pi * _h**5)
    K_VISC = 4.0 / (np.pi * _h**8)
else:
    K_POW2 = 15.0 / (2.0 * np.pi * _h**5)
    K_POW3 = 15.0 / (np.pi * _h**6)
    K_VISC = 315.0 / (64.0 * np.pi * _h**9)


def smoothing_kernel(distance):
    v = np.clip(SMOOTHING_RADIUS - np.asarray(distance, dtype=float), 0.0, None)
    return v * v * K_POW2


def smoothing_kernel_derivative(distance):
    v = np.clip(SMOOTHING_RADIUS - np.asarray(distance, dtype=float), 0.0, None)
    return -2.0 * v * K_POW2                      # d/dr, always <= 0


def spiky_kernel(distance):
    v = np.clip(SMOOTHING_RADIUS - np.asarray(distance, dtype=float), 0.0, None)
    return v**3 * K_POW3


def spiky_kernel_derivative(distance):
    v = np.clip(SMOOTHING_RADIUS - np.asarray(distance, dtype=float), 0.0, None)
    return -3.0 * v * v * K_POW3                  # d/dr, always <= 0


def viscosity_kernel(distance):
    d = np.asarray(distance, dtype=float)
    v = np.clip(SMOOTHING_RADIUS**2 - d * d, 0.0, None)
    return v**3 * K_VISC


"""
Setup
"""
def calibrate_rest_state():
    """Choose the particle mass so the initial lattice has density REST_DENSITY.
    (Your old TARGET_DENSITY=7.5 was ~1000x larger than the densities the kernels
    actually produce, so pressure was always ~ -150.)"""
    global PARTICLE_MASS, REST_NEAR_DENSITY
    n = int(np.ceil(SMOOTHING_RADIUS / PARTICLE_SPACING))
    axis = np.arange(-n, n + 1) * PARTICLE_SPACING
    grid = np.stack(np.meshgrid(*([axis] * DIM), indexing="ij"), axis=-1).reshape(-1, DIM)
    r = np.linalg.norm(grid, axis=1)                 # includes the particle itself (r = 0)
    PARTICLE_MASS = REST_DENSITY / smoothing_kernel(r).sum()
    REST_NEAR_DENSITY = PARTICLE_MASS * spiky_kernel(r).sum()


def create_grid():
    """Fill the domain with a jittered cubic lattice. Returns positions."""
    global positions, velocities, densities
    s = PARTICLE_SPACING
    axes = [np.arange(-e + s / 2, e, s) for e in HALF_EXTENTS]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, DIM)

    if SCENE == "pipe":
        rho = np.linalg.norm(pts[:, 1:], axis=1)
        # keep lattice sites up to s/4 from the wall so the lumen is filled to ~rest density
        pts = pts[rho <= fluidflow.wall_radius(pts[:, 0]) - s / 4]
    else:
        top = -YLIMITS + BOX_FILL_FRACTION * 2 * YLIMITS
        pts = pts[pts[:, 1] < top]

    rng = np.random.default_rng(0)
    pts = pts + rng.uniform(-0.02, 0.02, pts.shape) * s     # break perfect symmetry
    positions = pts
    velocities = np.zeros_like(positions)
    densities = np.zeros((len(positions), 2))
    calibrate_rest_state()
    return positions


"""
Force calculations (vectorised over all neighbour pairs)
"""
def _scatter(i, values, n):
    """Sum per-pair values (P,) or (P, DIM) onto particles i -> (N,) or (N, DIM)."""
    if values.ndim == 1:
        return np.bincount(i, weights=values, minlength=n)
    return np.stack([np.bincount(i, weights=values[:, k], minlength=n) for k in range(values.shape[1])], axis=1)


def calculate_densities(pairs, n):
    """density_i = m * sum_j W(r_ij), including the particle itself."""
    global densities
    i, j, offset, dist = pairs
    dens = _scatter(i, smoothing_kernel(dist), n) + smoothing_kernel(0.0)
    near = _scatter(i, spiky_kernel(dist), n) + spiky_kernel(0.0)
    densities = PARTICLE_MASS * np.stack([dens, near], axis=1)


def convert_density_to_pressure(density):
    p = (density - REST_DENSITY) * PRESSURE_MULTIPLIER
    return np.maximum(p, 0.0) if CLAMP_NEGATIVE_PRESSURE else p


def convert_density_to_near_pressure(near_density):
    return NEAR_PRESSURE_MULTIPLIER * near_density


def pressure_acceleration(pairs, n):
    """
    a_i = + m * sum_j  dir_ij * W'(r) * (P_i + P_j) / (2 rho_i rho_j)

    dir_ij points from i to j and W' <= 0, so positive pressure pushes i
    AWAY from j. The pairwise form is antisymmetric, so total momentum is conserved.
    """
    i, j, offset, dist = pairs
    d = np.maximum(dist, 1e-9)[:, None]
    direction = offset / d
    rho, near = densities[:, 0], densities[:, 1]
    p = convert_density_to_pressure(rho)
    pn = convert_density_to_near_pressure(near)

    shared = 0.5 * (p[i] + p[j]) / (rho[i] * rho[j])
    shared_near = 0.5 * (pn[i] + pn[j]) / (near[i] * near[j])
    mag = smoothing_kernel_derivative(dist) * shared + spiky_kernel_derivative(dist) * shared_near
    return PARTICLE_MASS * _scatter(i, direction * mag[:, None], n)


def viscosity_acceleration(pairs, n):
    """
    a_i = mu * rho0 * m * sum_j (v_j - v_i) W_visc(r) / (rho_i rho_j)

    ~ relaxation of v_i toward the kernel-weighted mean velocity of its
    neighbours at rate VISCOSITY_STRENGTH. Uses a velocity snapshot (no
    order dependence) and is pairwise antisymmetric (momentum conserving).
    """
    i, j, offset, dist = pairs
    rho = densities[:, 0]
    dv = velocities[j] - velocities[i]
    w = viscosity_kernel(dist) / (rho[i] * rho[j])
    return VISCOSITY_STRENGTH * REST_DENSITY * PARTICLE_MASS * _scatter(i, dv * w[:, None], n)


"""
Boundaries
"""
def resolve_collisions():
    global positions, velocities
    e, f = COLLISION_DAMPING, WALL_FRICTION

    if SCENE == "pipe":
        # periodic in x, curved wall in the other axes (a channel in 2D)
        x = positions[:, 0]
        lateral = positions[:, 1:]
        rho = np.maximum(np.linalg.norm(lateral, axis=1), 1e-9)
        R = fluidflow.wall_radius(x)
        out = rho > R
        if np.any(out):
            n = np.concatenate([-fluidflow.wall_radius_slope(x[out])[:, None], lateral[out] / rho[out, None]], axis=1)
            n /= np.linalg.norm(n, axis=1, keepdims=True)                    # outward unit normal
            positions[out, 1:] = lateral[out] * (R[out] / rho[out])[:, None]  # back onto the wall
            v = velocities[out]
            vn = np.einsum("ij,ij->i", v, n)
            v = v - np.where(vn > 0, (1 + e) * vn, 0.0)[:, None] * n         # reflect only if moving outward
            vt = v - np.einsum("ij,ij->i", v, n)[:, None] * n
            velocities[out] = v - f * vt                                      # tangential drag (wall friction)
    else:
        for k in range(DIM):
            for sign in (-1.0, 1.0):
                hit = sign * positions[:, k] >= HALF_EXTENTS[k]
                if not np.any(hit):
                    continue
                positions[hit, k] = sign * HALF_EXTENTS[k]
                moving_out = hit & (sign * velocities[:, k] > 0)
                velocities[moving_out, k] *= -e
                for t in range(DIM):
                    if t != k:
                        velocities[hit, t] *= (1.0 - f)


def simulation_step():
    global positions, velocities
    n = len(positions)

    velocities += (GRAVITY + fluidflow.driving_acceleration()) * DT
    predicted = positions + velocities * DT

    spatialHash.update_spatial_lookup(predicted, SMOOTHING_RADIUS)
    pairs = spatialHash.find_neighbor_pairs(predicted, SMOOTHING_RADIUS)

    calculate_densities(pairs, n)
    accel = pressure_acceleration(pairs, n) + viscosity_acceleration(pairs, n)   # both from the same snapshot
    velocities += accel * DT

    positions += velocities * DT
    fluidflow.wrap_periodic(positions)
    resolve_collisions()