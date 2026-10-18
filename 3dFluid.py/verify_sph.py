"""Self-checks for the SPH solver. Run: python verify_sph.py"""
import numpy as np, time
import constants as C
import fluidsim as F, spatialHash as S, fluidflow

ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

h = C.SMOOTHING_RADIUS
rng = np.random.default_rng(1)

# 1. kernel normalisation by Monte-Carlo integration over the support
pts = rng.uniform(-h, h, (2_000_000, C.DIM)); r = np.linalg.norm(pts, axis=1)
vol = (2*h)**C.DIM
for name, fn in [("smoothing", F.smoothing_kernel), ("spiky", F.spiky_kernel), ("viscosity", F.viscosity_kernel)]:
    integral = fn(r).mean() * vol
    check(f"{C.DIM}D kernel '{name}' integrates to 1", abs(integral-1) < 0.02, f"({integral:.4f})")

# 2. derivatives vs. finite differences
rr = np.linspace(0.5, h-0.5, 50); eps = 1e-5
for name, fn, dfn in [("smoothing", F.smoothing_kernel, F.smoothing_kernel_derivative),
                      ("spiky", F.spiky_kernel, F.spiky_kernel_derivative)]:
    fd = (fn(rr+eps) - fn(rr-eps)) / (2*eps)
    check(f"'{name}' derivative matches finite difference", np.allclose(fd, dfn(rr), rtol=1e-4, atol=1e-12))
    check(f"'{name}' derivative is <= 0", np.all(dfn(rr) <= 0))

# 3. hashed neighbour search == brute force (also tests periodic wrap and key collisions)
F.create_grid()
P = F.positions + rng.normal(0, 2.0, F.positions.shape)
P[:,0] = (P[:,0] + C.XLIMITS) % (2*C.XLIMITS) - C.XLIMITS if C.PERIODIC_X else P[:,0]
S.update_spatial_lookup(P, h)
i, j, off, d = S.find_neighbor_pairs(P, h)
D = P[None,:,:] - P[:,None,:]
if C.PERIODIC_X: D[...,0] -= 2*C.XLIMITS*np.round(D[...,0]/(2*C.XLIMITS))
dist = np.linalg.norm(D, axis=2); np.fill_diagonal(dist, np.inf)
bi, bj = np.where(dist < h)
a = set(zip(i.tolist(), j.tolist())); b = set(zip(bi.tolist(), bj.tolist()))
check("hashed neighbour pairs == brute force", a == b, f"({len(a)} pairs, {len(a ^ b)} mismatched, {len(i)==len(a) and 'no duplicates' or 'DUPLICATES'})")

# 4. rest state: lattice density ~ REST_DENSITY in the bulk
F.create_grid(); F.velocities[:] = 0
S.update_spatial_lookup(F.positions, h); pairs = S.find_neighbor_pairs(F.positions, h)
F.calculate_densities(pairs, len(F.positions))
core = np.abs(F.positions[:,1:]).max(axis=1) < C.PIPE_RADIUS*0.3 if C.SCENE=="pipe" else np.ones(len(F.positions),bool)
check("bulk density of initial lattice ~ REST_DENSITY", abs(F.densities[core,0].mean()-C.REST_DENSITY) < 0.05, f"({F.densities[core,0].mean():.3f})")

# 5. momentum conservation of pressure + viscosity forces (random velocities)
F.velocities = rng.normal(0, 3, F.positions.shape)
a = F.pressure_acceleration(pairs, len(F.positions)) + F.viscosity_acceleration(pairs, len(F.positions))
net = (F.PARTICLE_MASS * a).sum(axis=0); scale = np.abs(F.PARTICLE_MASS * a).sum()
check("pressure+viscosity conserve momentum", np.abs(net).max() < 1e-9*scale+1e-12, f"(net {np.abs(net).max():.2e})")
F.velocities[:] = 0

# 6. stability + flow development
F.create_grid()
STEPS = 1500
t = time.time()
for s in range(STEPS): F.simulation_step()
ms = (time.time()-t)/STEPS*1000
finite = np.isfinite(F.positions).all() and np.isfinite(F.velocities).all()
check("simulation stays finite", finite)
print(f"      {len(F.positions)} particles, {ms:.1f} ms/step")
c_snd = np.sqrt(C.PRESSURE_MULTIPLIER); vmax = np.linalg.norm(F.velocities, axis=1).max()
check("flow is weakly compressible (Mach < 0.3)", vmax / c_snd < 0.3, f"(max |v| {vmax:.2f}, c ~ {c_snd:.1f}, Mach {vmax/c_snd:.3f})")
rho = F.densities[:,0]
print(f"      density mean {rho.mean():.3f}, min {rho.min():.3f}, max {rho.max():.3f}  (rest = {C.REST_DENSITY})")
if C.SCENE == "pipe" and C.STENOSIS_SEVERITY == 0:
    c, prof = fluidflow.radial_velocity_profile(F.positions, F.velocities, 6)
    print("      axial velocity vs radius:", " ".join(f"{r:.0f}:{v:.2f}" for r, v in zip(c, prof)))
    # Poiseuille flow is v = v0 (1 - r^2/R^2): linear in r^2. Fit and report R^2.
    good = ~np.isnan(prof); A = np.polyfit(c[good]**2, prof[good], 1)
    pred = np.polyval(A, c[good]**2); r2 = 1 - ((prof[good]-pred)**2).sum() / ((prof[good]-prof[good].mean())**2).sum()
    check("axial profile is parabolic (Poiseuille-like)", r2 > 0.9 and A[0] < 0, f"(R^2 of fit in r^2 = {r2:.3f}; fitted wall radius ~ {np.sqrt(-A[1]/A[0]):.1f} vs {C.PIPE_RADIUS})")
print("ALL PASS" if ok else "SOME CHECKS FAILED")