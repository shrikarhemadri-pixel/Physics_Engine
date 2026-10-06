import itertools
import numpy as np
from constants import *

# Neighbour-cell offsets: 3^DIM cells (9 in 2D, 27 in 3D)
offsets = np.array(list(itertools.product((-1, 0, 1), repeat=DIM)), dtype=np.int64)

# Classic spatial-hashing primes (Teschner et al. 2003); one per axis
HASH_PRIMES = np.array([73856093, 19349663, 83492791], dtype=np.int64)[:DIM]
UINT_MAX_MASK = 0xFFFFFFFF

# Cells are measured from the lower corner of the domain so indices are >= 0
ORIGIN = -HALF_EXTENTS

# Filled by update_spatial_lookup()
spatial_lookup = np.zeros((0, 2), dtype=np.int64)   # rows: [particle_index, cell_key], sorted by key
start_indices = np.zeros(0, dtype=np.int64)         # first row in spatial_lookup for each key
cell_counts = np.zeros(0, dtype=np.int64)           # how many particles carry each key
table_size = 1


def num_cells_x(rad):
    """Number of cells along x (needed to wrap cells when x is periodic)."""
    return int(round(2 * XLIMITS / rad))


def getCellCoords(pos, rad):
    """Position(s) -> integer cell coordinates. Accepts (DIM,) or (N, DIM)."""
    cell = np.floor((np.asarray(pos, dtype=float) - ORIGIN) / rad).astype(np.int64)
    if PERIODIC_X:
        cell[..., 0] %= num_cells_x(rad)
    return cell


def getHashCell(cell):
    """Cell coordinates (..., DIM) -> one non-negative integer per cell."""
    cell = np.asarray(cell, dtype=np.int64)
    return (cell * HASH_PRIMES).sum(axis=-1) & UINT_MAX_MASK


def getKeyFromHash(hash, tablesize):
    return hash % tablesize


def update_spatial_lookup(points, radius):
    """Sort particles by cell key (vectorised version of your original)."""
    global spatial_lookup, start_indices, cell_counts, table_size
    n = len(points)
    table_size = 2 * n + 1                      # larger table -> fewer collisions
    cells = getCellCoords(points, radius)
    keys = getKeyFromHash(getHashCell(cells), table_size)
    order = np.argsort(keys, kind="stable")
    spatial_lookup = np.stack([order, keys[order]], axis=1)
    cell_counts = np.bincount(keys, minlength=table_size)
    start_indices = np.cumsum(cell_counts) - cell_counts


def find_neighbor_pairs(points, radius):
    """
    All ordered pairs (i, j), i != j, with |x_j - x_i| < radius.

    Returns i, j, offset (x_j - x_i, minimum-image in periodic x) and dist.
    Every pair appears twice, (i, j) and (j, i), so per-particle sums are
    just a scatter-add over i.

    Two details matter for correctness:
    * With a small hash table, two of the 3^DIM neighbour cells can share a
      key. Looping over that key twice would double-count its particles, so
      duplicate keys are dropped per particle.
    * Different cells sharing a key just yield extra candidates; the distance
      test below discards them.
    """
    n = len(points)
    m = len(offsets)
    cells = getCellCoords(points, radius)                       # (N, DIM)
    nb = cells[:, None, :] + offsets[None, :, :]                # (N, M, DIM)
    if PERIODIC_X:
        nb[..., 0] %= num_cells_x(radius)
    keys = np.sort(getKeyFromHash(getHashCell(nb), table_size), axis=1)   # (N, M)

    dup = np.zeros_like(keys, dtype=bool)
    dup[:, 1:] = keys[:, 1:] == keys[:, :-1]
    counts = np.where(dup, 0, cell_counts[keys]).ravel()
    starts = start_indices[keys].ravel()

    total = int(counts.sum())
    seg = np.repeat(np.arange(counts.size), counts)
    within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    j = spatial_lookup[starts[seg] + within, 0]
    i = seg // m

    offset = points[j] - points[i]
    if PERIODIC_X:
        length = 2 * XLIMITS
        offset[:, 0] -= length * np.round(offset[:, 0] / length)
    sqr = np.einsum("ij,ij->i", offset, offset)
    keep = (sqr < radius * radius) & (i != j)
    return i[keep], j[keep], offset[keep], np.sqrt(sqr[keep])