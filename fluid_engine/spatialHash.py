import numpy as np
import sys
from constants import *
offsets = np.array([(-1, -1), (-1, 0), (-1, 1),
                ( 0, -1), ( 0, 0), ( 0, 1),
                ( 1, -1), ( 1, 0), ( 1, 1)])

TABLE_SIZE = NUM_PARTICLES
spatial_lookup = np.zeros((NUM_PARTICLES, 2))
start_indices = np.zeros(TABLE_SIZE, dtype=int)
spatial_offsets = np.zeros((NUM_PARTICLES))
UINT_MAX_MASK = 0xFFFFFFFF


# hashK1 = 15823
hashK1 = 231104250941
hashK2 = 9737333


#converting the positions into integer cells
def getCellCoords(pos, rad):
    pos = np.array(pos)
    cell = np.floor(pos/rad).astype(int)

    return cell


#converting cell coordinates to a single number
def getHashCell(cell):
    cell = np.array(cell)
    cell = cell.astype(int)
    a = (int(cell[0] * hashK1)) & UINT_MAX_MASK
    b = (int(cell[1] * hashK2)) & UINT_MAX_MASK

    return int(a+b)


def getKeyFromHash(hash, tabelsize):
    return int(hash % tabelsize)

def update_spatial_lookup(points, radius):
    global spatial_lookup, start_indices
    for i in range(len(points)):
        cell = getCellCoords(points[i], radius)
        hash = getHashCell(cell)
        key = getKeyFromHash(hash, len(points))
        spatial_lookup[i] = np.array([i, key])
        start_indices[i] = sys.maxsize
    order = np.argsort(spatial_lookup[:, 1], kind='stable')
    spatial_lookup = spatial_lookup[order]

    for i in range(len(points)):
        key = int(spatial_lookup[i][1])
        key_prev = sys.maxsize if i == 0 else spatial_lookup[i - 1][1]

        if key != key_prev:
            start_indices[key] = int(i)

