from constants import *
import numpy as np
import math
import spatialHash
import time
import fluidflow
positions = np.zeros((NUM_PARTICLES, 2))
velocities = np.zeros((NUM_PARTICLES, 2))
densities = np.zeros((NUM_PARTICLES, 2))
predicted_positions = np.zeros((NUM_PARTICLES, 2))



def create_grid(numParticles, radius, positions):

    part_per_row = int(math.sqrt(numParticles))
    part_per_collumn = (numParticles-1)/part_per_row + 1
    spacing = radius * 2 + PARTICLE_SPACING
    for i in range(numParticles):
        position = np.array([((i%part_per_row - part_per_row/2. + 0.5) * spacing), 
                        ((i// part_per_row - part_per_collumn/2. + 0.5) * spacing)])
        positions[i] = position


"""
Smoothing Functions
"""
def smoothing_kernel(distance):
    if distance >= SMOOTHING_RADIUS:
        return 0
    volume = (np.pi * SMOOTHING_RADIUS**4) / 6
    return (SMOOTHING_RADIUS - distance) * (SMOOTHING_RADIUS - distance) / volume

def smoothing_kernel_derrivative(distance):
    if distance >= SMOOTHING_RADIUS:
        return 0
    
    scale = 12 / (SMOOTHING_RADIUS**4 * np.pi)
    return (distance - SMOOTHING_RADIUS) * scale

def spiky_kernel(distance):
    if distance >= SMOOTHING_RADIUS:
        return 0
    
    volume = np.pi * SMOOTHING_RADIUS**4 / 2
    diff = SMOOTHING_RADIUS - distance

    return diff * diff * diff / volume

def spiky_kernel_derrivative(distance):
    if distance >=  SMOOTHING_RADIUS:
        return 0
    volume = np.pi * SMOOTHING_RADIUS**4 / 2
    coefficient = 3 / volume
    diff = SMOOTHING_RADIUS - distance

    return coefficient * diff * diff

def viscosity_kernel(distance):
    if distance >= SMOOTHING_RADIUS:
        return 0
    volume = np.pi * SMOOTHING_RADIUS**8 / 4
    value = SMOOTHING_RADIUS * SMOOTHING_RADIUS - distance * distance

    return value**3 / volume

    
"""
Force Calculations
"""
def resolve_collisions(index):
    if positions[index][0] >= XLIMITS:
        positions[index][0] = XLIMITS
        velocities[index][0] *= -0.95
    if positions[index][0] <= -XLIMITS:
        positions[index][0] = -XLIMITS
        velocities[index][0] *= -0.95
    if positions[index][1] >= YLIMITS:
        positions[index][1] = YLIMITS
        velocities[index][1] *= -0.95
    if positions[index][1] <= -YLIMITS:
        positions[index][1] = -YLIMITS
        velocities[index][1] *= -0.95

def calculate_density(index, positions):
    global densities
    density = 0
    near_density = 0
    for i in range(NUM_PARTICLES):
        
        dist = np.sqrt((positions[index][0] - positions[i][0])**2 + (positions[index][1] - positions[i][1])**2)
        if dist < SMOOTHING_RADIUS:
            density_influence = smoothing_kernel(dist)
            near_density_influence = spiky_kernel(dist)
            density += density_influence
            near_density += near_density_influence
    densities[index] = np.array([density, near_density])

def calculate_density_vTwo(index, positions):
    global densities
    density = 0
    near_density = 0
    origin_cell = spatialHash.getCellCoords(positions[index], SMOOTHING_RADIUS)
    sqr_radius = SMOOTHING_RADIUS**2
    for i in range(len(spatialHash.offsets)):
        hash = spatialHash.getHashCell(origin_cell + spatialHash.offsets[i])
        key = spatialHash.getKeyFromHash(hash, NUM_PARTICLES)
        cell_start_index = int(spatialHash.start_indices[key])

        if cell_start_index == len(spatialHash.spatial_lookup):
            continue
        for j in range(cell_start_index, len(spatialHash.spatial_lookup)):

            neighbour_index = spatialHash.spatial_lookup[j][0]

            if spatialHash.spatial_lookup[j][1] != key:
                break
            
            neighbour_pos = positions[int(neighbour_index)]
            offset_to_neighbour = neighbour_pos - positions[index]
            sqr_dst_to_neighbour =  np.dot(offset_to_neighbour, offset_to_neighbour)

            if sqr_dst_to_neighbour <= sqr_radius:


                dist = np.sqrt(sqr_dst_to_neighbour)

                density_influence = smoothing_kernel(dist)
                near_density_influence = spiky_kernel(dist)
                density += density_influence
                near_density += near_density_influence

    densities[index] = np.array([density, near_density])
 
def convert_density_to_pressure(density):
    return (density - TARGET_DENSITY) * PRESSURE_MULTIPLIER

def convert_density_to_near_pressure(near_density):
    return NEAR_PRESSURE_MULTIPLIER * near_density

def calculate_external_forces(index):
    global velocities
    velocities[index] += GRAVITY * DT

    #viscosity factor 
    viscosity = np.array([0., 0.])
    
    origin_cell = spatialHash.getCellCoords(positions[index], SMOOTHING_RADIUS)
    sqr_radius = SMOOTHING_RADIUS **2
    for i in range(len(spatialHash.offsets)):
        
        hash = spatialHash.getHashCell(origin_cell + spatialHash.offsets[i])
        key  = spatialHash.getKeyFromHash(hash, NUM_PARTICLES)
        cell_start_index = int(spatialHash.start_indices[key])

        if cell_start_index == len(spatialHash.spatial_lookup):
            continue
        for j in range(cell_start_index, len(spatialHash.spatial_lookup)):
            neighbour_index = spatialHash.spatial_lookup[j][0] 
            
            if spatialHash.spatial_lookup[j][1] != key:
                break
            
            

            if neighbour_index == index:
                continue

            neighbour_pos = positions[int(neighbour_index)]
            offset_to_neighbour = neighbour_pos - positions[index]
            sqr_dst_to_neighbour =  np.dot(offset_to_neighbour, offset_to_neighbour)


            if sqr_dst_to_neighbour <= sqr_radius:
                
                dist =  np.sqrt(sqr_dst_to_neighbour)
                influence = viscosity_kernel(dist)
                viscosity += (velocities[int(neighbour_index)] - velocities[index]) * influence
                
    viscosity *= VISCOSITY_STRENGTH
    if densities[index][0] == 0:
        return
    accel = viscosity/densities[index][0]
    velocities[index] += accel * DT

def apply_mouse_force(mouse_pos, mouse_screen_pos, attract=True):
    global velocities
    
    # convert mouse screen position to simulation coordinates
    sim_x = mouse_screen_pos[0] - (WIDTH // 2)
    sim_y = (HEIGHT // 2) - mouse_screen_pos[1]
    mouse_sim_pos = np.array([sim_x, sim_y], dtype=float)
    
    # vector from each particle to mouse
    diff = mouse_sim_pos - positions  # shape (N, 2)
    dists = np.linalg.norm(diff, axis=1)  # shape (N,)
    
    mask = (dists < MOUSE_RADIUS) & (dists > 0)
    
    if not np.any(mask):
        return
    
    # direction toward mouse
    dirs = diff[mask] / dists[mask, np.newaxis]
    
    # strength falls off toward edge of radius
    strength = (1 - dists[mask] / MOUSE_RADIUS) * MOUSE_STRENGTH
    
    if attract:
        velocities[mask] += dirs * strength[:, np.newaxis] * DT
    else:
        velocities[mask] -= dirs * strength[:, np.newaxis] * DT

def calculate_pressure_force(index, positions):
    pressure = convert_density_to_pressure(densities[index])
    near_pressure = convert_density_to_near_pressure(densities[index])
    pressure_force = np.zeros(2)
    density = densities[index]


    for i in range(NUM_PARTICLES):
        if i != index:
            dist =  np.sqrt((positions[index][0] - positions[i][0])**2 + (positions[index][1] - positions[i][1])**2)
            offsets = positions[i] - positions[index]
            if dist > 0: 
                dir_to = offsets/dist
            else:
                dir_to = np.array([0, 1])
            if dist < SMOOTHING_RADIUS:
                neighbour_density = densities[i][0]
                neighbour_near_density = densities[i][1]
                neighbour_pressure = convert_density_to_pressure(neighbour_density)
                neighbour_near_pressure = convert_density_to_near_pressure(neighbour_near_density)

                shared_pressure = (pressure + neighbour_pressure) * 0.5
                shared_near_pressure = (near_pressure + neighbour_near_pressure) * 0.5
                pressure_force += -dir_to * smoothing_kernel_derrivative(dist) * shared_pressure/neighbour_density
                pressure_force += -dir_to * spiky_kernel_derrivative(dist) * shared_near_pressure/neighbour_near_density
    
    velocities[index] += (pressure_force/density) * DT

def calculate_pressure_force_vTwo(index, positions):
    pressure = convert_density_to_pressure(densities[index])
    near_pressure = convert_density_to_near_pressure(densities[index])
    pressure_force = np.zeros(2)
    density = densities[index][0]
    
    origin_cell = spatialHash.getCellCoords(positions[index], SMOOTHING_RADIUS)
    sqr_radius = SMOOTHING_RADIUS **2
    for i in range(len(spatialHash.offsets)):
        
        hash = spatialHash.getHashCell(origin_cell + spatialHash.offsets[i])
        key  = spatialHash.getKeyFromHash(hash, NUM_PARTICLES)
        cell_start_index = int(spatialHash.start_indices[key])

        if cell_start_index == len(spatialHash.spatial_lookup):
            continue
        for j in range(cell_start_index, len(spatialHash.spatial_lookup)):
            neighbour_index = spatialHash.spatial_lookup[j][0] 
            
            if spatialHash.spatial_lookup[j][1] != key:
                break
            
            

            if neighbour_index == index:
                continue

            neighbour_pos = positions[int(neighbour_index)]
            offset_to_neighbour = neighbour_pos - positions[index]
            sqr_dst_to_neighbour =  np.dot(offset_to_neighbour, offset_to_neighbour)


            if sqr_dst_to_neighbour <= sqr_radius:
                
                dist =  np.sqrt(sqr_dst_to_neighbour)

                if dist > 0: 
                    dir_to = offset_to_neighbour/dist
                else:
                    dir_to = np.array([0., 1.])

        
                neighbour_density = densities[int(neighbour_index)][0]
                neighbour_near_density = densities[int(neighbour_index)][1]
                neighbour_pressure = convert_density_to_pressure(neighbour_density)
                neighbour_near_pressure = convert_density_to_near_pressure(neighbour_near_density)

                shared_pressure = (pressure + neighbour_pressure) * 0.5
                shared_near_pressure = (near_pressure + neighbour_near_pressure) * 0.5
                pressure_force += -dir_to * smoothing_kernel_derrivative(dist) * shared_pressure/neighbour_density
                pressure_force += -dir_to * spiky_kernel_derrivative(dist) * shared_near_pressure/neighbour_near_density
                
    velocities[index] += (pressure_force/density) * DT

def update_positions(index):
    global positions, velocities
    positions[index] += velocities[index] * DT

def simulation_step():
    
    predicted_positions = positions + velocities * DT
    spatialHash.update_spatial_lookup(positions, SMOOTHING_RADIUS)

    for i in range(NUM_PARTICLES):
        calculate_external_forces(i)
    
    for i in range(NUM_PARTICLES):
        calculate_density_vTwo(i, predicted_positions)

    for i in range(NUM_PARTICLES):
        calculate_pressure_force_vTwo(i, predicted_positions)
    
    fluidflow.update_fluid_flow()
    
    for i in range(NUM_PARTICLES):
        update_positions(i)
        resolve_collisions(i)




#neighbours are 1, 9, 10