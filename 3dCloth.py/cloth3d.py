import numpy as np
import math
import cloth3dconstants


    


particle_positions = np.zeros((cloth3dconstants.NUM_PARTICLES, 3))
particle_previous_positions = np.zeros((cloth3dconstants.NUM_PARTICLES, 3))
particle_velocities = np.zeros((cloth3dconstants.NUM_PARTICLES, 3))
particle_masses = np.zeros((cloth3dconstants.NUM_PARTICLES))
simulation_time = 0.0


def create_grid(numParticles, radius, positions, physical_width=None):
    part_per_row = int(math.sqrt(numParticles))
    part_per_collumn = int((numParticles - 1) / part_per_row + 1)

    if physical_width is not None:
        # Fit the grid to a specific physical size (used to match a low-res grid's footprint)
        spacing = physical_width / (part_per_row - 1)
    else:
        # Original behavior: spacing determined by particle radius, unchanged
        spacing = radius * 2 + cloth3dconstants.PARTICLE_SPACING

    for i in range(numParticles):
        position = np.array([
            ((i % part_per_row - part_per_row / 2. + 0.5) * spacing),
            ((i // part_per_row - part_per_collumn / 2. + 0.5) * spacing), 0
        ])
        positions[i] = position

    return int(part_per_row), int(part_per_collumn)

def create_amplified_grid(numParticles_low, radius, resolution_multiplier, positions_high):
    """
    Builds a grid covering the SAME physical area as a numParticles_low grid,
    but with resolution_multiplier times as many particles per row/column.
    Every resolution_multiplier-th high-res particle lands exactly on a
    low-res particle's position, so the two grids stay in exact correspondence.
    """
    part_per_row_low = int(math.sqrt(numParticles_low))

    # Same spacing formula create_grid uses by default (physical_width=None case)
    spacing_low = radius * 2 + cloth3dconstants.PARTICLE_SPACING
    physical_width_low = (part_per_row_low - 1) * spacing_low

    # Construct a part_per_row that's guaranteed to line up with the low-res grid
    part_per_row_high = (part_per_row_low - 1) * resolution_multiplier + 1
    numParticles_high = part_per_row_high * part_per_row_high  # perfect square, by construction

    part_per_row_result, part_per_collumn_result = create_grid(
        numParticles_high, radius, positions_high, physical_width=physical_width_low
    )

    return part_per_row_result, part_per_collumn_result, numParticles_high

def find_all_neighbors(numParticles, part_per_row, part_per_column):
    neighbors = [[] for _ in range(numParticles)]

    for i in range(numParticles):
        row = i // part_per_row
        col = i % part_per_row

        # Left
        if col > 0:
            neighbors[i].append(i - 1)

        # Right
        if col < part_per_row - 1 and i + 1 < numParticles:
            neighbors[i].append(i + 1)

        # Up
        if row > 0:
            neighbors[i].append(i - part_per_row)

        # Down
        if i + part_per_row < numParticles:
            neighbors[i].append(i + part_per_row)

        #Up-Left
        if row > 0 and col > 0:
            neighbors[i].append(i - part_per_row - 1)
            
        #Up-Right
        if row > 0 and col < part_per_row - 1:
            neighbors[i].append(i - part_per_row + 1)
        
        #Down-Left
        if row < part_per_column - 1 and col > 0:
            neighbors[i].append(i + part_per_row - 1)
        
        #Down-Right
        if row < part_per_column - 1 and col < part_per_row - 1:
            neighbors[i].append(i + part_per_row + 1)
    return neighbors

def calculate_rest_lengths(neighbors, positions):
    # Same [particle_index: [len_to_neighbor0, len_to_neighbor1, ...]] shape as neighbors
    rest_lengths = [[] for _ in range(len(neighbors))]

    for i in range(len(neighbors)):
        for j in neighbors[i]:
            dist = np.linalg.norm(positions[j] - positions[i])
            rest_lengths[i].append(dist)

    return rest_lengths


def calculate_spring_forces(index, neighbors, rest_lengths):
    spring_force = 0
    for k, i in enumerate(neighbors[index]):
        direction = particle_positions[i] - particle_positions[index]
        dist = np.linalg.norm(direction)

        if dist == 0:
            continue

        normalized_direction = direction / dist
        displacement_scalar = dist - rest_lengths[index][k]
        displacement_vector = displacement_scalar * normalized_direction
        spring_force += cloth3dconstants.STIFFNESS * displacement_vector
    
    return spring_force

def calculate_wind_force(position, time):
    phase_offset = position[0] * cloth3dconstants.RIPPLE_SPACING
    wind_x = (cloth3dconstants.WIND_STRENGTH * math.sin(
        cloth3dconstants.WIND_FREQUENCY * time - phase_offset
    ))
    wind_y = (cloth3dconstants.WIND_STRENGTH * 0.3 * math.sin(
        cloth3dconstants.WIND_FREQUENCY * time - phase_offset
    ))
    wind_z = cloth3dconstants.WIND_STRENGTH * 0.6 * math.sin(
        cloth3dconstants.WIND_FREQUENCY * time - phase_offset + 1.0  # phase-shifted so z doesn't move in lockstep with x/y
    )
    return np.array([wind_x, wind_y, wind_z])

def update_loop(neighbors, rest_lengths):
    global simulation_time
    simulation_time += cloth3dconstants.DT
    
    #calculate spring forces
    for i in range(cloth3dconstants.NUM_PARTICLES):
        if i != cloth3dconstants.PIN and i != cloth3dconstants.PIN2: # and i != cloth3dconstants.PIN3: # and i != cloth3dconstants.PIN4:
            spring_forces = calculate_spring_forces(i, neighbors, rest_lengths)
            wind_force = calculate_wind_force(particle_positions[i], simulation_time)
            total_force = spring_forces + cloth3dconstants.GRAVITY + wind_force
            acceleration = total_force  # mass = 1 for now

            current_position = particle_positions[i].copy()

            new_position = (
                current_position
                + cloth3dconstants.DAMPING * (current_position - particle_previous_positions[i])
                + acceleration * cloth3dconstants.DT ** 2
            )

            particle_previous_positions[i] = current_position
            particle_positions[i] = new_position
   