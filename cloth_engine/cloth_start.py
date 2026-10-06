import numpy as np
import math
import cloth_constants


    


particle_positions = np.zeros((cloth_constants.NUM_PARTICLES, 2))
particle_previous_positions = np.zeros((cloth_constants.NUM_PARTICLES, 2))
particle_velocities = np.zeros((cloth_constants.NUM_PARTICLES, 2))
particle_masses = np.zeros((cloth_constants.NUM_PARTICLES))
simulation_time = 0.0
def create_grid(numParticles, radius, positions):

    part_per_row = int(math.sqrt(numParticles))
    part_per_collumn = (numParticles-1)/part_per_row + 1
    spacing = radius * 2 + cloth_constants.PARTICLE_SPACING
    for i in range(numParticles):
        position = np.array([((i%part_per_row - part_per_row/2. + 0.5) * spacing), 
                        ((i// part_per_row - part_per_collumn/2. + 0.5) * spacing)])
        positions[i] = position

    return int(part_per_row), int(part_per_collumn)

def create_high_res_grid(part_per_row_low, part_per_collumn_low, spacing_low, subdivision, positions):
    """
    Builds a high-res grid that exactly contains the low-res grid's particles
    as a subset, by subdividing each low-res quad into subdivision x subdivision
    fine quads.
    """
    part_per_row = (part_per_row_low - 1) * subdivision + 1
    part_per_collumn = (part_per_collumn_low - 1) * subdivision + 1
    spacing = spacing_low / subdivision
    numParticles = part_per_row * part_per_collumn

    for i in range(numParticles):
        position = np.array([
            ((i % part_per_row - part_per_row / 2. + 0.5) * spacing),
            ((i // part_per_row - part_per_collumn / 2. + 0.5) * spacing)
        ])
        positions[i] = position

    return int(part_per_row), int(part_per_collumn), int(numParticles)

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
        spring_force += cloth_constants.STIFFNESS * displacement_vector
    
    return spring_force

def calculate_wind_force(position, time):
    phase_offset = position[0] * cloth_constants.RIPPLE_SPACING
    wind_x = (cloth_constants.WIND_STRENGTH * math.sin(
        cloth_constants.WIND_FREQUENCY * time - phase_offset
    ))
    wind_y = (cloth_constants.WIND_STRENGTH * 0.3 * math.sin(
        cloth_constants.WIND_FREQUENCY * time - phase_offset
    ))
    return np.array([wind_x, wind_y])

def update_loop(neighbors, rest_lengths):
    global simulation_time
    simulation_time += cloth_constants.DT
    
    #calculate spring forces
    for i in range(cloth_constants.NUM_PARTICLES):
        if i != cloth_constants.PIN and i != cloth_constants.PIN2 and i != cloth_constants.PIN3: # and i != cloth_constants.PIN4:
            spring_forces = calculate_spring_forces(i, neighbors, rest_lengths)
            wind_force = calculate_wind_force(particle_positions[i], simulation_time)
            total_force = spring_forces + cloth_constants.GRAVITY + wind_force
            acceleration = total_force  # mass = 1 for now

            current_position = particle_positions[i].copy()

            new_position = (
                current_position
                + cloth_constants.DAMPING * (current_position - particle_previous_positions[i])
                + acceleration * cloth_constants.DT ** 2
            )

            particle_previous_positions[i] = current_position
            particle_positions[i] = new_position
   