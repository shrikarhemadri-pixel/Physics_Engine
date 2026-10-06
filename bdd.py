from simulator_particle_core import * 
from fluid_engine import fluidsim
import copy
def rk4(timeStep, partList):
    #Rate of change of the position- the velocity
    k1x = [None]*len(partList)
    k2x = [None]*len(partList)
    k3x = [None]*len(partList)
    k4x = [None]*len(partList)
    #Rate of change of the velocities- the acceleration 
    k1v = [None]*len(partList)
    k2v = [None]*len(partList)
    k3v = [None]*len(partList)
    k4v = [None]*len(partList)
    #Final Rate of changes for velocity and acceleration
    rk4x = [None]*len(partList)
    rk4v = [None]*len(partList)
    #Calculating k1x and k1v for all particles
    for i in range(len(partList)):
        kxtemp = partList[i].velocity.copy()
        k1x[i] = kxtemp
        kvtemp = netAcceleration(partList, i)
        k1v[i] = kvtemp
    #Calculating k2x and k2v for all particles
    for a in range(len(partList)):
        posTemp = partList[a].position.copy()
        xTemp = partList[a].position + (timeStep/2)*k1x[a]
        partList[a].position = xTemp
        vTemp = partList[a].velocity + (timeStep/2)*k1v[a]
        k2x[a] = vTemp
        k2v[a] = (netAcceleration(partList, a)) 
        partList[a].position = posTemp
    #Calculating k3x and k3v for all particles
    for b in range(len(partList)):
        posTemp = partList[b].position.copy()
        xTemp = partList[b].position + (timeStep/2)*k2x[b]
        partList[b].position = xTemp
        vTemp = partList[b].velocity + (timeStep/2)*k2v[b]
        k3x[b]=(vTemp)
        k3v[b]=(netAcceleration(partList, b)) 
        partList[b].position = posTemp
    #Calculating k4x and k4v for all particles
    for c in range(len(partList)):
        posTemp = partList[c].position.copy()
        xTemp = partList[c].position + (timeStep)*k3x[c]
        partList[c].position = xTemp
        vTemp = partList[c].velocity + (timeStep)*k3v[c]
        k4x[c]=(vTemp)
        k4v[c]=(netAcceleration(partList, c)) 
        partList[c].position = posTemp
    for x in range(len(partList)):
        rk4x[x] = (k1x[x]+ 2*k2x[x] + 2*k3x[x] + k4x[x])/6
        rk4v[x] = (k1v[x]+ 2*k2v[x] + 2*k3v[x] + k4v[x])/6
    return rk4x, rk4v

def rk42(timeStep, partList):
    # Rate of change of the position- the velocity
    k1x = [None]*len(partList)
    k2x = [None]*len(partList)
    k3x = [None]*len(partList)
    k4x = [None]*len(partList)
    # Rate of change of the velocities- the acceleration 
    k1v = [None]*len(partList)
    k2v = [None]*len(partList)
    k3v = [None]*len(partList)
    k4v = [None]*len(partList)
    # Final Rate of changes for velocity and acceleration
    rk4x = [None]*len(partList)
    rk4v = [None]*len(partList)

    # ---------------------------
    # Calculating k1x and k1v for all particles
    for i in range(len(partList)):
        kxtemp = partList[i].velocity.copy()
        k1x[i] = kxtemp
        kvtemp = netAcceleration(partList, i)
        k1v[i] = kvtemp

    # ---------------------------
    # Calculating k2x and k2v
    posTempList = [p.position.copy() for p in partList]  # store original positions

    # Update all particles to midpoint positions
    for k in range(len(partList)):
        partList[k].position = posTempList[k] + (timeStep/2)*k1x[k]

    # Compute k2 slopes
    for l in range(len(partList)):
        k2x[l] = partList[i].velocity + (timeStep/2)*k1v[l]
        k2v[l] = netAcceleration(partList, l)

    # Restore original positions
    for a in range(len(partList)):
        partList[a].position = posTempList[a]

    # ---------------------------
    # Calculating k3x and k3v
    # Update all particles to midpoint positions using k2x
    for b in range(len(partList)):
        partList[b].position = posTempList[b] + (timeStep/2)*k2x[b]

    for c in range(len(partList)):
        k3x[c] = partList[c].velocity + (timeStep/2)*k2v[c]
        k3v[c] = netAcceleration(partList, c)

    # Restore original positions
    for d in range(len(partList)):
        partList[d].position = posTempList[d]

    # ---------------------------
    # Calculating k4x and k4v
    # Update all particles to full step positions using k3x
    for e in range(len(partList)):
        partList[e].position = posTempList[e] + (timeStep)*k3x[e]

    for f in range(len(partList)):
        k4x[f] = partList[f].velocity + (timeStep)*k3v[f]
        k4v[f] = netAcceleration(partList, f)

    # Restore original positions
    for g in range(len(partList)):
        partList[g].position = posTempList[g]

    # ---------------------------
    # Weighted average to get final RK4 slopes
    for x in range(len(partList)):
        rk4x[x] = (k1x[x] + 2*k2x[x] + 2*k3x[x] + k4x[x]) / 6
        rk4v[x] = (k1v[x] + 2*k2v[x] + 2*k3v[x] + k4v[x]) / 6

    return rk4x, rk4v


def rk43(timeStep, partList):

    #Rates of Changes of the position ------Velocity
    k1x = [None]*len(partList)
    k2x = [None]*len(partList)
    k3x = [None]*len(partList)
    k4x = [None]*len(partList)

    #Rates of Changes of the velocities ---------Acceleration
    k1v = [None]*len(partList)
    k2v = [None]*len(partList)
    k3v = [None]*len(partList)
    k4v = [None]*len(partList)


    #Final Rates of Changes of the positions and velocities
    rk4x =  [None]*len(partList)
    rk4v =  [None]*len(partList)


    #Making copies of the particles
    partList2 = [None] * len(partList)
    for x in range(len(partList)):
        partList2[x] = partList[x].copy()
    # ---------------------
    #Computing k1x and k1v
    for i in range(len(partList)):
        k1x[i] = partList[i].velocity.copy()
        k1v[i] = netAcceleration(partList, i)

    # ---------------------
    #Computing k2x and k2v

    #Storing the inital Conditions of each particle
    p0 = [None]*len(partList)
    v0 = [None]*len(partList)

    for k in range(len(partList)):

        p0[k] = partList[k].position.copy()
        v0[k] = partList[k].velocity.copy()

    #Calculating for k2x and k2v 
    for a in range(len(partList)):
        partList2[a].position = p0[a] +  k1x[a]* (timeStep/2)
        partList2[a].velocity = v0[a] + k1v[a]* (timeStep/2)

    for b in range(len(partList)):
        k2x[b] = partList2[b].velocity.copy()
        k2v[b] = netAcceleration(partList2, b)

    #Calculating for k3x and k3v 
    for c in range(len(partList)):
        partList2[c].position = p0[c] +  k2x[c] * (timeStep/2)
        partList2[c].velocity = v0[c] + k2v[c] * (timeStep/2)

    for d in range(len(partList)):
        k3x[d] = partList2[d].velocity.copy()
        k3v[d] = netAcceleration(partList2, d)

    
    #Calculating for k4x and k4v
    for e in range(len(partList)):
        partList2[e].position = p0[e] +  k3x[e] * (timeStep)
        partList2[e].velocity = v0[e] + k3v[e] * (timeStep)

    for f in range(len(partList)):
        k4x[f] = partList2[f].velocity.copy()
        k4v[f] = netAcceleration(partList2, f)



    for z in range(len(partList)):
        rk4x[z] = (k1x[z] + 2 * k2x[z] + 2 * k3x[z] + k4x[z])/6
        rk4v[z] = (k1v[z] + 2 * k2v[z] + 2 * k3v[z] + k4v[z])/6
        
    return rk4x, rk4v
            
def leapFrog(partList, timeStep, accelList):
    #Compute the accelerations for all particles

    for i in range(len(partList)):
        accelList[i] = (netAcceleration(partList, i))

    #Half kick velocities
    for a in range(len(partList)):
        partList[a].velocity += accelList[a] * timeStep/2

    #Full Drift Positions
    for b in range(len(partList)):
        partList[b].position += partList[b].velocity * timeStep
    
    #Calculate accelerations at new positions
    for c in range(len(partList)):
        accelList[c] = netAcceleration(partList, c)
    

    #Half kick velocities 
    for d in range(len(partList)):
        partList[d].velocity += accelList[d] * timeStep/2
    
    return [(p.position, p.velocity) for p in partList]

def Yoshida(partList, timeStep, accelList):
    w0 = - (2 ** (1/3))/(2- (2 ** (1/3)))
    w1 = 1/(2-(2**1/3))

    #position fractions
    c1 = w1/2
    c4 = w1/2

    c2 = (w0 + w1)/2
    c3 = (w0 + w1)/2
    #velocity fractions
    d1 = w1
    d3 = w1

    d2 = w0

    #Compute the initial accelerations for all particles

    for i in range(len(partList)):
        accelList[i] = (netAcceleration(partList, i))

    #fractional kick of velocities
    for a in range(len(partList)):
        partList[a].velocity += accelList[a] * d1 * timeStep

    #fractional drift of positions
    for b in range(len(partList)):
        partList[b].position += partList[b].velocity * timeStep * c1
    
    #Calculate accelerations at new positions
    for c in range(len(partList)):
        accelList[c] = netAcceleration(partList, c)
    

    #fractional kick velocities 
    for d in range(len(partList)):
        partList[d].velocity += accelList[d] * timeStep * d2

    #fractional drift of positions
    for e in range(len(partList)):
        partList[e].position += c2 * partList[e].velocity * timeStep

    #Calculate accelerations at new positions
    for x in range(len(partList)):
        accelList[x] = netAcceleration(partList, x)
    
    #fractional kick of velocities
    for f in range(len(partList)):
        partList[f].velocity += accelList[f] * timeStep * d3
    
    #fractional drift of positions
    for g in range(len(partList)):
        partList[g].position += c3 * partList[g].velocity * timeStep
    
    #final position update
    for h in range(len(partList)):
        partList[h].position += c4 * partList[h].velocity * timeStep
    
    return [(p.position, p.velocity) for p in partList]

def rotate_vector(v, degrees):
    R = np.array([
        [np.cos(np.radians(degrees)), -np.sin(np.radians(degrees))],
        [np.sin(np.radians(degrees)),  np.cos(np.radians(degrees))]
    ])
    return R @ v
def rotate_vector2(v, degrees):
    R = np.array([
        [np.cos(degrees), -np.sin(degrees)],
        [np.sin(degrees),  np.cos(degrees)]
    ])
    return R @ v
new_vec = rotate_vector2([0, 1], np.pi/2)

print(new_vec)

import copy
import numpy as np

def bulirsch_stoer_alg(H, n, partlist, accel_list, sequence=[2,4,6,8,10,12], tolerance=1e-6):
    """
    Full Bulirsch-Stoer step combining:
      1. Modified midpoint method (inner engine)
      2. Richardson extrapolation via Neville's algorithm (outer wrapper)
    
    Bugs fixed from original:
      - Inner loop used range(n) instead of range(len(partlist)) for particles
      - Used index i instead of j in the final averaging loop
      - accel_list recomputed before velocity reconstruction (correct order preserved)
      - Richardson extrapolation added as outer loop over the sequence
    """

    n_particles = len(partlist)

    # ------------------------------------------------------------------
    # INNER FUNCTION: Modified Midpoint
    # Runs one full midpoint sweep with n sub-steps on a COPY of partlist
    # Returns positions as a numpy array of shape (n_particles, 2)
    # and writes updated velocity back into the trial system
    # ------------------------------------------------------------------
    def modified_midpoint(trial_list, H, n):
        h = H / n
        trial_accels = [None] * n_particles

        # Store z_0 — positions before the Euler bootstrap
        partPrev = [None] * n_particles
        for i in range(n_particles):
            partPrev[i] = copy.deepcopy(trial_list[i])

        # --- Euler bootstrap: z_1 = z_0 + h * v_0 ---
        for i in range(n_particles):
            trial_list[i].position += trial_list[i].velocity * h
            # velocity is NOT updated here — midpoint stencil is position-only

        # --- Main midpoint loop: z_{m+1} = z_{m-1} + 2h * f(z_m) ---
        for k in range(n - 1):

            # Compute accelerations at current positions z_m
            for i in range(n_particles):                        # FIX 1: was range(n)
                trial_accels[i] = netAcceleration(trial_list, i)

            # Advance all particles and slide the window
            for i in range(n_particles):                        # FIX 2: was range(n)
                pos_next = partPrev[i].position + 2 * h * trial_accels[i]
                partPrev[i] = copy.deepcopy(trial_list[i])     # slide window back
                trial_list[i].position = pos_next

        # --- Recompute accelerations at final positions z_n ---
        trial_accels = [netAcceleration(trial_list, i) for i in range(n_particles)]

        # --- Reconstruct velocity BEFORE overwriting position ---
        # v ≈ (z_n - z_{n-1}) / h using the last two positions in the window
        for j in range(n_particles):                            # FIX 3: was i not j
            trial_list[j].velocity = (
                trial_list[j].position - partPrev[j].position
            ) / h

        # --- Final position averaging ---
        for j in range(n_particles):                            # FIX 4: was i not j
            trial_list[j].position = 0.5 * (
                trial_list[j].position
                + partPrev[j].position
                + h * trial_accels[j]                          # FIX 5: was accel_list[i]
            )

        # Return positions as numpy array for extrapolation arithmetic
        return np.array([p.position.copy() for p in trial_list]), trial_list

    # ------------------------------------------------------------------
    # OUTER LOOP: Richardson Extrapolation via Neville's Algorithm
    #
    # tableau[col] holds the current best diagonal entry at column col.
    # Each new row adds one raw midpoint result, then extrapolates
    # rightward across the tableau cancelling one more power of h² each time.
    # ------------------------------------------------------------------
    tableau   = []      # current diagonal entries
    step_sizes = []     # h_k = H / n_k for each sequence entry
    best_trial = None   # the trial system that produced the converged result

    for row_idx, n in enumerate(sequence):

        # --- Step 1: Run modified midpoint with n sub-steps on a copy ---
        trial_system = [copy.deepcopy(p) for p in partlist]
        T_raw, trial_system = modified_midpoint(trial_system, H, n)

        step_sizes.append(H / n)

        # --- Step 2: Neville extrapolation across this new row ---
        # Start the row with the raw midpoint result
        current_row = [T_raw.copy()]

        for col_idx in range(row_idx):
            T_left  = tableau[col_idx]          # upper-left diagonal entry
            T_right = current_row[col_idx]      # current row, previous column

            h_low   = step_sizes[col_idx]       # older, larger step size
            h_high  = step_sizes[row_idx]       # newer, smaller step size

            # Neville formula — cancels one more power of h²
            ratio_sq = (h_low / h_high) ** 2
            T_extrap = T_right + (T_right - T_left) / (ratio_sq - 1.0)

            current_row.append(T_extrap)

        # --- Step 3: Update tableau diagonal ---
        if row_idx == 0:
            tableau.append(current_row[0])
        else:
            tableau[row_idx - 1] = current_row[-2]  # update previous diagonal slot
            tableau.append(current_row[-1])          # add new diagonal entry

        # --- Step 4: Adaptive error estimate ---
        if row_idx >= 1:
            T_best = tableau[-1]
            T_prev = tableau[-2]

            error_abs = np.linalg.norm(T_best - T_prev)
            error_rel = error_abs / (np.linalg.norm(T_best) + 1e-12)

            print(f"  Row {row_idx+1} | n={n:>3} | "
                  f"abs={error_abs:.3e} | rel={error_rel:.3e}")

            # --- Step 5: Convergence check ---
            if error_rel < tolerance:
                print(f"  Converged at row {row_idx+1} with n={n} sub-steps")

                # Write converged positions AND velocities back to original partlist
                for i in range(n_particles):
                    partlist[i].position = T_best[i].copy()
                    partlist[i].velocity = trial_system[i].velocity.copy()

                # Update the caller's accel_list at the new positions
                for i in range(n_particles):
                    accel_list[i] = netAcceleration(partlist, i)

                return True  # success

    # ------------------------------------------------------------------
    # Did not converge within sequence — caller should reduce H
    # ------------------------------------------------------------------
    print(f"  Warning: did not converge. Final rel_error={error_rel:.3e}")
    return False




def burlischStoer2(H, accel_list, partlist,n=[2, 4, 6, 8, 10, 12] ):
    for a in range(len(n)):
        h = H/n[a]
        Pos_i_2 = [None] * len(partlist)
        Pos_i = [None] * len(partlist)
        Vel_i_2 = [None] * len(partlist)
        Vel_i = [None] * len(partlist)
        
        #Gathering Accelerations
        for x in range(len(partlist)):
            accel_list[x] = netAcceleration(partlist, x)
        #Euler Step 
        for i in range(len(partlist)):
            #Euluer Step for the positions
            Pos_i_2[i] = partlist[i].position.copy()
            partlist[i].position += h * partlist[i].velocity
            Pos_i[i] = partlist[i].position.copy()

            #Euler step for the velocities
            Vel_i_2[i] = partlist[i].velocity.copy()
            partlist[i].velocity += h * accel_list[i]
            Vel_i[i] = partlist[i].velocity.copy()
        for r in range(len(partlist)):
            accel_list[r] = netAcceleration(partlist, r)
        #Main iterative loop 
        for b in range(1, n[a]):
            for c in range(len(partlist)):
                #Modified Midpoint for the positions
                pos_next = Pos_i_2[c] + 2 * h * Vel_i[c]
                Pos_i_2[c] = Pos_i[c].copy()
                Pos_i[c] = pos_next
                
               
                #Modified Midpoint for the velocities
                vel_next = Vel_i_2[c] + 2 * h * accel_list[c]
                Vel_i_2[c] = Vel_i[c].copy()
                Vel_i[c] = vel_next
                
            for y in range(len(partlist)):
                partlist[y].position = Pos_i[y]
                partlist[y].velocity = Vel_i[y]

            for x in range(len(partlist)):    
                #Update Accelerations
                accel_list[x] = netAcceleration(partlist, x)
        
        #Final Position/Velocity vectors
        for t in range(len(partlist)):
            x_final = 0.5 * (Pos_i[t] + Pos_i_2[t] + h * Vel_i[t])
            v_final = 0.5 * (Vel_i[t] + Vel_i_2[t] + h * accel_list[t])




            partlist[t].velocity = v_final
            partlist[t].position = x_final
        #Richardson Extrapolation Step

def richardsonExtrapolation(H, accel_list, partlist, n=[2, 4, 6, 8, 10, 12]):
    
    num_particles = len(partlist)
    num_levels = len(n)

    # Save original state
    orig_pos = [p.position.copy() for p in partlist]
    orig_vel = [p.velocity.copy() for p in partlist]

    tableau_pos = [None] * num_levels
    tableau_vel = [None] * num_levels

    for a in range(num_levels):
        # Reset to original state before each BS run
        for i in range(num_particles):
            partlist[i].position = orig_pos[i].copy()
            partlist[i].velocity = orig_vel[i].copy()
            accel_list[i] = netAcceleration(partlist, i)

        # Run your original burlischStoer2 for a single n value
        burlischStoer2(H, accel_list, partlist, n=[n[a]])

        # Capture the result
        tableau_pos[a] = [None] * num_levels
        tableau_vel[a] = [None] * num_levels
        tableau_pos[a][0] = [partlist[i].position.copy() for i in range(num_particles)]
        tableau_vel[a][0] = [partlist[i].velocity.copy() for i in range(num_particles)]

        # Richardson extrapolation across previous rows
        for k in range(1, a + 1):
            ratio_sq = (n[a] / n[a - k]) ** 2
            denom = ratio_sq - 1.0

            tableau_pos[a][k] = [None] * num_particles
            tableau_vel[a][k] = [None] * num_particles

            for p in range(num_particles):
                tableau_pos[a][k][p] = (
                    tableau_pos[a][k-1][p]
                    + (tableau_pos[a][k-1][p] - tableau_pos[a-1][k-1][p]) / denom
                )
                tableau_vel[a][k][p] = (
                    tableau_vel[a][k-1][p]
                    + (tableau_vel[a][k-1][p] - tableau_vel[a-1][k-1][p]) / denom
                )

    # Write best estimate back to particles
    best = num_levels - 1
    for i in range(num_particles):
        partlist[i].position = tableau_pos[best][best][i].copy()
        partlist[i].velocity = tableau_vel[best][best][i].copy()
        accel_list[i] = netAcceleration(partlist, i)



#Update fluid attributes using 

    





def assignCell(partlist, smoothing_radius):
    grid_cells = []
    for i in range(len(partlist)):
        cellNumX = int(partlist[i].position[0]/smoothing_radius)
        cellNumY = int(partlist[i].position[1]/smoothing_radius)

        grid_cells.append(((cellNumX, cellNumY), i))
        partlist[i].cell = [cellNumX, cellNumY]
    grid_cells.sort(key=lambda x:x[0])

    return grid_cells


def createCellLookup(sortedGrid):
    startCell = {}
    endCell = {}

    currentCell = sortedGrid[0][0]
    startCell[currentCell] = 0

    for i, (cell, particle_index) in enumerate(sortedGrid):

        if cell != currentCell:
            endCell[currentCell] = i
            startCell[cell] = i
            currentCell = cell

    endCell[currentCell] = len(sortedGrid)

    return startCell, endCell

def updateFluidAttributesNew(densList, partlist, smoothing_radius, timeStep, gravity=False, ):
    gridCells = assignCell(partlist, smoothing_radius)
    cell_start, cell_end = createCellLookup(gridCells)

    for i in range(len(partlist)):
        cell = partlist[i].cell
        partlistAppended = []
        local_index = None

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):                      # ← both fixes: dy loop
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor not in cell_start:          # ← indented correctly inside dy
                    continue
                start = cell_start[neighbor]
                end = cell_end[neighbor]

                for j in range(start, end):
                    particle_index = gridCells[j][1]
                    if particle_index == i:
                        local_index = len(partlistAppended)
                    partlistAppended.append(partlist[particle_index])

        if not partlistAppended or local_index is None:
            continue

        densList[i] = calculateDensity(partlist[i].position, partlistAppended, smoothing_radius)

        pressureForce = calculatePressureForce(densList, local_index, partlistAppended, smoothing_radius)

        if densList[i] == 0:
            continue

        pressureAcceleration = pressureForce / densList[i]
        partlist[i].velocity += (pressureAcceleration) * timeStep
        partlist[i].position += partlist[i].velocity[0] * timeStep

    detect_and_resolve_collisions(partlist, timeStep)
def updateFluidAttributes(densList, timeStep, partlist, smoothing_radius):
    for i in range(len(partlist)):
        densList[i] = calculateDensity(partlist[i].position, partlist, smoothing_radius)


    for a in range(len(partlist)):
        pressureForce = calculatePressureForce(a, partlist, smoothing_radius)
        pressureAcceleration = pressureForce/densList[a]
        partlist[a].velocity += pressureAcceleration*timeStep

    
    for b in range(len(partlist)):
        partlist[b].position += partlist[b].velocity * timeStep
        detect_and_resolve_collisions(partlist, timeStep)
    



#Vectorizing calculateSharedPressure() and convertDensitytoPressure

def convertDensitytoPressure(density):
    densityError = density-targetDensity
    pressure = densityError * pressureMultiplier
    return pressure

def calculateSharedPressure(densityA, densityB):
    pressureA = convertDensitytoPressure(densityA)
    pressureB = convertDensitytoPressure(densityB)

    return (pressureA + pressureB)/2




#Vectorizing calculatePressureForce
def calculatePressureForce(densList, partIndex, partlist, smoothing_radius):
    pressureForce = 0
    for i in range(len(partlist)):
        if partIndex != i: 
    
            offset = partlist[i].position - partlist[partIndex].position
            dst = np.linalg.norm(offset)
            dir = get_random_dir() if dst == 0 else offset/dst
            slope = smoothingKernelDerrivative(smoothing_radius, dst)
            density = densList[i]
            sharedPressure = calculateSharedPressure(density, densList[partIndex])
            pressureForce += sharedPressure * dir *partlist[i].mass/density*slope
    
    return pressureForce

def calculatePressureForce(densList, partIndex, partlist, smoothing_radius):
    positions = np.array([p.position for p in partlist])
    masses = np.array([p.mass for p in partlist])

    offsets = positions - positions[partIndex]
    dsts = np.linalg.norm(offsets, axis=1)

    mask = np.arange(len(partlist)) != partIndex

    dirs = np.empty_like(offsets)

    nonzero = mask & (dsts != 0)
    dirs[nonzero] = offsets[nonzero] / dsts[nonzero, None]

    zero = mask & (dsts == 0)
    if np.any(zero):
        angles = np.random.uniform(0, 2*np.pi, np.count_nonzero(zero))
        dirs[zero] = np.column_stack((np.cos(angles), np.sin(angles)))

    slopes = smoothingKernelDerrivative(smoothing_radius, dsts)
    sharedPressures = calculateSharedPressure(densList, densList[partIndex])

    factor = sharedPressures * masses / densList * slopes

    pressureForces = dirs * factor[:, None]

    return np.sum(pressureForces[mask], axis=0)



def get_neighbors(index, partlist, gridCells, cell_start, cell_end):
    cell = partlist[index].cell
    neighbors = []
    for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):                      # ← both fixes: dy loop
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor not in cell_start:          # ← indented correctly inside dy
                    continue
                start = cell_start[neighbor]
                end = cell_end[neighbor]

                for j in range(start, end):
                    particle_index = gridCells[j][1]
                    neighbors.append(particle_index)
    return neighbors


def updateFluidAttributesNew(densList, partlist, smoothing_radius, timeStep, gravity=False, ):
    gridCells = assignCell(partlist, smoothing_radius)
    cell_start, cell_end = createCellLookup(gridCells)

    if gravity == False:
        g = np.array([0., 0.])
    else:
        g = np.array([0., -9.8])

    for i in range(len(partlist)):
        cell = partlist[i].cell
        partlistAppended = []
        local_index = None
        neighbors = get_neighbors(i, partlist, gridCells, cell_start, cell_end)

        if not partlistAppended or local_index is None:
            continue

        densList[i] = calculateDensity(partlist[i].position, neighbors, smoothing_radius)

        pressureForce = calculatePressureForce(densList, local_index, partlistAppended, smoothing_radius)

        if densList[local_index] == 0:
            continue
        
            
        pressureAcceleration = (pressureForce)/ densList[local_index]
        pressureAcceleration += g
        partlist[i].velocity += (pressureAcceleration) * timeStep
        partlist[i].position += partlist[i].velocity * timeStep


        if partlist[i].position[0] < -10 or partlist[i].position[0] > 10:
            partlist[i].velocity *= -1
        if partlist[i].position[1] < -10 or partlist[i].position[1] > 10:
            partlist[i].velocity *= -1

    detect_and_resolve_collisions(partlist, timeStep)



def updateFluidAttributesNew2(densList, partlist, smoothing_radius, timeStep, gravity=False):
    gridCells = assignCell(partlist, smoothing_radius)
    cell_start, cell_end = createCellLookup(gridCells)
    g = np.array([0., -9.8]) if gravity else np.array([0., 0.])

    # Compute all densities once before the main loop
    for i in range(len(partlist)):
        neighbors = get_neighbors(i, partlist, gridCells, cell_start, cell_end)
        neighbor_particles = [partlist[j] for j in neighbors]
        densList[i] = calculateDensity(partlist[i].position, neighbor_particles, smoothing_radius)

    # Now update forces and positions
    for i in range(len(partlist)):
        neighbors = get_neighbors(i, partlist, gridCells, cell_start, cell_end)
        neighbor_particles = [partlist[j] for j in neighbors]
        local_index = neighbors.index(i)

        if not neighbor_particles or local_index is None:
            continue

        neighbor_densities = np.array([densList[j] for j in neighbors])

        pressureForce = calculatePressureForce(neighbor_densities, local_index, neighbor_particles, smoothing_radius)

        if densList[i] == 0:
            continue

        pressureAcceleration = pressureForce / densList[i]
        pressureAcceleration += g
        partlist[i].velocity += pressureAcceleration * timeStep
        partlist[i].position += partlist[i].velocity * timeStep

        if partlist[i].position[0] < -10 or partlist[i].position[0] > 10:
            partlist[i].velocity *= -1
        if partlist[i].position[1] < -10 or partlist[i].position[1] > 10:
            partlist[i].velocity *= -1

    detect_and_resolve_collisions(partlist, timeStep)




def updateSpatialLookup(points, radius):
    startIndices = [INT_MAX] * len(points)
    spatialLookup = [0] * len(points)
    for i in range(len(points)):
        cellX, cellY = positionToCell(points[i])
        cellKey = getKeyfromHash(hashCell(cellX, cellY), len(spatialLookup))
        spatialLookup[i] = (i, cellKey)
        startIndices[i] = sys.maxsize

    spatialLookup.sort(key=lambda x:x[1])

    for i in range(len(points)):
        key = spatialLookup[i][1]
        key_prev = INT_MAX if i == 0 else spatialLookup[i - 1][1]

        if key != key_prev:
            startIndices[key] = i