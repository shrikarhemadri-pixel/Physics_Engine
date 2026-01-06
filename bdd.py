from simulator_particle_core import * 

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
            

p1 = Particle(1, [1, 0], [0, 0])
p2 = Particle(1, [0, 0], [0, 0])

print(rk43(dt, [p1, p2]))


