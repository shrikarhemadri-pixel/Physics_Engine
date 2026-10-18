import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import copy

""" Particle Simulator: Simulates the motion of Particles using gravitational and electrostatic forces"""
"""Constants used often for the program"""
G = 6.67430e-11 #N*m^2/kg^2
G = 1
k = 9e9 #N*m^2/C^2
seconds_per_day = 86400 #s 
dt = 0.001 #s it is advisable to increase or decrease dt based on the system attempting to be simulated
earth_radius = 6378137 #meters
earth_mass = 5.9722e24 #kg 
moon_mass = 7.3e22 #kg
sun_mass = 1.989e30 #kg
part_dictionary_x = {} #nested list for storing particle positions- x
part_dictionary_y = {} #nested list for storing particle positions- y

velocity_x = {} #nested list for storing particle velocities- x
velocity_y = {} #nested list for storing particle velocities- y
new_particles = {}
c = 2.998e8 #m/s

#Make a custom class with a Particle with attributes, mass, velocity, and position
class Particle:
    def __init__(self, mass=0., position=[0.,0.], velocity=[0.,0.], charge=0., volume=0., radius=0., cell=[0, 0]):
        self.mass = mass #in kg
        self.position = np.array(position, dtype=float) #in m
        self.velocity = np.array(velocity, dtype=float) #in m/s
        self.charge = charge #in coulumbs
        self.volume = volume #
        self.radius  = radius
        self.cell = cell

    """ 
    The first initialization method was to give attributes that a basic particle may have
    Mass, Position, and Velocity
    """
    #Copy constructor
    def copy(self):
        return Particle(
            mass=self.mass,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            charge=self.charge,
            volume=self.volume,
            radius=self.radius
        )


    #Calculate Kinetic Energy of the Particle
    def kineticEnergy(self):
        #By using the instance mehtod, convert the velocity vector into a magnitude value
        #the kinetic energy accesses the static method to calculate the total kinetic energy
        speed = np.sqrt(self.velocity[0]**2+self.velocity[1]**2)
        return 0.5*self.mass*speed**2
    def momentum(self):
        return self.mass * self.velocity
    
    def positionX(self):
        return self.position[0]
    def positionY(self):
        return self.position[1]



""" End of Class

Start of Main Methods
"""

"""
Calculating Distance
"""


def Distance(p1, p2):
    #Calculate the distance as a vector and magnitude of the to particles positions 
    epsilon = 1e-3
    return np.sqrt((p1.position[0]-p2.position[0])**2 + (p1.position[1]-p2.position[1])**2 + epsilon**2), (p2.position - p1.position)


def total_energy(partList):
    KE = 0  # Kinetic energy
    PE = 0  # Potential energy
    
    # Kinetic energy
    for p in partList:
        KE += 0.5 * p.mass * np.dot(p.velocity, p.velocity)
    
    # Potential energy
    for i in range(len(partList)):
        for j in range(i+1, len(partList)):
            dist, r = Distance(partList[i], partList[j])
            PE += -G * partList[i].mass * partList[j].mass / dist
    
    return KE + PE
def kineticEnergy(partList):
    KE = 0 
    for p in partList:
        KE += 0.5 * p.mass * np.dot(p.velocity, p.velocity)
    return KE

"""Calculating Force
"""
def gForce(p1, p2):
    #Calculations to return both force vectors and magnitudes
    distance,vec = Distance(p1, p2)
    #General Newtons Gravitational Formula
    forceG = (G*p1.mass*p2.mass)/distance**2
    #The unit vector of the force
    r_hat = vec/distance
    #Multiplying the unit vector by the force magnitude to get vector values
    force_vector = r_hat*forceG
    return force_vector
def cForce(p1, p2):
    #Getting the distance magnitudes and vectors
    distance, vec = Distance(p1, p2)
    #Coulumb force formula
    forceC = (k*p1.charge*p2.charge)/distance**2
    #Getting the unit vecotr of the force
    r_hat = vec/distance
    forceC_vector = r_hat*forceC
    return forceC_vector
#Calculating the Acceleration of the system
def netForce(particle_list, index):
    f = np.array([0., 0.])
    for i in range(len(particle_list)):
        if i != index:
            
            f +=  (gForce(particle_list[index], particle_list[i]) 
                    -cForce(particle_list[index], particle_list[i]))

    return f
"""
Calculating Acceleration
"""
def netAcceleration(particle_list, index):
    #The gForce method returns both a vector for the force as well as that vectors magnitude
    """Simply by using the logic from F = ma, dividing both force values should give us the acceleration
    of the system both for vectors and scalars. 
    """
    force_vector = netForce(particle_list, index)
    force_mag = np.linalg.norm(force_vector)
    return force_vector/particle_list[index].mass

def detect_and_resolve_collisions(total_system, timestep):
    """
    Leapfrog-compatible collision detection and resolution.
    
    Called AFTER the drift (position update) step, BEFORE the second half-kick.
    Detects overlapping particles by radius, resolves them with elastic collision
    physics, and separates them to prevent tunneling on the next step.
    
    Pairs each particle against every other exactly once (i < j guard).
    """
    n = len(total_system)
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = total_system[i]
            p2 = total_system[j]

            # --- Collision threshold ---
            # Use sum of radii if defined, else fall back to a softening length
            collision_dist = (p1.radius + p2.radius) if (p1.radius > 0 and p2.radius > 0) else 1e-2

            # --- Distance between particles ---
            delta = p2.position - p1.position          # displacement vector p1 -> p2
            dist  = np.linalg.norm(delta)

            if dist < collision_dist:
                # ---- 1. Elastic velocity exchange along collision normal ----
                normal = delta / (dist + 1e-12)        # unit vector along line of centers
                rel_vel = p1.velocity - p2.velocity    # relative velocity

                # Only resolve if particles are approaching (dot product < 0)
                approach_speed = np.dot(rel_vel, normal)
                if approach_speed <= 0:
                    continue

                # 1D elastic collision impulse projected onto normal
                impulse = (2 * approach_speed) / (p1.mass + p2.mass)

                p1.velocity -= impulse * p2.mass * normal
                p2.velocity += impulse * p1.mass * normal

                # ---- 2. Positional correction (prevent overlap / tunneling) ----
                # Push the particles apart so they no longer overlap
                overlap   = collision_dist - dist
                correction = normal * (overlap / 2.0)

                p1.position -= correction
                p2.position += correction

"""
Implementing a Runge Kutta 4th degree method of numerical integration into the system
"""
def rk4_accel(partList, timeStep):
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
        k2x[b] = partList2[b].velocity
        k2v[b] = netAcceleration(partList2, b)

    #Calculating for k3x and k3v 
    for c in range(len(partList)):
        partList2[c].position = p0[c] +  k2x[c] * (timeStep/2)
        partList2[c].velocity = v0[c] + k2v[c] * (timeStep/2)

    for d in range(len(partList)):
        k3x[d] = partList2[d].velocity
        k3v[d] = netAcceleration(partList2, d)

    
    #Calculating for k4x and k4v
    for e in range(len(partList)):
        partList2[e].position = p0[e] +  k3x[e] * (timeStep)
        partList2[e].velocity = v0[e] + k3v[e] * (timeStep)

    for f in range(len(partList)):
        k4x[f] = partList2[f].velocity
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
    detect_and_resolve_collisions(partList, timeStep)
    #Calculate accelerations at new positions
    for c in range(len(partList)):
        accelList[c] = netAcceleration(partList, c)
    

    #Half kick velocities 
    for d in range(len(partList)):
        partList[d].velocity += accelList[d] * timeStep/2


def yoshida4(partList, timeStep, accelList):
    w1 = 1 / (2 - 2**(1/3))
    w0 = -2**(1/3) / (2 - 2**(1/3))

    leapFrog(partList, w1 * timeStep, accelList)
    leapFrog(partList, w0 * timeStep, accelList)
    leapFrog(partList, w1 * timeStep, accelList)

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

"""
Updating previous methods/attributes
"""

acceleration_plot = {}

def update_attributes(total_system, time_step=dt):
    #Using Kinematic formulas, updating velocity, and position of the system, and then 
    #Use step counts to update the position, velocity, and acceleration of the system
    acceleration_list = []
 
    for i in range(len(total_system)):
        acceleration_list.append(netAcceleration(total_system, i))

    rk4x, rk4v = rk4_accel(total_system, time_step)

    for k in range(len(acceleration_list)):
        
        v = total_system[k].velocity.copy()
        p = total_system[k].position.copy()

        total_system[k].velocity = v +  rk4v[k] * (time_step)
        total_system[k].position = p + rk4x[k] * (time_step)

def update_attributes2(total_system, time_step=dt):
    acceleration_list = [None] * len(total_system)
    
    leapFrog(total_system,time_step, acceleration_list)
        
def update_attributes3(total_system, time_step=dt):
    acceleration_list = [None] * len(total_system)
    
    yoshida4(total_system, time_step, acceleration_list)

def update_attributes4(total_system, time_step=dt):
    acceleration_list = []
    for i in range(len(total_system)):
        acceleration_list.append(netAcceleration(total_system, i))
    richardsonExtrapolation(dt, acceleration_list, total_system)

        





        
        


"""Quadratic Function used to develop a path of intersection per each timestep"""
def quadraticFunction(acc_1, acc_2, initial_velocity_1, initial_velocity_2, initial_position_1, initial_position_2, t_initial):
    # Solve the quadratic equation ax**2 + bx + c = 0
    a = 0.5 * (acc_1 - acc_2)
    b = initial_velocity_1 - initial_velocity_2
    c = initial_position_1 - initial_position_2
    if c==0 and a==0 and b==0:
        return "constant"
    # calculate the discriminant
    d = (b**2) - (4*a*c)

    if d < 0:
        return -1
   
    # find two solutions
    sol1 = (-b-np.sqrt(d))/(2*a)
    sol2 = (-b+np.sqrt(d))/(2*a)

    if t_initial < sol1 < t_initial +dt:
        return  sol1
    elif t_initial < sol2 < t_initial +dt:
        return sol2
    else:
        return -1

"""The collision function that makes an elastic collision- updating the position and velocity of the particles using momentum rather than acceleration """
def collision(p1, p2, timestep):
    #Formula for perfectly elastic collisions- the vectors of the velocity basically reverse
    p1.velocity[0] = ((p1.mass-p2.mass)*p1.velocity[0]+2*p2.mass*p2.velocity[0])/(p1.mass+p2.mass)
    p1.velocity[1] = ((p1.mass-p2.mass)*p1.velocity[1]+2*p2.mass*p2.velocity[1])/(p1.mass+p2.mass)

    p2.velocity[0] = -((p2.mass-p1.mass)*p2.velocity[0]+2*p1.mass*p1.velocity[0])/(p1.mass+p2.mass)
    p2.velocity[1] = -((p2.mass-p1.mass)*p2.velocity[1]+2*p1.mass*p1.velocity[1])/(p1.mass+p2.mass)

    #Updating the positions based on said velocity
    p1.position += p1.velocity*timestep
    p2.position += p2.velocity*timestep



"""
Simulation Functions
"""
def createParticles(num_particles, mass, ca):
    random_num = random.randint(-1e5, 1e5)
    for i in range(num_particles):
        new_particles[i] = Particle(mass, position=[random_num, random_num], velocity=[random_num, random_num],charge=ca)
        if i > 0:
            if new_particles[i].position == new_particles[i-1]:
                new_particles[i].position = [random_num, random_num]
        
def makeLists(num_particles):
    for  i in range((num_particles)):
        part_dictionary_x[i] = []
        part_dictionary_y[i] = []

        velocity_x[i] = []
        velocity_y[i] = []

        acceleration_plot[i] = []

 

energyList = []
keList = []
def Simulate(iterations, total_system=list):
    makeLists(len(total_system))
    # Storing initial positions of all particles
    for k in range(len(total_system)):
            part_dictionary_x[k].append(total_system[k].position[0])
            part_dictionary_y[k].append(total_system[k].position[1])
            velocity_x[k].append(total_system[k].position[0])
            velocity_y[k].append(total_system[k].position[1])

    for iter in range(iterations):
        x_collision_time = 0
        y_collision_time = 0
        time_t = 0
        energyList.append(total_energy(total_system))
        keList.append(kineticEnergy(total_system))
        update_attributes2(total_system)

        for item in range(len(total_system)):
            #Appending all of the attributes to lists to store the values and use them for plots and animations
            part_dictionary_x[item].append(total_system[item].position[0])
            part_dictionary_y[item].append(total_system[item].position[1])
            velocity_x[item].append(total_system[item].velocity[0])
            velocity_y[item].append(total_system[item].velocity[1])
        time_t+=dt
    # print(f"the first value is {energyList[0]}")    
    # print(f"The last value is {energyList[len(energyList) - 1]}")
    print(f"the first kinetic energy value is {keList[0]}")
    print(f"the last kiinetic energy value is {keList[len(energyList)-1]}")
    print(f"The total energy of the system is {energyList[0]}")
    
 
"""fluids: Start of Fluid Simulation classes
"""
class fluid: 
    def __init__(self):
        pass


"""
Plotting Functions
"""
#Easy function for quiver plots
def quiver_plot(x_list, y_list, x_axis, y_axis, title):
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
   
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    for i in range(len(x_list)):
        plt.scatter(x_list[i], y_list[i])
        plt.quiver(x_list[i][:-1], y_list[i][:-1], np.diff(x_list[i]), np.diff(y_list[i]), angles='xy', scale_units='xy', scale=1)
        if min(x_list[i]) < min_x:
            min_x = min(x_list[i])
        if min(y_list[i]) < min_y:
            min_y = min(y_list[i])
        if max(x_list[i]) > max_x:
            max_x = max(x_list[i])
        if max(y_list[i]) > max_y:
            max_y = max(y_list[i])
    
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.show()
#Easy function for 3D plots
def threeD_plot(x, y, z, x_axis, y_axis, z_axis):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel(z_axis)
    ax.plot(x, y, z)
    plt.show()

def animate_plot(i):
    fig = plt.figure()
    l, = plt.plot([], [], 'k-')

    metadata = dict(title='Movie', artist='Shrikar')
    writer = PillowWriter(fps=20, metadata=metadata)
    plt.xlim(min(part_dictionary_x[i]), max(part_dictionary_x[i]) )
    plt.ylim(min(part_dictionary_y[i]), max(part_dictionary_y[i]) )

    xlist = []
    ylist = []

    with writer.saving(fig, "ParticleSimulator.gif", 100):
        for x, y in zip(part_dictionary_x[i], part_dictionary_y[i]):
            xlist.append(x)
            ylist.append(y)
            l.set_data(xlist, ylist)
        
            writer.grab_frame()








