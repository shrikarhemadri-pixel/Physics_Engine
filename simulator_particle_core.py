import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

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
    def __init__(self, mass=0., position=[0.,0.], velocity=[0.,0.], charge=0., volume=0., radius=0.):
        self.mass = mass #in kg
        self.position = np.array(position, dtype=float) #in m
        self.velocity = np.array(velocity, dtype=float) #in m/s
        self.charge = charge #in coulumbs
        self.volume = volume #
        self.radius  = radius

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
    return np.sqrt((p1.position[0]-p2.position[0])**2 + (p1.position[1]-p2.position[1])**2), (p2.position - p1.position)


def total_energy(partList):
    KE = 0  # Kinetic energy
    PE = 0  # Potential energy
    
    # Kinetic energy
    for p in partList:
        KE += 0.5 * p.mass * np.dot(p.velocity, p.velocity)
    
    # Potential energy
    for i in range(len(partList)):
        for j in range(i+1, len(partList)):
            r = partList[i].position - partList[j].position
            dist = np.linalg.norm(r)
            PE += -G * partList[i].mass * partList[j].mass / dist
    
    return KE + PE

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

 


def Simulate(iterations, total_system=list):
    makeLists(len(total_system))
    # Storing initial positions of all particles
    for k in range(len(total_system)):
            part_dictionary_x[k].append(total_system[k].position[0])
            part_dictionary_y[k].append(total_system[k].position[1])
            velocity_x[k].append(total_system[k].position[0])
            velocity_y[k].append(total_system[k].position[1])
            for l in range(len(total_system)):
                if k != l:
                    print(f"the distance between {k} and {l} is {Distance(total_system[k], total_system[l])}")


    for iter in range(iterations):
        x_collision_time = 0
        y_collision_time = 0
        time_t = 0
        #Checking for intersections/collisions
        for i in range(len(total_system)):
            #Recieving initial accelerations of the first particle
            accel_1 = netAcceleration(total_system, i)
            #Beginning of the nested for loop- The loop checks each particle against all of the other particles involved in the system
            for k in range(len(total_system)):
                if i != k and k > i: #Making sure that the particle isn't checked against itself and that it isn't checked twice
                    #Recieving initial accelerations of the second particle
                    accel_2 = netAcceleration(total_system, k)
                    #the collision solution of the x coordinate and y coordinate
                    x_collision_time = quadraticFunction(accel_1[0], accel_2[0], total_system[i].velocity[0], total_system[k].velocity[0], total_system[i].position[0], total_system[k].position[0], time_t)
                    y_collision_time = quadraticFunction(accel_1[1], accel_2[1], total_system[i].velocity[1], total_system[k].velocity[1], total_system[i].position[1], total_system[k].position[1], time_t)
                    #Making sure that the time that the x coordinates intersect and the time that the y coordinates intersect are equal to each other
                    if ((x_collision_time == y_collision_time) and (y_collision_time != -1)) or (y_collision_time == "constant" and x_collision_time != -1) or (x_collision_time == "constant" and y_collision_time != -1):
                        #System updates the attributes of the system (moving the particle) at the timestep at the time that the particles collide with each other
                        #The particle should be at the same position now
                        collision_time = x_collision_time
                        if x_collision_time == "constant":
                            collision_time = y_collision_time
                        
                        update_attributes(total_system, collision_time)
                        print(f'{i} collides with {k} there is a collision at {collision_time} seconds at {total_system[i].position} with a velocity of {total_system[i].velocity}')
                        print(f'time_t is equal to {time_t}')
                        #collision function is implemented- The velocities of the particle changes elastically
                        collision(total_system[i], total_system[k], (dt-collision_time))
                        print(f'the velocity of {i} is {total_system[i].velocity} and the position of {i} is {total_system[i].position}')
                        print(f'the velocity of {k} is {total_system[k].velocity} and the position of {k} is {total_system[k].position}')
                        part_dictionary_x[i].append(total_system[i].position[0])
                        part_dictionary_y[i].append(total_system[i].position[1])
                        velocity_x[i].append(total_system[i].velocity[0])
                        velocity_y[i].append(total_system[i].velocity[1])
                        part_dictionary_x[k].append(total_system[k].position[0])
                        part_dictionary_y[k].append(total_system[k].position[1])
                        velocity_x[k].append(total_system[k].velocity[0])
                        velocity_y[k].append(total_system[k].velocity[1])
                    
                   
                        



        update_attributes(total_system)

        print(total_energy(total_system))
        for item in range(len(total_system)):
            #Appending all of the attributes to lists to store the values and use them for plots and animations
            part_dictionary_x[item].append(total_system[item].position[0])
            part_dictionary_y[item].append(total_system[item].position[1])
            velocity_x[item].append(total_system[item].velocity[0])
            velocity_y[item].append(total_system[item].velocity[1])
        time_t+=dt
        
    
 





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








