"""  
Test Case 1: Pilot

partA = Particle(1e11, [-1, 0])
partB = Particle(1e11, [1, 0])

#for i in range(100):
#  print(f"Acceleration Vec P1: {update_attributes(partA, partB)}")


accelerationMagPA, accelerationVecPA, accelerationMagPB, accelerationVecPB = netAcceleration(partA, partB)

xA = [partA.position[0]]
yA = [partA.position[1]]



xB = []
yB = []

for i in range(100):
    update_attributes(partA, partB)
    xA.append(partA.position[0])
    yA.append(partA.position[1])

#Taking the difference of position and taking it over dt to equal velocity
dxA = np.diff(xA)/dt
dyA = np.diff(yA)/dt
#Taking the difference of velocity and taking it over dt to equal acceleration
dx2A = np.diff(dxA)/dt
d2yA = np.diff(dyA)/dt

"""


"""
Test Case 2: Satelite Orbiting Moon

height_of_object = 3.5786e7 #meters


earth = Particle(5.9722e24, [0., 0.], [0., 0.])
orbital_velocity = np.sqrt((G*earth.mass)/(height_of_object+earth_radius))

satelite = Particle(6600, [0, height_of_object+earth_radius],[orbital_velocity,0] )

satelite_position_x = [satelite.position[0]]
satelite_position_y  = [satelite.position[1]]
dt_list = [0]
for i in range(864000*0.25):
    update_attributes(satelite, earth)
    satelite_position_x.append(satelite.position[0])
    satelite_position_y.append(satelite.position[1])
    dt_list.append(dt*(i+1))


print("y list", satelite_position_y)
quiver_plot(satelite_position_x,satelite_position_y, "time (s)", "Y position (m) ", "Satelite Y position (m/s)")

"""

"""

Test Case 3: twin moons orbiting a planet without simulating gravitation between the two moons

planet = Particle(5.9722e24)


altitude = 3.5786e7 #meters
orbital_velocity = np.sqrt((G*planet.mass)/(altitude+earth_radius))


moon = Particle(7.3e22, [0, altitude + earth_radius], [orbital_velocity, 0])
moon_2 = Particle(7.3e22, [0, -(altitude + earth_radius)], [orbital_velocity, 0] )

#Attempting to make a system that contains three objects
#Testing Secnairos where Gravity does not have a massive between two moons


Position Points for the moons



#X values
moon_x = [moon.position[0]]
moon_2_xpos = [moon_2.position[0]]

#Y values
moon_y = [moon.position[1]]
moon_2_ypos = [moon_2.position[1]]

#Generating Distance values 
distance_mag, distance_vec = Distance(moon, moon_2)
moon_2_distance = [distance_mag]

#Getting time values
dt_list = [0]
for i in range(8640):
    update_attributes(planet, moon)
    moon_x.append(moon.position[0])
    moon_y.append(moon.position[1])
    update_attributes(planet, moon_2)
    moon_2_xpos.append(moon_2.position[0])
    moon_2_ypos.append(moon_2.position[1])
    moon_2_distance.append(distance_mag)
    dt_list.append(dt*(i+1))

quiver_plot(dt_list, moon_2_distance, "Time (s)", "Distance(m)", "Distance between the moons (m)")

"""

"""
Test Case 4: twin moons orbiting a planet with gravity being an effect

#X values
moon_x = [moon.position[0]]
moon_2_xpos = [moon_2.position[0]]

#Y values
moon_y = [moon.position[1]]
moon_2_ypos = [moon_2.position[1]]

#Generating Distance values 
distance_mag, distance_vec = Distance(moon, moon_2)
moon_2_distance = [distance_mag]

#Getting time values
dt_list = [0]


for i in range(86400):
    update_attributes(planet, moon)
    update_attributes(planet, moon_2)
    update_attributes(moon, moon_2)

    #Update moon one position
    moon_x.append(moon.position[0])
    moon_y.append(moon.position[1])
    #Update moon two position
    moon_2_xpos.append(moon_2.position[0])
    moon_2_ypos.append(moon_2.position[1])
    #Distance Calculation
    moon_2_distance.append(distance_mag)
    #Time plots
    dt_list.append(dt*(i+1))

quiver_plot(moon_x, moon_y, "Time (s)", "Distance (m)", "Moon 1 orbit")


"""

"""
planet = Particle(5.9722e24)


altitude = 3.5786e7 #meters
orbital_velocity = np.sqrt((G*planet.mass)/(altitude+earth_radius))


moon = Particle(7.3e22, [0, altitude + earth_radius], [orbital_velocity, 0])
moon_2 = Particle(7.3e22, [0, -(altitude + earth_radius)], [-orbital_velocity, 0] )

total_system = [moon, moon_2, planet]

#Attempting to make a system that contains three objects
#Testing Secnairos where Gravity does have a massive between two moons


Position Points for the moons



#X values
moon_x = [moon.position[0]]
moon_2_xpos = [moon_2.position[0]]

#Y values
moon_y = [moon.position[1]]
moon_2_ypos = [moon_2.position[1]]

#Generating Distance values 

moon_2_distance = [Distance(moon, moon_2)[0]]

#Getting time values
dt_list = [0]

#Values When Gravity is affecting the attribuites
for i in range(96000):

    for k in range(len(total_system)):
        update_attributes(total_system[k], netAcceleration(total_system, k))

    #Update moon one position
    moon_x.append(moon.position[0])
    moon_y.append(moon.position[1])
    #Update moon two position
    moon_2_xpos.append(moon_2.position[0])
    moon_2_ypos.append(moon_2.position[1])
    #Distance Calculation
    moon_2_distance.append(Distance(moon, moon_2)[0])
    #Time plots
    dt_list.append(dt*(i+1))


plt.title("Comparing twin moon trajectory")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")

plt.xlim(min(moon_2_xpos), max(moon_x))
plt.ylim(min(moon_2_ypos), max(moon_y))
plt.quiver(moon_x[:-1], moon_y[:-1], np.diff(moon_x), np.diff(moon_y), color='blue', angles='xy', scale_units='xy', scale=1, label='Moon1')
plt.quiver(moon_2_xpos[:-1], moon_2_ypos[:-1], np.diff(moon_2_xpos), np.diff(moon_2_ypos), color='red', angles='xy', scale_units='xy', scale=1, label='Moon2')
plt.legend()
plt.show()



"""