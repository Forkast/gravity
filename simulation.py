import pygame
import random
import math

class OctreeNode:
    def __init__(self, position, size):
        self.position = position  # Position of the node (center point)
        self.size = size  # Size of the node
        self.children = []  # Child nodes (subdivisions of the space)
        self.particles = []  # Particles contained within this node
    
    def traverse(self):
        # Generator function to yield each particle in the octree

        for particle in self.particles:
            yield particle

        for child in self.children:
            yield from child.traverse()

    def delete_particle(self, particle):
        # Delete particle from the octree
        if particle in self.particles:
            self.particles.remove(particle)
            return
        
        for child in self.children:
            if particle.in_boundary(child.boundary):
                child.delete_particle(particle)
                return
    
    def in_boundary(self, position, child):
        # Check if position is within the boundary of the child node
        x, y, z = position
        half_size = child.size / 2
        child_x, child_y, child_z = child.position

        return (
            child_x - half_size <= x <= child_x + half_size and
            child_y - half_size <= y <= child_y + half_size and
            child_z - half_size <= z <= child_z + half_size
        )
    
    def boundary(self):
        # Return the boundary of the node as a tuple of (min_bound, max_bound)
        half_size = self.size / 2
        x, y, z = self.position
        min_bound = (x - half_size, y - half_size, z - half_size)
        max_bound = (x + half_size, y + half_size, z + half_size)
        return min_bound, max_bound


class Octree:
    def __init__(self, position, size, max_particles=4, max_depth=10):
        self.root = OctreeNode(position, size)
        self.max_particles = max_particles
        self.max_depth = max_depth

    def insert(self, particle, node=None):
        if node is None:
            node = self.root

        if len(node.particles) < self.max_particles or len(node.children) >= self.max_depth:
            node.particles.append(particle)
            particle.node = node
        else:
            if not node.children:
                self.subdivide(node)
            child_index = self.get_child_index(particle.position, node.position)
            self.insert(particle, node.children[child_index])

    def subdivide(self, node):
        half_size = node.size / 2
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    child_position = [
                        node.position[0] + half_size * (i - 0.5),
                        node.position[1] + half_size * (j - 0.5),
                        node.position[2] + half_size * (k - 0.5)
                    ]
                    node.children.append(OctreeNode(child_position, half_size))

    def get_child_index(self, position, node_position):
        x, y, z = position
        nx, ny, nz = node_position
        if x < nx:
            if y < ny:
                if z < nz:
                    return 0
                else:
                    return 1
            else:
                if z < nz:
                    return 2
                else:
                    return 3
        else:
            if y < ny:
                if z < nz:
                    return 4
                else:
                    return 5
            else:
                if z < nz:
                    return 6
                else:
                    return 7

    def query(self, position, node=None):
        if node is None:
            node = self.root

        if not node.children:
            return node.particles

        child_index = self.get_child_index(position, node.position)
        return self.query(position, node.children[child_index])
    
    def traverse(self):
        yield from self.root.traverse()

    def __getitem__(self, index):
        for particle in self.root.traverse():
            if index == 0:
                return particle
            index -= 1

        raise IndexError("Index out of range")
        
    def __len__(self):
        return self.get_particle_count()

    def get_particle_count(self, node=None):
        if node is None:
            node = self.root

        count = len(node.particles)

        for child in node.children:
            count += self.get_particle_count(child)

        return count

    def __iter__(self):
        return self.traverse()

    def delete(self, particle):
        self.root.delete_particle(particle)

# Particle class representing each individual particle
class Particle:
    def __init__(self, position, velocity, mass, color, node=None):
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.set_radius(mass)
        self.color = color
        self.node = node  # Reference to the octree node
    
    def set_radius(self, mass):
        self.radius = mass ** (1/6)

# Generate particles with random mass, position, and velocity
# Create the octree
window_width, window_height = 1024, 768
game_width = 1000
game_boundary = game_width * 2
octree = Octree([window_width / 2, window_height / 2, 50], window_width, max_particles=8, max_depth=10)

def random3():
    return (random.uniform(30, 255), random.uniform(30, 255), random.uniform(30, 255))

# Generate particles with random mass, position, and velocity
num_particles = 200
particles = []
to_delete = set()
for _ in range(num_particles):
    mass = random.uniform(10000, 15000)
    position = [
        random.uniform(-game_width, game_width),
        random.uniform(-game_width, game_width),
        random.uniform(-game_width, game_width)
    ]
    velocity = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
    color = random3()
    particle = Particle(position, velocity, mass, color)
    particles.append(particle)
    octree.insert(particle)

# Function to calculate the gravitational force between two particles
def calculate_internal_force(particle, other_particle):
    # Calculate the distance between the particles
    distance = math.sqrt(sum([(p1 - p2) ** 2 for p1, p2 in zip(particle.position, other_particle.position)]))
    direction = [(p2 - p1) / distance for p1, p2 in zip(particle.position, other_particle.position)]
#     print("internal")
    
    return calculate_force(particle.mass, other_particle.mass, distance, direction)
    
def calculate_force(mass1, mass2, distance, direction):
    G = 6.67430e-11  # Gravitational constant
    c = 1 # 299792458 # speed of light
    λ = 1e-19 # cosmological constant
    # Calculate the gravitational force between the particles
    gravitational_force = (G * mass1 * mass2) / (distance ** 2)
    
    # Calculate the modified force term based on MOND
    mond_force = (λ * mass1 * (c ** 2) * distance) / 3
    
    # Calculate the total force
    total_force = gravitational_force - mond_force
#     print(str(gravitational_force) + " " + str(mond_force) + " " + str(total_force))
    
    # Calculate the force vector
    force_vector = [p * total_force for p in direction]
    
    return force_vector

def calculate_external_force(particle):
    scaled = 1000 # sanity scale for now
    particles_mass = sum(particle.mass for particle in particles) / scaled
    surface_density = particles_mass / game_boundary ** 3
    outer_radius = 2 * game_boundary  # Adjust the outer radius as desired
    external_particle_mass = surface_density * (outer_radius ** 3 - game_boundary ** 3)
    
    position = particle.position
    distance_from_boundary = game_boundary - math.sqrt(sum(p ** 2 for p in position))
    # Calculate the direction of the external force
    direction = [-p / distance_from_boundary for p in position]

    # Calculate the external force vector
    external_force_vector = calculate_force(particle.mass, external_particle_mass, distance_from_boundary, direction)
    
    return external_force_vector

# Function to handle collision between two particles
def handle_collision(particle1, particle2, octtree to_delete):
    # Generate a random number to determine the collision outcome
    outcome = random.random()
    
    if outcome < 0.3:  # 30% chance to bounce
        # Calculate the normalized collision normal vector
        collision_normal = [(p1 - p2) / distance for p1, p2 in zip(particle.position, other_particle.position)]

        # Calculate the relative velocity
        relative_velocity = [v1 - v2 for v1, v2 in zip(particle.velocity, other_particle.velocity)]

        # Calculate the impulse
        impulse = sum([(v1 - v2) * n for v1, v2, n in zip(particle.velocity, other_particle.velocity, collision_normal)])
        impulse /= (1 / particle.mass + 1 / other_particle.mass)

        # Update velocities based on impulse and mass
        particle.velocity = [v - impulse / particle.mass * n for v, n in zip(particle.velocity, collision_normal)]
        other_particle.velocity = [v + impulse / other_particle.mass * n for v, n in zip(other_particle.velocity, collision_normal)]
    elif outcome < 0.6:  # 30% chance to merge
        total_mass = particle1.mass + particle2.mass
        particle1.position = [(p1 * particle1.mass + p2 * particle2.mass) / total_mass for p1, p2 in
                              zip(particle1.position, particle2.position)]
        particle1.velocity = [(v1 * particle1.mass + v2 * particle2.mass) / total_mass for v1, v2 in
                              zip(particle1.velocity, particle2.velocity)]
        particle1.mass = total_mass
        to_delete.add(particle2)
    else:  # 40% chance to split
        new_mass = particle1.mass / 2
        new_position = [p + random.uniform(-1, 1) for p in particle1.position]
        new_velocity = [v + random.uniform(-1, 1) for v in particle1.velocity]
        
        particle1.mass = new_mass
        particle1.velocity = new_velocity
        
        new_particle = Particle(new_position, new_velocity, new_mass, random3())
        octtree.insert(new_particle)

# Define some colors
white = (255, 255, 255)
black = (0, 0, 0)

# Game loop
# Initialize Pygame and create a window
pygame.init()
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Gravity Simulation")

# Define the camera position and velocity
camera_position = [window_width / 2, window_height / 2, 50]
camera_velocity = [0, 0, 0]
camera_rotation_x = 0
move_speed = 10
rotation_speed = 1

running = True
while running:
    # Handle events and user input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Get the state of all keys
    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        camera_velocity[0] = 1 * move_speed  # Move camera up
    elif keys[pygame.K_DOWN]:
        camera_velocity[0] = -1  * move_speed # Move camera down
    else:
        camera_velocity[0] = 0

    if keys[pygame.K_LEFT]:
        camera_velocity[1] = 1 * move_speed # Move camera left
    elif keys[pygame.K_RIGHT]:
        camera_velocity[1] = -1  * move_speed # Move camera right
    else:
        camera_velocity[1] = 0

    if keys[pygame.K_a]:
        camera_rotation_x -= 0.1 * rotation_speed  # Rotate camera counter-clockwise
    elif keys[pygame.K_d]:
        camera_rotation_x += 0.1 * rotation_speed # Rotate camera clockwise

    # Clear the window
    window.fill(black)

    # Calculate gravitational forces and update particle velocities
    for i in range(len(octtree)):
        particle1 = octtree[i]
        # Calculate gravitational forces between particle1 and other particles
        for j in range(i + 1, len(octtree)):
            particle2 = octtree[j]
            force = calculate_internal_force(particle1, particle2)
            acceleration1 = [f / particle1.mass for f in force]
            acceleration2 = [-a for a in acceleration1]
            particle1.velocity = [v + a for v, a in zip(particle1.velocity, acceleration1)]
            particle2.velocity = [v + a for v, a in zip(particle2.velocity, acceleration2)]
        force = calculate_external_force(particle1)
        acceleration1 = [f / particle1.mass for f in force]
        particle1.velocity = [v + a for v, a in zip(particle1.velocity, acceleration1)]

    # Update particle positions and handle collisions
    for particle in octtree:
        particle.position = [p + v for p, v in zip(particle.position, particle.velocity)]

    # Check for collisions with the boundaries of the window
    for i in range(3):
        for particle in octtree:
            if particle.position[i] < -game_boundary or particle.position[i] > game_boundary:
                particle.velocity[i] *= -1

    for particle in octtree:
        # Query the octree for nearby particles
        nearby_particles = octree.query(particle.position)

        # Handle collisions between particles
        for other_particle in nearby_particles:
            if particle != other_particle:
                distance = math.sqrt(sum([(p1 - p2) ** 2 for p1, p2 in zip(particle.position, other_particle.position)]))
                if distance < particle.mass + other_particle.mass:
                    handle_collision(particle, other_particle, octtree, to_delete)
    
    print(to_delete)
    for p in to_delete:
        octtree.delete(p)
    to_delete.clear()

    # Update the octree by reinserting the particle
    octree.insert(particle, particle.node)

    # Update the camera position
    camera_position = [pos + vel for pos, vel in zip(camera_position, camera_velocity)]

    # Draw particles with respect to the camera position and rotation
    for particle in particles:
        # Calculate the particle's position relative to the camera
        relative_position = [p - c for p, c in zip(particle.position, camera_position)]
        # Rotate the particle's position around the camera
        rotated_position = [
            relative_position[1] * math.cos(camera_rotation_x) - relative_position[2] * math.sin(camera_rotation_x),
            relative_position[0],
            relative_position[1] * math.sin(camera_rotation_x) + relative_position[2] * math.cos(camera_rotation_x),
        ]
        # Rescale and shift the particle's position for drawing
        scaled_position = [int(rp / 10) + window_width // 2 for rp in rotated_position[:2]]
        pygame.draw.circle(window, particle.color, scaled_position, int(particle.radius))

    # Update the display
    pygame.display.flip()

# Quit the simulation
pygame.quit()
