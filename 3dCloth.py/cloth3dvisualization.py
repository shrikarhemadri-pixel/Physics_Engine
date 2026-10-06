import pygame
import numpy as np
import math
import cloth3d
from cloth3dconstants import *

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cloth Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont("monospace", 20)

DRAW_RADIUS = max(int(RADIUS * 3), 3)

# --- Camera orientation (radians). Adjust live with arrow keys. ---
camera_yaw = 0.0
camera_pitch = 0.0

# --- Rotate world-space points into camera space ---
def rotate_to_camera_space(positions, yaw, pitch):
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # yaw: rotate around the vertical (Y) axis
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    x1 = x * cos_y + z * sin_y
    z1 = -x * sin_y + z * cos_y
    y1 = y

    # pitch: rotate around the horizontal (X) axis
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    y2 = y1 * cos_p - z1 * sin_p
    z2 = y1 * sin_p + z1 * cos_p

    return np.stack([x1, y2, z2], axis=1)

# --- Inverse rotation: camera space back to world space (needed to unproject the mouse) ---
def rotate_to_world_space(point, yaw, pitch):
    x, y, z = point

    # undo pitch first (reverse order of the forward rotation)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    y1 = y * cos_p + z * sin_p
    z1 = -y * sin_p + z * cos_p

    # undo yaw
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    x1 = x * cos_y - z1 * sin_y
    z2 = x * sin_y + z1 * cos_y

    return np.array([x1, y1, z2])

# --- Same inverse rotation, but for an (N,3) batch of points at once ---
def rotate_to_world_space_batch(points, yaw, pitch):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    y1 = y * cos_p + z * sin_p
    z1 = -y * sin_p + z * cos_p

    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    x1 = x * cos_y - z1 * sin_y
    z2 = x * sin_y + z1 * cos_y

    return np.stack([x1, y1, z2], axis=1)

# --- Coordinate conversion ---
def to_screen_coords(positions):
    # perspective projection: things closer to the camera (larger z) scale up,
    # things farther away (smaller z) scale down toward a vanishing point
    depth = CAMERA_DISTANCE - positions[:, 2]
    depth = np.maximum(depth, 0.01)  # guard against divide-by-zero / behind-camera particles
    scale = CAMERA_DISTANCE / depth

    screen_pos = np.zeros((len(positions), 2))
    screen_pos[:, 0] = positions[:, 0] * scale + (WIDTH // 2)
    screen_pos[:, 1] = (HEIGHT // 2) - positions[:, 1] * scale
    return screen_pos.astype(int), scale

# --- Color based on speed ---
def speed_to_color(speeds, min_speed=0.0, max_speed=20.0):
    """
    Maps speed to color:
    slow  -> blue  (0, 50, 255)
    mid   -> cyan  (0, 255, 255)
    fast  -> white (255, 255, 255)
    """
    normalized = np.clip((speeds - min_speed) / (max_speed - min_speed), 0.0, 1.0)

    colors = np.zeros((len(speeds), 3), dtype=np.uint8)

    slow_mask = normalized < 0.5
    t_slow = normalized[slow_mask] * 2
    colors[slow_mask, 0] = 0
    colors[slow_mask, 1] = (t_slow * 255).astype(np.uint8)
    colors[slow_mask, 2] = 255

    fast_mask = ~slow_mask
    t_fast = (normalized[fast_mask] - 0.5) * 2
    colors[fast_mask, 0] = (t_fast * 255).astype(np.uint8)
    colors[fast_mask, 1] = 255
    colors[fast_mask, 2] = 255

    return colors

# --- Draw ---
def draw_cloth(positions, velocities, edges):
    camera_positions = rotate_to_camera_space(positions, camera_yaw, camera_pitch)
    screen_positions, scale = to_screen_coords(camera_positions)

    # depth sort: draw farthest particles/edges first so nearer ones occlude correctly
    draw_order = np.argsort(camera_positions[:, 2])

    # edges, ordered by the average depth of their two endpoints
    edge_depth = [(camera_positions[i, 2] + camera_positions[j, 2]) / 2 for (i, j) in edges]
    edges_sorted = [e for _, e in sorted(zip(edge_depth, edges))]

    for (i, j) in edges_sorted:
        pygame.draw.line(
            screen, (80, 90, 110),
            screen_positions[i], screen_positions[j], 1
        )

    speeds = np.linalg.norm(velocities, axis=1)
    max_speed = max(speeds.max(), 1.0)
    colors = speed_to_color(speeds, min_speed=0.0, max_speed=max_speed)

    for i in draw_order:
        pos = screen_positions[i]
        color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))

        if i == PIN:
            color = (255, 80, 80)

        radius = max(int(DRAW_RADIUS * scale[i]), 1)

        # glow effect
        glow_surface = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        glow_color = (color[0], color[1], color[2], 40)
        pygame.draw.circle(glow_surface, glow_color,
                            (radius * 2, radius * 2), radius * 2)
        screen.blit(glow_surface, (pos[0] - radius * 2, pos[1] - radius * 2))

        # core particle
        pygame.draw.circle(screen, color, (pos[0], pos[1]), radius)

# --- Drag interaction: grab the nearest particle (in screen pixels) and latch
#     its position to the mouse while held. No force field, no distance-in-3D
#     guessing -- this can't "miss" the way a force field can.
GRAB_RADIUS_PIXELS = 25  # move this into cloth3dconstants.py if you want it tunable there

dragged_particle = None

def find_nearest_particle(mouse_screen_pos, screen_positions):
    dx = screen_positions[:, 0] - mouse_screen_pos[0]
    dy = screen_positions[:, 1] - mouse_screen_pos[1]
    pixel_dist = np.sqrt(dx * dx + dy * dy)
    nearest = np.argmin(pixel_dist)
    if pixel_dist[nearest] < GRAB_RADIUS_PIXELS:
        return int(nearest)
    return None

def compute_drag_target(index, mouse_screen_pos, world_positions, yaw, pitch):
    # unproject the mouse onto this specific particle's own depth plane, so
    # the target always matches where it's currently rendered, in z as well
    camera_positions = rotate_to_camera_space(world_positions, yaw, pitch)
    depth = max(CAMERA_DISTANCE - camera_positions[index, 2], 0.01)
    scale = CAMERA_DISTANCE / depth

    sim_x = (mouse_screen_pos[0] - (WIDTH // 2)) / scale
    sim_y = ((HEIGHT // 2) - mouse_screen_pos[1]) / scale
    target_camera = np.array([sim_x, sim_y, camera_positions[index, 2]])

    return rotate_to_world_space(target_camera, yaw, pitch)

# --- Init ---
part_per_row, part_per_column = cloth3d.create_grid(
    NUM_PARTICLES, RADIUS, cloth3d.particle_positions, physical_width=None
)
neighbors = cloth3d.find_all_neighbors(NUM_PARTICLES, part_per_row, part_per_column)
rest_lengths = cloth3d.calculate_rest_lengths(neighbors, cloth3d.particle_positions)

# build a deduplicated edge list once, purely for drawing lines between neighbors
edges = set()
for i in range(NUM_PARTICLES):
    for j in neighbors[i]:
        edges.add((min(i, j), max(i, j)))
edges = list(edges)

# --- Main loop ---
running = True
while running:
    mouse_screen_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_r:
                cloth3d.create_grid(NUM_PARTICLES, RADIUS, cloth3d.particle_positions, physical_width=None)
                cloth3d.particle_velocities[:] = 0
                dragged_particle = None
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            camera_positions = rotate_to_camera_space(cloth3d.particle_positions, camera_yaw, camera_pitch)
            screen_positions, _ = to_screen_coords(camera_positions)
            candidate = find_nearest_particle(mouse_screen_pos, screen_positions)
            if candidate is not None and candidate != PIN:
                dragged_particle = candidate
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            dragged_particle = None

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        camera_yaw -= CAMERA_ROTATE_SPEED
    if keys[pygame.K_RIGHT]:
        camera_yaw += CAMERA_ROTATE_SPEED
    if keys[pygame.K_UP]:
        camera_pitch = max(camera_pitch - CAMERA_ROTATE_SPEED, -math.pi / 2 + 0.01)
    if keys[pygame.K_DOWN]:
        camera_pitch = min(camera_pitch + CAMERA_ROTATE_SPEED, math.pi / 2 - 0.01)

    cloth3d.update_loop(neighbors, rest_lengths)

    if dragged_particle is not None:
        target = compute_drag_target(dragged_particle, mouse_screen_pos, cloth3d.particle_positions, camera_yaw, camera_pitch)
        # velocity implied by this frame's motion -- gives a natural "throw" on release
        cloth3d.particle_velocities[dragged_particle] = (target - cloth3d.particle_positions[dragged_particle]) / DT
        cloth3d.particle_positions[dragged_particle] = target

    screen.fill((5, 5, 15))
    draw_cloth(cloth3d.particle_positions, cloth3d.particle_velocities, edges)

    # highlight the grabbed particle
    if dragged_particle is not None:
        pygame.draw.circle(screen, (0, 255, 100), mouse_screen_pos, GRAB_RADIUS_PIXELS, 2)

    # HUD
    fps = clock.get_fps()
    speeds = np.linalg.norm(cloth3d.particle_velocities, axis=1)
    hud = font.render(
        f"FPS: {fps:.1f}  Particles: {NUM_PARTICLES}  "
        f"Max speed: {speeds.max():.2f}  "
        f"LMB: grab & drag  R: reset",
        True, (200, 200, 200)
    )
    screen.blit(hud, (10, 10))

    pygame.display.flip()
    clock.tick()

pygame.quit()