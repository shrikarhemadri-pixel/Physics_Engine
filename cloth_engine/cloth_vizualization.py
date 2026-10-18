import pygame
import numpy as np
import cloth_start
from cloth_constants import *

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cloth Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont("monospace", 20)

DRAW_RADIUS = max(int(RADIUS * 3), 3)

# --- Coordinate conversion ---
def to_screen_coords(positions):
    screen_pos = np.zeros_like(positions)
    screen_pos[:, 0] = positions[:, 0] + (WIDTH // 2)
    screen_pos[:, 1] = (HEIGHT // 2) - positions[:, 1]
    return screen_pos.astype(int)

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
    screen_positions = to_screen_coords(positions)

    # draw structural constraints as lines first, so particles sit on top
    for (i, j) in edges:
        pygame.draw.line(
            screen, (80, 90, 110),
            screen_positions[i], screen_positions[j], 1
        )

    speeds = np.linalg.norm(velocities, axis=1)
    max_speed = max(speeds.max(), 1.0)
    colors = speed_to_color(speeds, min_speed=0.0, max_speed=max_speed)

    for i in range(len(screen_positions)):
        pos = screen_positions[i]
        color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))

        if i == PIN:
            color = (255, 80, 80)

        # glow effect
        glow_surface = pygame.Surface((DRAW_RADIUS * 4, DRAW_RADIUS * 4), pygame.SRCALPHA)
        glow_color = (color[0], color[1], color[2], 40)
        pygame.draw.circle(glow_surface, glow_color,
                            (DRAW_RADIUS * 2, DRAW_RADIUS * 2), DRAW_RADIUS * 2)
        screen.blit(glow_surface, (pos[0] - DRAW_RADIUS * 2, pos[1] - DRAW_RADIUS * 2))

        # core particle
        pygame.draw.circle(screen, color, (pos[0], pos[1]), DRAW_RADIUS)

# --- Mouse force ---
def apply_mouse_force(mouse_screen_pos, attract=True):
    sim_x = mouse_screen_pos[0] - (WIDTH // 2)
    sim_y = (HEIGHT // 2) - mouse_screen_pos[1]
    mouse_sim_pos = np.array([sim_x, sim_y], dtype=float)

    diff = mouse_sim_pos - cloth_start.particle_positions
    dists = np.linalg.norm(diff, axis=1)
    mask = (dists < MOUSE_RADIUS) & (dists > 0)
    mask[PIN] = False  # never yank the pinned particle around
    # mask[PIN2] = False
    # mask[PIN3] = False
    # mask[PIN4] = False
    if not np.any(mask):
        return

    dirs = diff[mask] / dists[mask, np.newaxis]
    strength = (1 - dists[mask] / MOUSE_RADIUS) * MOUSE_STRENGTH

    if attract:
        cloth_start.particle_velocities[mask] += dirs * strength[:, np.newaxis] * DT
    else:
        cloth_start.particle_velocities[mask] -= dirs * strength[:, np.newaxis] * DT

# --- Init ---
part_per_row, part_per_column = cloth_start.create_grid(
    NUM_PARTICLES, RADIUS, cloth_start.particle_positions
)
neighbors = cloth_start.find_all_neighbors(NUM_PARTICLES, part_per_row, part_per_column)
rest_lengths = cloth_start.calculate_rest_lengths(neighbors, cloth_start.particle_positions)

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
    left_click, _, right_click = pygame.mouse.get_pressed()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_r:
                cloth_start.create_grid(NUM_PARTICLES, RADIUS, cloth_start.particle_positions)
                cloth_start.particle_velocities[:] = 0

    cloth_start.update_loop(neighbors, rest_lengths)

    if left_click:
        apply_mouse_force(mouse_screen_pos, attract=True)
    if right_click:
        apply_mouse_force(mouse_screen_pos, attract=False)

    screen.fill((5, 5, 15))
    draw_cloth(cloth_start.particle_positions, cloth_start.particle_velocities, edges)

    # mouse radius indicator
    if left_click or right_click:
        color = (0, 255, 100) if left_click else (255, 50, 50)
        pygame.draw.circle(screen, color, mouse_screen_pos, int(MOUSE_RADIUS), 2)

    # HUD
    fps = clock.get_fps()
    speeds = np.linalg.norm(cloth_start.particle_velocities, axis=1)
    hud = font.render(
        f"FPS: {fps:.1f}  Particles: {NUM_PARTICLES}  "
        f"Max speed: {speeds.max():.2f}  "
        f"LMB: attract  RMB: repel  R: reset",
        True, (200, 200, 200)
    )
    screen.blit(hud, (10, 10))

    pygame.display.flip()
    clock.tick()

pygame.quit()