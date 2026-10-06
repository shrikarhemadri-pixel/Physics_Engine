import pygame
import numpy as np
import fluidsim
from constants import *

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fluid Simulation")
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
    slow  → blue  (0, 50, 255)
    mid   → cyan  (0, 255, 255)
    fast  → white (255, 255, 255)
    """
    # normalize speeds to 0-1
    normalized = np.clip((speeds - min_speed) / (max_speed - min_speed), 0.0, 1.0)

    colors = np.zeros((len(speeds), 3), dtype=np.uint8)

    # slow → mid: blue to cyan (increase green)
    slow_mask = normalized < 0.5
    t_slow = normalized[slow_mask] * 2  # 0 to 1
    colors[slow_mask, 0] = 0
    colors[slow_mask, 1] = (t_slow * 255).astype(np.uint8)
    colors[slow_mask, 2] = 255

    # mid → fast: cyan to white (increase red)
    fast_mask = ~slow_mask
    t_fast = (normalized[fast_mask] - 0.5) * 2  # 0 to 1
    colors[fast_mask, 0] = (t_fast * 255).astype(np.uint8)
    colors[fast_mask, 1] = 255
    colors[fast_mask, 2] = 255

    return colors

# --- Draw ---
def draw_particles(positions, velocities):
    screen_positions = to_screen_coords(positions)

    # compute speeds
    speeds = np.linalg.norm(velocities, axis=1)
    max_speed = max(speeds.max(), 1.0)  # avoid division by zero
    colors = speed_to_color(speeds, min_speed=0.0, max_speed=max_speed)

    for i in range(len(screen_positions)):
        pos = screen_positions[i]
        color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))

        # draw glow effect
        glow_surface = pygame.Surface((DRAW_RADIUS*4, DRAW_RADIUS*4), pygame.SRCALPHA)
        glow_color = (color[0], color[1], color[2], 40)
        pygame.draw.circle(glow_surface, glow_color,
                         (DRAW_RADIUS*2, DRAW_RADIUS*2), DRAW_RADIUS*2)
        screen.blit(glow_surface, (pos[0] - DRAW_RADIUS*2, pos[1] - DRAW_RADIUS*2))

        # draw core particle
        pygame.draw.circle(screen, color, (pos[0], pos[1]), DRAW_RADIUS)

# --- Mouse force ---
def apply_mouse_force(mouse_screen_pos, attract=True):
    sim_x = mouse_screen_pos[0] - (WIDTH // 2)
    sim_y = (HEIGHT // 2) - mouse_screen_pos[1]
    mouse_sim_pos = np.array([sim_x, sim_y], dtype=float)

    diff = mouse_sim_pos - fluidsim.positions
    dists = np.linalg.norm(diff, axis=1)
    mask = (dists < MOUSE_RADIUS) & (dists > 0)

    if not np.any(mask):
        return

    dirs = diff[mask] / dists[mask, np.newaxis]
    strength = (1 - dists[mask] / MOUSE_RADIUS) * MOUSE_STRENGTH

    if attract:
        fluidsim.velocities[mask] += dirs * strength[:, np.newaxis] * DT
    else:
        fluidsim.velocities[mask] -= dirs * strength[:, np.newaxis] * DT

# --- Init ---
fluidsim.create_grid(NUM_PARTICLES, RADIUS, fluidsim.positions)

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
                fluidsim.create_grid(NUM_PARTICLES, RADIUS, fluidsim.positions)
                fluidsim.velocities[:] = 0

    fluidsim.simulation_step()

    if left_click:
        apply_mouse_force(mouse_screen_pos, attract=True)
    if right_click:
        apply_mouse_force(mouse_screen_pos, attract=False)

    screen.fill((5, 5, 15))
    draw_particles(fluidsim.positions, fluidsim.velocities)

    # mouse radius indicator
    if left_click or right_click:
        color = (0, 255, 100) if left_click else (255, 50, 50)
        pygame.draw.circle(screen, color, mouse_screen_pos,
                         int(MOUSE_RADIUS), 2)

    # HUD
    fps = clock.get_fps()
    speeds = np.linalg.norm(fluidsim.velocities, axis=1)
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