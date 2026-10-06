import os
import numpy as np
import pygame
import fluidsim
import fluidflow
from constants import *

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SPH Fluid Simulation (3D)" if DIM == 3 else "SPH Fluid Simulation (2D)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("monospace", 18)

# ---- orbit camera (left-drag: rotate, wheel: zoom, SPACE: pause, R: reset, ESC: quit) ----
yaw, pitch = 0.6, 0.35
cam_dist = 420.0
FOCAL = 1300.0
paused = False
color_max = 1.0                     # smoothed colour scale so colours don't flicker


def rotation_matrix(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return rx @ ry


def project(points):
    """Sim coords (N, DIM) -> screen (N, 2) ints, depth (N,), perspective scale (N,)."""
    p = np.zeros((len(points), 3))
    p[:, :DIM] = points
    p = p @ rotation_matrix(yaw, pitch).T
    depth = np.maximum(cam_dist - p[:, 2], 1.0)
    scale = FOCAL / depth
    sx = WIDTH / 2 + p[:, 0] * scale
    sy = HEIGHT / 2 - p[:, 1] * scale
    return np.stack([sx, sy], axis=1), depth, scale


def speed_to_color(speeds, max_speed):
    """slow -> blue, mid -> cyan, fast -> white"""
    t = np.clip(speeds / max_speed, 0.0, 1.0)
    colors = np.zeros((len(speeds), 3), dtype=np.uint8)
    slow = t < 0.5
    colors[slow, 1] = (t[slow] * 2 * 255).astype(np.uint8)
    colors[slow, 2] = 255
    colors[~slow, 0] = ((t[~slow] - 0.5) * 2 * 255).astype(np.uint8)
    colors[~slow, 1] = 255
    colors[~slow, 2] = 255
    return colors


def draw_wireframe():
    col = (70, 80, 110)
    if SCENE == "pipe":
        xs = np.linspace(-XLIMITS, XLIMITS, 13)
        if DIM == 3:
            ang = np.linspace(0, 2 * np.pi, 33)
            for x in xs:                                            # rings
                R = float(fluidflow.wall_radius(x))
                ring = np.stack([np.full_like(ang, x), R * np.cos(ang), R * np.sin(ang)], axis=1)
                pygame.draw.lines(screen, col, False, project(ring)[0].tolist(), 1)
            xl = np.linspace(-XLIMITS, XLIMITS, 40)
            for a in np.linspace(0, 2 * np.pi, 9)[:-1]:             # long. lines
                R = fluidflow.wall_radius(xl)
                line = np.stack([xl, R * np.cos(a), R * np.sin(a)], axis=1)
                pygame.draw.lines(screen, col, False, project(line)[0].tolist(), 1)
        else:
            xl = np.linspace(-XLIMITS, XLIMITS, 40)
            R = fluidflow.wall_radius(xl)
            for sgn in (-1, 1):
                pygame.draw.lines(screen, col, False, project(np.stack([xl, sgn * R], axis=1))[0].tolist(), 1)
    else:
        corners = np.array(np.meshgrid(*[[-e, e] for e in HALF_EXTENTS], indexing="ij")).reshape(DIM, -1).T
        sc = project(corners)[0]
        for a in range(len(corners)):
            for b in range(a + 1, len(corners)):
                if np.count_nonzero(corners[a] != corners[b]) == 1:      # edges only
                    pygame.draw.line(screen, col, sc[a], sc[b], 1)


def draw_particles(positions, velocities):
    global color_max
    sp, depth, scale = project(positions)
    speeds = np.linalg.norm(velocities, axis=1)
    color_max = max(color_max * 0.995, float(speeds.max()), 0.1)
    colors = speed_to_color(speeds, color_max)
    radii = np.maximum((PARTICLE_SPACING * 0.4 * scale).astype(int), 1)
    sp = sp.astype(int)
    for i in np.argsort(-depth):                                         # far -> near
        pygame.draw.circle(screen, (int(colors[i][0]), int(colors[i][1]), int(colors[i][2])),
                           (sp[i][0], sp[i][1]), int(radii[i]))


def reset():
    global color_max
    fluidsim.create_grid()
    color_max = 1.0


reset()
MAX_FRAMES = int(os.environ.get("SPH_MAX_FRAMES", 0))     # >0: run headless for N frames (testing)
frame, running, dragging = 0, True, False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                reset()
            elif event.key == pygame.K_SPACE:
                paused = not paused
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            dragging = False
        elif event.type == pygame.MOUSEMOTION and dragging and DIM == 3:
            yaw += event.rel[0] * 0.01
            pitch = float(np.clip(pitch + event.rel[1] * 0.01, -1.5, 1.5))
        elif event.type == pygame.MOUSEWHEEL:
            cam_dist = float(np.clip(cam_dist * (0.9 ** event.y), 150, 1500))

    if not paused:
        fluidsim.simulation_step()

    screen.fill((5, 5, 15))
    draw_wireframe()
    draw_particles(fluidsim.positions, fluidsim.velocities)

    speeds = np.linalg.norm(fluidsim.velocities, axis=1)
    hud = font.render(
        f"{DIM}D  FPS: {clock.get_fps():.0f}  N: {len(fluidsim.positions)}  mean|v|: {speeds.mean():.2f}  "
        f"drag: rotate  wheel: zoom  SPACE: pause  R: reset", True, (200, 200, 200))
    screen.blit(hud, (10, 10))
    pygame.display.flip()
    clock.tick()

    frame += 1
    if MAX_FRAMES and frame >= MAX_FRAMES:
        if os.environ.get("SPH_SCREENSHOT"):
            pygame.image.save(screen, os.environ["SPH_SCREENSHOT"])
        running = False

pygame.quit()