"""
Cloth + rope simulation with SPOOK-style constraints.

Python conversion of the supplied C++/OpenGL program.

Dependencies:
    python -m pip install numpy PyOpenGL PyOpenGL_accelerate

Linux may also require FreeGLUT, for example:
    sudo apt install freeglut3-dev

Controls:
    Left-drag              Rotate the camera
    Middle-drag            Zoom
    Drag a cloth vertex    Move that vertex
    Esc                    Exit
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass

import numpy as np

try:
    from OpenGL.GL import (
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_COMPONENT,
        GL_DEPTH_TEST,
        GL_DOUBLE,
        GL_FLOAT,
        GL_FRONT_AND_BACK,
        GL_LINES,
        GL_MODELVIEW,
        GL_MODELVIEW_MATRIX,
        GL_POINTS,
        GL_PROJECTION,
        GL_PROJECTION_MATRIX,
        GL_TRIANGLES,
        GL_UNSIGNED_SHORT,
        GL_VIEWPORT,
        GL_LINE,
        glBegin,
        glClear,
        glColor3f,
        glEnable,
        glEnd,
        glGetDoublev,
        glGetIntegerv,
        glLoadIdentity,
        glMatrixMode,
        glMultMatrixf,
        glPointSize,
        glPolygonMode,
        glPopMatrix,
        glPushMatrix,
        glReadPixels,
        glRotatef,
        glTranslatef,
        glVertex3f,
        glViewport,
    )
    from OpenGL.GLU import gluPerspective, gluUnProject
    from OpenGL.GLUT import (
        GLUT_CURSOR_INHERIT,
        GLUT_CURSOR_LEFT_RIGHT,
        GLUT_CURSOR_UP_DOWN,
        GLUT_DEPTH,
        GLUT_DOUBLE,
        GLUT_DOWN,
        GLUT_ELAPSED_TIME,
        GLUT_LEFT_BUTTON,
        GLUT_MIDDLE_BUTTON,
        GLUT_RGBA,
        GLUT_UP,
        glutCloseFunc,
        glutCreateWindow,
        glutDisplayFunc,
        glutGet,
        glutIdleFunc,
        glutInit,
        glutInitDisplayMode,
        glutInitWindowSize,
        glutKeyboardFunc,
        glutMainLoop,
        glutMotionFunc,
        glutMouseFunc,
        glutPostRedisplay,
        glutReshapeFunc,
        glutSetCursor,
        glutSetWindowTitle,
        glutSolidCube,
        glutSwapBuffers,
        glutWireCube,
    )
    from OpenGL.error import NullFunctionError
except ImportError as exc:
    raise SystemExit(
        "PyOpenGL is required. Install it with:\n"
        "  python -m pip install numpy PyOpenGL PyOpenGL_accelerate"
    ) from exc


# ---------------------------------------------------------------------------
# Window and camera settings
# ---------------------------------------------------------------------------

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 1024
WINDOW_TITLE = "Cloth + Rope (SPOOK) Simulation"
GRID_SIZE = 10

old_mouse_x = 0
old_mouse_y = 0
rotation_x = 15.0
rotation_y = 0.0
mouse_state = 1  # 0 = zoom, 1 = rotate
camera_distance = -23.0

window_width = WINDOW_WIDTH
window_height = WINDOW_HEIGHT


# ---------------------------------------------------------------------------
# Cloth parameters
# ---------------------------------------------------------------------------

CLOTH_NUM_X = 20
CLOTH_NUM_Y = 20
CLOTH_NUM_POINTS = (CLOTH_NUM_X + 1) * (CLOTH_NUM_Y + 1)

CLOTH_SIZE = 4.0
HALF_CLOTH_SIZE = CLOTH_SIZE / 2.0

STRUCTURAL_SPRING = 0
SHEAR_SPRING = 1
BEND_SPRING = 2

DEFAULT_DAMPING = -0.0125
KS_STRUCT, KD_STRUCT = 0.5, -0.25
KS_SHEAR, KD_SHEAR = 0.5, -0.25
KS_BEND, KD_BEND = 0.85, -0.25

GRAVITY = np.array([0.0, -0.00981, 0.0], dtype=np.float64)
MASS = 0.5


@dataclass(slots=True)
class Spring:
    p1: int
    p2: int
    rest_length: float
    ks: float
    kd: float
    spring_type: int


cloth_indices = np.empty(0, dtype=np.uint16)
cloth_springs: list[Spring] = []

cloth_positions = np.zeros((CLOTH_NUM_POINTS, 3), dtype=np.float64)
cloth_velocities = np.zeros_like(cloth_positions)
cloth_forces = np.zeros_like(cloth_positions)


# ---------------------------------------------------------------------------
# Rope and SPOOK distance-constraint data
# ---------------------------------------------------------------------------

ROPE_PARTICLES = 4
ROPE_CONSTRAINTS = 3
ROPE_PARTICLE_MASS = 0.5
ROPE_RELAXATION_TIME = 1.3
ROPE_STIFFNESS = 0.8
ROPE_GAUSS_SEIDEL_ITERATIONS = 10
ROPE_ANCHOR_INDEX = 0

rope_positions = np.zeros((ROPE_PARTICLES, 3), dtype=np.float64)
rope_velocities = np.zeros((ROPE_PARTICLES, 3), dtype=np.float64)
rope_forces = np.zeros_like(rope_velocities)
rope_inverse_masses = np.full(
    ROPE_PARTICLES, 1.0 / ROPE_PARTICLE_MASS, dtype=np.float64
)

# One scalar distance constraint and one scalar Lagrange multiplier for each
# neighbouring particle pair.
rope_rest_lengths = np.zeros(ROPE_CONSTRAINTS, dtype=np.float64)
rope_constraint_violations = np.zeros(
    ROPE_CONSTRAINTS, dtype=np.float64
)
rope_constraint_multipliers = np.zeros(
    ROPE_CONSTRAINTS, dtype=np.float64
)
rope_constraint_rhs = np.zeros(ROPE_CONSTRAINTS, dtype=np.float64)

# Block Jacobian G. Its shape is constraints x particles x xyz. Each nonzero
# entry is a complete 3D unit vector, not a sum of its components.
rope_constraint_jacobian = np.zeros(
    (ROPE_CONSTRAINTS, ROPE_PARTICLES, 3), dtype=np.float64
)


# ---------------------------------------------------------------------------
# Time stepping and OpenGL state
# ---------------------------------------------------------------------------

# Assignment 3 specifies h = 0.02 seconds. The same fixed step is used by the
# integrator and all SPOOK terms so units cannot drift apart.
TIME_STEP = 0.02
accumulator = 0.0
last_physics_time = time.perf_counter()
last_render_time = time.perf_counter()

selected_cloth_index = -1

viewport = np.zeros(4, dtype=np.int32)
model_view_matrix = np.identity(4, dtype=np.float64)
projection_matrix = np.identity(4, dtype=np.float64)

WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
camera_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
view_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)

fps_start_time = time.perf_counter()
total_frames = 0
fps = 0.0


# ---------------------------------------------------------------------------
# Ellipsoid collision data
# ---------------------------------------------------------------------------

ELLIPSOID_RADIUS = 1.0
ELLIPSOID_OBJECT_RADIUS = 1.0
ELLIPSOID_CENTER = np.zeros(3, dtype=np.float64)

ellipsoid_transform = np.identity(4, dtype=np.float64)
inverse_ellipsoid_transform = np.identity(4, dtype=np.float64)


# ---------------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------------


def translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.identity(4, dtype=np.float64)
    matrix[:3, 3] = (x, y, z)
    return matrix


def scale_matrix(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.identity(4, dtype=np.float64)
    matrix[0, 0] = x
    matrix[1, 1] = y
    matrix[2, 2] = z
    return matrix


def rotation_x_matrix(angle_degrees: float) -> np.ndarray:
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cosine, -sine, 0.0],
            [0.0, sine, cosine, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def as_opengl_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return a float32 column-major representation for glMultMatrixf."""
    return np.ascontiguousarray(matrix.T, dtype=np.float32)


# ---------------------------------------------------------------------------
# Simulation initialization
# ---------------------------------------------------------------------------


def add_spring(
    index_a: int,
    index_b: int,
    ks: float,
    kd: float,
    spring_type: int,
) -> None:
    delta = cloth_positions[index_a] - cloth_positions[index_b]
    cloth_springs.append(
        Spring(
            p1=index_a,
            p2=index_b,
            rest_length=float(np.linalg.norm(delta)),
            ks=ks,
            kd=kd,
            spring_type=spring_type,
        )
    )


def initialize_cloth() -> None:
    global cloth_indices

    vertices_per_row = CLOTH_NUM_X + 1
    vertices_per_column = CLOTH_NUM_Y + 1

    count = 0
    for row in range(vertices_per_column):
        for column in range(vertices_per_row):
            cloth_positions[count] = (
                (column / (vertices_per_row - 1) * 2.0 - 1.0)
                * HALF_CLOTH_SIZE,
                CLOTH_SIZE + 1.0,
                row / (vertices_per_column - 1) * CLOTH_SIZE,
            )
            count += 1

    indices: list[int] = []
    for row in range(CLOTH_NUM_Y):
        for column in range(CLOTH_NUM_X):
            i0 = row * (CLOTH_NUM_X + 1) + column
            i1 = i0 + 1
            i2 = i0 + CLOTH_NUM_X + 1
            i3 = i2 + 1

            if (column + row) % 2:
                indices.extend((i0, i2, i1, i1, i2, i3))
            else:
                indices.extend((i0, i2, i3, i0, i3, i1))

    cloth_indices = np.asarray(indices, dtype=np.uint16)

    cloth_springs.clear()

    # Horizontal structural springs.
    for row in range(vertices_per_column):
        for column in range(vertices_per_row - 1):
            add_spring(
                row * vertices_per_row + column,
                row * vertices_per_row + column + 1,
                KS_STRUCT,
                KD_STRUCT,
                STRUCTURAL_SPRING,
            )

    # Vertical structural springs.
    for column in range(vertices_per_row):
        for row in range(vertices_per_column - 1):
            add_spring(
                row * vertices_per_row + column,
                (row + 1) * vertices_per_row + column,
                KS_STRUCT,
                KD_STRUCT,
                STRUCTURAL_SPRING,
            )

    # Diagonal shear springs.
    for row in range(vertices_per_column - 1):
        for column in range(vertices_per_row - 1):
            add_spring(
                row * vertices_per_row + column,
                (row + 1) * vertices_per_row + column + 1,
                KS_SHEAR,
                KD_SHEAR,
                SHEAR_SPRING,
            )
            add_spring(
                (row + 1) * vertices_per_row + column,
                row * vertices_per_row + column + 1,
                KS_SHEAR,
                KD_SHEAR,
                SHEAR_SPRING,
            )

    # Bend springs were commented out in the original C++ source.


def initialize_rope() -> None:
    rope_positions[:] = np.array(
        [
            [-1.0, 10.0, 7.0],
            [0.0, 9.5, 4.0],
            [-1.0, 9.0, 5.0],
            [-1.0, 8.5, 6.0],
        ],
        dtype=np.float64,
    )

    # Start from rest. Nonzero values can be assigned here to make the rope
    # swing, but positive Y velocities will physically move particles upward.
    rope_velocities.fill(0.0)
    rope_forces.fill(0.0)
    rope_constraint_multipliers.fill(0.0)

    rope_inverse_masses.fill(1.0 / ROPE_PARTICLE_MASS)
    rope_inverse_masses[ROPE_ANCHOR_INDEX] = 0.0

    for constraint in range(ROPE_CONSTRAINTS):
        segment = (
            rope_positions[constraint]
            - rope_positions[constraint + 1]
        )
        rope_rest_lengths[constraint] = float(np.linalg.norm(segment))

    update_rope_constraint_geometry()


def update_rope_constraint_geometry() -> None:
    """Recompute g(x) and the full 3D block Jacobian G."""
    rope_constraint_jacobian.fill(0.0)

    for constraint in range(ROPE_CONSTRAINTS):
        first_particle = constraint
        second_particle = constraint + 1

        segment = (
            rope_positions[first_particle]
            - rope_positions[second_particle]
        )
        current_length = float(np.linalg.norm(segment))
        if current_length <= 1e-12:
            raise RuntimeError(
                f"Rope segment {constraint} has zero length."
            )

        unit_direction = segment / current_length
        rope_constraint_violations[constraint] = (
            current_length - rope_rest_lengths[constraint]
        )
        rope_constraint_jacobian[
            constraint, first_particle
        ] = unit_direction
        rope_constraint_jacobian[
            constraint, second_particle
        ] = -unit_direction


def initialize_ellipsoid() -> None:
    global ellipsoid_transform, inverse_ellipsoid_transform

    transform = (
        translation_matrix(0.0, 2.0, 0.0)
        @ rotation_x_matrix(85.0)
        @ scale_matrix(
            ELLIPSOID_RADIUS,
            ELLIPSOID_RADIUS,
            ELLIPSOID_RADIUS / 2.0,
        )
    )

    # These assignments preserve the naming and behavior of the C++ source.
    ellipsoid_transform = np.linalg.inv(transform)
    inverse_ellipsoid_transform = np.linalg.inv(ellipsoid_transform)


def initialize_gl() -> None:
    global last_physics_time, last_render_time, fps_start_time

    glEnable(GL_DEPTH_TEST)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glPointSize(7.0)

    initialize_cloth()
    initialize_rope()
    initialize_ellipsoid()

    now = time.perf_counter()
    last_physics_time = now
    last_render_time = now
    fps_start_time = now


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


def compute_cloth_forces() -> None:
    cloth_forces[:] = DEFAULT_DAMPING * cloth_velocities

    # The two top corners are fixed.
    movable = np.ones(CLOTH_NUM_POINTS, dtype=bool)
    movable[0] = False
    movable[CLOTH_NUM_X] = False
    cloth_forces[movable] += GRAVITY

    for spring in cloth_springs:
        p1 = cloth_positions[spring.p1]
        p2 = cloth_positions[spring.p2]
        v1 = cloth_velocities[spring.p1]
        v2 = cloth_velocities[spring.p2]

        delta_position = p1 - p2
        distance = float(np.linalg.norm(delta_position))
        if distance <= 1e-12:
            continue

        delta_velocity = v1 - v2
        left_term = -spring.ks * (distance - spring.rest_length)
        right_term = spring.kd * (
            float(np.dot(delta_velocity, delta_position)) / distance
        )
        spring_force = (
            (left_term + right_term) * delta_position / distance
        )

        if spring.p1 not in (0, CLOTH_NUM_X):
            cloth_forces[spring.p1] += spring_force
        if spring.p2 not in (0, CLOTH_NUM_X):
            cloth_forces[spring.p2] -= spring_force


def solve_rope_constraints(delta_time: float) -> None:
    """Advance the rope by one SPOOK-constrained time step."""
    if delta_time <= 0.0:
        raise ValueError("delta_time must be positive.")

    update_rope_constraint_geometry()

    velocities_at_step_start = rope_velocities.copy()

    # The assignment treats GRAVITY as force. Damping must depend on velocity;
    # it is not a constant force added to every coordinate.
    rope_forces[:] = DEFAULT_DAMPING * velocities_at_step_start
    for particle in range(ROPE_PARTICLES):
        if rope_inverse_masses[particle] > 0.0:
            rope_forces[particle] += GRAVITY
    rope_forces[ROPE_ANCHOR_INDEX].fill(0.0)

    unconstrained_velocities = (
        velocities_at_step_start
        + delta_time
        * rope_inverse_masses[:, np.newaxis]
        * rope_forces
    )

    spook_gamma = 1.0 / (
        1.0 + 4.0 * ROPE_RELAXATION_TIME / delta_time
    )
    compliance = 1.0 / ROPE_STIFFNESS
    spook_regularization = (
        4.0 * compliance * spook_gamma / (delta_time * delta_time)
    )

    jacobian_times_start_velocity = np.einsum(
        "cpd,pd->c",
        rope_constraint_jacobian,
        velocities_at_step_start,
    )
    rope_constraint_rhs[:] = (
        -4.0
        * spook_gamma
        / delta_time
        * rope_constraint_violations
        + spook_gamma * jacobian_times_start_velocity
    )

    # Lambda is scalar for each bilateral distance constraint. Starting from
    # zero avoids carrying a force from a different linearization.
    rope_constraint_multipliers.fill(0.0)
    constrained_velocities = unconstrained_velocities.copy()

    for _ in range(ROPE_GAUSS_SEIDEL_ITERATIONS):
        for constraint in range(ROPE_CONSTRAINTS):
            jacobian_row = rope_constraint_jacobian[constraint]

            jacobian_times_velocity = float(
                np.einsum(
                    "pd,pd->",
                    jacobian_row,
                    constrained_velocities,
                )
            )
            residual = (
                jacobian_times_velocity
                + spook_regularization
                * rope_constraint_multipliers[constraint]
                - rope_constraint_rhs[constraint]
            )

            schur_diagonal = spook_regularization
            for particle in range(ROPE_PARTICLES):
                schur_diagonal += (
                    rope_inverse_masses[particle]
                    * float(
                        np.dot(
                            jacobian_row[particle],
                            jacobian_row[particle],
                        )
                    )
                )

            delta_multiplier = -residual / schur_diagonal
            rope_constraint_multipliers[
                constraint
            ] += delta_multiplier

            # A distance constraint is bilateral, so the multiplier is not
            # clamped. It may pull a stretched segment or push a compressed one.
            for particle in range(ROPE_PARTICLES):
                constrained_velocities[particle] += (
                    rope_inverse_masses[particle]
                    * jacobian_row[particle]
                    * delta_multiplier
                )

    constrained_velocities[ROPE_ANCHOR_INDEX].fill(0.0)
    rope_velocities[:] = constrained_velocities

    for particle in range(ROPE_PARTICLES):
        if rope_inverse_masses[particle] == 0.0:
            continue
        rope_positions[particle] += delta_time * rope_velocities[particle]

        if rope_positions[particle, 1] < 0.0:
            rope_positions[particle, 1] = 0.0
            if rope_velocities[particle, 1] < 0.0:
                rope_velocities[particle, 1] = 0.0


def compute_forces(delta_time: float) -> None:
    compute_cloth_forces()
    solve_rope_constraints(delta_time)


def integrate_euler(delta_time: float) -> None:
    old_velocities = cloth_velocities.copy()
    cloth_velocities[:] += cloth_forces * (delta_time / MASS)
    cloth_positions[:] += delta_time * old_velocities

    below_ground = cloth_positions[:, 1] < 0.0
    cloth_positions[below_ground, 1] = 0.0


def ellipsoid_collision() -> None:
    for index in range(CLOTH_NUM_POINTS):
        world_position = np.append(cloth_positions[index], 1.0)
        transformed = inverse_ellipsoid_transform @ world_position
        delta0 = transformed[:3] - ELLIPSOID_CENTER
        distance = float(np.linalg.norm(delta0))

        if distance >= ELLIPSOID_OBJECT_RADIUS:
            continue

        if distance <= 1e-12:
            delta0 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            distance = 1e-12

        correction = (
            (ELLIPSOID_OBJECT_RADIUS - distance) * delta0 / distance
        )
        delta = np.zeros(3, dtype=np.float64)

        # This is the row-wise equivalent of the GLM indexing in the source.
        for axis in range(3):
            inverse_axis = ellipsoid_transform[axis, :3].copy()
            denominator = float(np.dot(inverse_axis, inverse_axis))
            if denominator > 1e-12:
                inverse_axis /= denominator
                delta[axis] = float(np.dot(correction, inverse_axis))

        cloth_positions[index] += delta
        cloth_velocities[index].fill(0.0)


def step_physics(delta_time: float) -> None:
    compute_forces(delta_time)
    integrate_euler(delta_time)
    ellipsoid_collision()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def draw_grid() -> None:
    glBegin(GL_LINES)
    glColor3f(0.5, 0.5, 0.5)

    for index in range(-GRID_SIZE, GRID_SIZE + 1):
        glVertex3f(float(index), 0.0, float(-GRID_SIZE))
        glVertex3f(float(index), 0.0, float(GRID_SIZE))

        glVertex3f(float(-GRID_SIZE), 0.0, float(index))
        glVertex3f(float(GRID_SIZE), 0.0, float(index))

    glEnd()


def draw_ellipsoid() -> None:
    glColor3f(0.0, 1.0, 0.0)
    glPushMatrix()
    glMultMatrixf(as_opengl_matrix(ellipsoid_transform))
    glutWireCube(ELLIPSOID_OBJECT_RADIUS)
    glPopMatrix()


def draw_rope() -> None:
    glBegin(GL_LINES)
    glColor3f(0.0, 1.0, 1.0)

    for particle in range(ROPE_PARTICLES - 1):
        p1 = rope_positions[particle]
        p2 = rope_positions[particle + 1]
        glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))
        glVertex3f(float(p2[0]), float(p2[1]), float(p2[2]))

    glEnd()

    glBegin(GL_POINTS)
    glColor3f(1.0, 1.0, 1.0)
    for point in rope_positions:
        glVertex3f(float(point[0]), float(point[1]), float(point[2]))
    glEnd()


def draw_cloth() -> None:
    glBegin(GL_TRIANGLES)
    glColor3f(0.0, 1.0, 1.0)

    for offset in range(0, len(cloth_indices), 3):
        for index in cloth_indices[offset : offset + 3]:
            point = cloth_positions[int(index)]
            glVertex3f(
                float(point[0]),
                float(point[1]),
                float(point[2]),
            )

    glEnd()

    glBegin(GL_POINTS)
    glColor3f(1.0, 1.0, 1.0)
    for point in cloth_positions:
        glVertex3f(float(point[0]), float(point[1]), float(point[2]))
    glEnd()


def on_render() -> None:
    global model_view_matrix, camera_right, view_direction
    global last_render_time, fps_start_time, total_frames, fps

    now = time.perf_counter()
    frame_time = now - last_render_time
    last_render_time = now

    total_frames += 1
    fps_elapsed = now - fps_start_time
    if fps_elapsed >= 1.0:
        fps = total_frames / fps_elapsed
        total_frames = 0
        fps_start_time = now

    title = f"{WINDOW_TITLE} | FPS: {fps:5.1f} | {frame_time * 1000:6.2f} ms"
    glutSetWindowTitle(title.encode("utf-8"))

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glTranslatef(0.0, 0.0, camera_distance)
    glRotatef(rotation_x, 1.0, 0.0, 0.0)
    glRotatef(rotation_y, 0.0, 1.0, 0.0)

    model_view_matrix = np.asarray(
        glGetDoublev(GL_MODELVIEW_MATRIX), dtype=np.float64
    )

    view_direction = np.array(
        [
            -model_view_matrix[0, 2],
            -model_view_matrix[1, 2],
            -model_view_matrix[2, 2],
        ],
        dtype=np.float64,
    )
    camera_right = np.cross(view_direction, WORLD_UP)

    draw_grid()
    draw_ellipsoid()
    draw_rope()
    draw_cloth()

    glutSwapBuffers()


def on_reshape(new_width: int, new_height: int) -> None:
    global viewport, projection_matrix, window_width, window_height

    window_width = max(1, new_width)
    window_height = max(1, new_height)

    glViewport(0, 0, window_width, window_height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(
        60.0,
        float(window_width) / float(window_height),
        1.0,
        100.0,
    )

    viewport = np.asarray(glGetIntegerv(GL_VIEWPORT), dtype=np.int32)
    projection_matrix = np.asarray(
        glGetDoublev(GL_PROJECTION_MATRIX), dtype=np.float64
    )

    glMatrixMode(GL_MODELVIEW)


# ---------------------------------------------------------------------------
# Input callbacks
# ---------------------------------------------------------------------------


def on_mouse_down(button: int, state: int, x: int, y: int) -> None:
    global old_mouse_x, old_mouse_y, mouse_state, selected_cloth_index

    if state == GLUT_DOWN:
        old_mouse_x = x
        old_mouse_y = y

        depth_value = glReadPixels(
            x,
            window_height - y,
            1,
            1,
            GL_DEPTH_COMPONENT,
            GL_FLOAT,
        )
        win_z = float(np.asarray(depth_value).reshape(-1)[0])
        if math.isclose(win_z, 1.0):
            win_z = 0.0

        object_position = gluUnProject(
            x,
            window_height - y,
            win_z,
            model_view_matrix,
            projection_matrix,
            viewport,
        )
        click_point = np.asarray(object_position, dtype=np.float64)

        distances = np.linalg.norm(cloth_positions - click_point, axis=1)
        nearest_index = int(np.argmin(distances))
        if distances[nearest_index] < 0.1:
            selected_cloth_index = nearest_index
            print(f"Intersected cloth vertex at index {nearest_index}")

    mouse_state = 0 if button == GLUT_MIDDLE_BUTTON else 1

    if state == GLUT_UP:
        selected_cloth_index = -1
        glutSetCursor(GLUT_CURSOR_INHERIT)


def on_mouse_move(x: int, y: int) -> None:
    global old_mouse_x, old_mouse_y
    global camera_distance, rotation_x, rotation_y

    if selected_cloth_index == -1:
        if mouse_state == 0:
            zoom_factor = 1.0 + (y - old_mouse_y) / 60.0
            # Avoid crossing or reaching the camera origin.
            if zoom_factor > 0.05:
                camera_distance *= zoom_factor
        else:
            rotation_y += (x - old_mouse_x) / 5.0
            rotation_x += (y - old_mouse_y) / 5.0
    else:
        delta = 1500.0 / max(abs(camera_distance), 1e-9)
        value_x = (x - old_mouse_x) / delta
        value_y = (old_mouse_y - y) / delta

        if abs(value_x) > abs(value_y):
            glutSetCursor(GLUT_CURSOR_LEFT_RIGHT)
        else:
            glutSetCursor(GLUT_CURSOR_UP_DOWN)

        cloth_velocities[selected_cloth_index].fill(0.0)
        cloth_positions[selected_cloth_index, 0] += (
            camera_right[0] * value_x
        )

        new_y = (
            cloth_positions[selected_cloth_index, 1]
            + WORLD_UP[1] * value_y
        )
        if new_y > 0.0:
            cloth_positions[selected_cloth_index, 1] = new_y

        cloth_positions[selected_cloth_index, 2] += (
            camera_right[2] * value_x + WORLD_UP[2] * value_y
        )

    old_mouse_x = x
    old_mouse_y = y
    glutPostRedisplay()


def on_keyboard(key: bytes, _x: int, _y: int) -> None:
    if key == b"\x1b":  # Escape
        shutdown()
        raise SystemExit(0)


def on_idle() -> None:
    global accumulator, last_physics_time

    now = time.perf_counter()
    elapsed = min(now - last_physics_time, 0.1)
    last_physics_time = now
    accumulator += elapsed

    # Use seconds consistently. The C++ source mixed millisecond timer output
    # with a seconds-based time step.
    steps = 0
    while accumulator >= TIME_STEP and steps < 6:
        step_physics(TIME_STEP)
        accumulator -= TIME_STEP
        steps += 1

    if steps == 6:
        accumulator = 0.0

    glutPostRedisplay()


def shutdown() -> None:
    cloth_springs.clear()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT)
    glutCreateWindow(WINDOW_TITLE.encode("utf-8"))

    glutDisplayFunc(on_render)
    glutReshapeFunc(on_reshape)
    glutIdleFunc(on_idle)
    glutMouseFunc(on_mouse_down)
    glutMotionFunc(on_mouse_move)
    glutKeyboardFunc(on_keyboard)

    # Some GLUT implementations do not expose a close callback.
    try:
        if bool(glutCloseFunc):
            glutCloseFunc(shutdown)
    except (NameError, RuntimeError, NullFunctionError):
        pass

    initialize_gl()
    glutMainLoop()


if __name__ == "__main__":
    main()