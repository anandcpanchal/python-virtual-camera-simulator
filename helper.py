import math
import numpy as np

# --- Color Helper Functions ---
def hex_to_rgb_tuple(hex_color_str):
    hex_color_str = str(hex_color_str).lstrip('#')
    if len(hex_color_str) == 6:
        try:
            return tuple(int(hex_color_str[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            pass  # Fall through to named colors or default

    named_colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
                    "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
                    "white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128),
                    "grey": (128, 128, 128), "darkorchid": (153, 50, 204),
                    "dimgray": (105, 105, 105), "lightgrey": (211, 211, 211),
                    "darkslategrey": (47, 79, 79)}  # Added more common names

    if str(hex_color_str).lower() in named_colors:
        return named_colors[str(hex_color_str).lower()]

    # print(f"Warning: Could not parse color '{hex_color_str}', defaulting to grey.")
    return (128, 128, 128)


def rgb_tuple_to_hex(rgb_tuple_int):
    r, g, b = rgb_tuple_int
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


# --- Transformation Helper Functions ---
def create_translation_matrix(tx, ty, tz):
    return np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])


def create_rotation_matrix_x(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def create_rotation_matrix_y(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def create_rotation_matrix_z(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def create_scale_matrix(sx, sy, sz):
    return np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]])


# --- Camera & Geometry Helper Functions ---
def create_intrinsic_matrix(fx, fy, cx, cy, s=0):
    return np.array([[fx, s, cx], [0, fy, cy], [0, 0, 1]])


def create_view_matrix(camera_position, target_position, up_vector):
    eye = np.array(camera_position, dtype=float)
    target = np.array(target_position, dtype=float)
    up = np.array(up_vector, dtype=float)

    F = target - eye
    if np.linalg.norm(F) < 1e-9: F = np.array([0, 0, -1.0])  # Avoid zero vector
    f = F / np.linalg.norm(F)

    if np.linalg.norm(up) < 1e-9: up = np.array([0, 1.0, 0])
    UP_norm = up / np.linalg.norm(up)

    S = np.cross(f, UP_norm)
    if np.linalg.norm(S) < 1e-9:  # f and UP_norm are collinear
        if abs(f[1]) > 0.99:
            UP_norm = np.array([0, 0, 1.0 if f[1] < 0 else -1.0])  # if f is Y-axis, up is Z
        else:
            UP_norm = np.array([0, 1.0, 0])  # Default up if f is not Y-axis
        S = np.cross(f, UP_norm)
        if np.linalg.norm(S) < 1e-9:  # Still collinear (e.g. f is Z, up is Z)
            S = np.cross(f, np.array([1.0, 0, 0]))  # Try X as new up
    s_norm = S / np.linalg.norm(S)
    U = np.cross(s_norm, f)  # U is already normalized if s_norm and f are ortho-normal

    R_world_to_cam = np.array([s_norm, U, -f])
    t_world_to_cam = -R_world_to_cam @ eye

    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R_world_to_cam
    view_matrix[:3, 3] = t_world_to_cam
    return view_matrix


def parse_mtl_file(filepath):
    materials = {}
    current_material_name = None
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts: continue
                cmd, vals = parts[0], parts[1:]
                if cmd == 'newmtl':
                    current_material_name = vals[0]
                    materials[current_material_name] = {}
                elif current_material_name:
                    if cmd == 'Kd' and len(vals) >= 3:
                        try:
                            r, g, b = float(vals[0]), float(vals[1]), float(vals[2])
                            rgb_int = (min(255, int(r * 255)), min(255, int(g * 255)), min(255, int(b * 255)))
                            materials[current_material_name]['Kd_rgb_int'] = rgb_int
                        except ValueError:
                            print(f"Warning [MTL Ln {line_num}]: Invalid Kd values {vals}")
    except FileNotFoundError:
        print(f"Warning: MTL file not found: {filepath}")
    except Exception as e:
        print(f"Error parsing MTL {filepath}: {e}")
    return materials


def create_cone_wireframe(apex, direction, height, base_radius, num_segments=12):
    vertices, edges = [], []
    vertices.append(np.array(apex))
    apex_idx = 0
    if np.linalg.norm(direction) < 1e-6:
        direction_norm = np.array([0, 0, 1.0])
    else:
        direction_norm = np.array(direction) / np.linalg.norm(direction)
    base_center = np.array(apex) + direction_norm * height
    ref_vec = np.array([0, 1, 0]) if np.abs(np.dot(direction_norm, np.array([0, 1, 0]))) < 0.99 else np.array([1, 0, 0])
    u = np.cross(direction_norm, ref_vec)
    u /= np.linalg.norm(u)
    v = np.cross(direction_norm, u)
    base_indices = []
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments
        pt = base_center + base_radius * (math.cos(angle) * u + math.sin(angle) * v)
        vertices.append(pt)
        base_indices.append(len(vertices) - 1)
    for bi in base_indices: edges.append((apex_idx, bi))
    for i in range(num_segments): edges.append((base_indices[i], base_indices[(i + 1) % num_segments]))
    return np.array(vertices), edges