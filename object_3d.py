import os
from helper import *

# --- Object3D Class ---
class Object3D:
    def __init__(self, vertices_local, edges=None, faces=None,
                 face_colors_rgb_int=None, default_color_hex="#B0B0B0"):  # Default light grey
        self.vertices_local_orig = np.array(vertices_local, dtype=float)
        if self.vertices_local_orig.shape[0] > 0:
            if self.vertices_local_orig.shape[1] == 3:
                self.vertices_local = np.hstack(
                    [self.vertices_local_orig, np.ones((self.vertices_local_orig.shape[0], 1))])
            elif self.vertices_local_orig.shape[1] == 4:
                self.vertices_local = self.vertices_local_orig
            else:
                self._init_empty_verts()
        else:
            self._init_empty_verts()

        self.edges = edges if edges else []
        self.faces = faces if faces else []
        self.face_colors_rgb_int = face_colors_rgb_int if face_colors_rgb_int and len(face_colors_rgb_int) == len(
            self.faces) else []
        self.default_rgb_int = hex_to_rgb_tuple(default_color_hex)
        self.translation = np.array([0.0, 0.0, 0.0])
        self.rotation_euler_deg = np.array([0.0, 0.0, 0.0])
        self.scale = np.array([1.0, 1.0, 1.0])

    def _init_empty_verts(self):
        # print("Warning: Object3D initialized with malformed or empty vertices.")
        self.vertices_local_orig = np.empty((0, 3), dtype=float)
        self.vertices_local = np.empty((0, 4), dtype=float)

    def get_model_matrix(self):
        TR = create_translation_matrix(*self.translation)
        RX = create_rotation_matrix_x(math.radians(self.rotation_euler_deg[0]))
        RY = create_rotation_matrix_y(math.radians(self.rotation_euler_deg[1]))
        RZ = create_rotation_matrix_z(math.radians(self.rotation_euler_deg[2]))
        S = create_scale_matrix(*self.scale)
        return TR @ RZ @ RY @ RX @ S  # Apply scale first, then rotations, then translation

    def get_face_color_rgb_int(self, face_index):
        if self.face_colors_rgb_int and 0 <= face_index < len(self.face_colors_rgb_int) and self.face_colors_rgb_int[
            face_index] is not None:
            return self.face_colors_rgb_int[face_index]
        return self.default_rgb_int

    @classmethod
    def load_from_obj(cls, filepath, default_color_hex="#B0B0B0"):
        vertices, faces_data = [], []
        materials, current_material_name, obj_dir = {}, None, os.path.dirname(filepath)
        try:
            with open(filepath, 'r') as f:
                for ln, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if not parts: continue
                    cmd, args = parts[0], parts[1:]
                    if cmd == 'mtllib' and args:
                        mtl_fp = args[0] if os.path.isabs(args[0]) else os.path.join(obj_dir, args[0])
                        materials.update(parse_mtl_file(mtl_fp))
                    elif cmd == 'usemtl' and args:
                        current_material_name = args[0]
                    elif cmd == 'v' and len(args) >= 3:
                        vertices.append([float(args[0]), float(args[1]), float(args[2])])
                    elif cmd == 'f' and args:
                        try:
                            faces_data.append({'indices': tuple(int(p.split('/')[0]) - 1 for p in args),
                                               'material': current_material_name})
                        except:
                            print(f"Warning: Skipping malformed face (line {ln}): {line.strip()}")
            if not vertices:
                print(f"Warning: No vertices in {filepath}")
                return cls([], default_color_hex=default_color_hex)

            faces_list = [fd['indices'] for fd in faces_data]
            resolved_face_colors = []
            default_rgb = hex_to_rgb_tuple(default_color_hex)
            for fd in faces_data:
                mat_name = fd['material']
                if mat_name and mat_name in materials and 'Kd_rgb_int' in materials[mat_name]:
                    resolved_face_colors.append(materials[mat_name]['Kd_rgb_int'])
                else:
                    resolved_face_colors.append(default_rgb)

            edges = set()
            for face in faces_list:
                for i in range(len(face)):
                    v1_idx, v2_idx = face[i], face[(i + 1) % len(face)]
                    if 0 <= v1_idx < len(vertices) and 0 <= v2_idx < len(vertices): edges.add(
                        tuple(sorted((v1_idx, v2_idx))))

            print(f"Loaded: {os.path.basename(filepath)}, V:{len(vertices)}, F:{len(faces_list)}, Mtl:{len(materials)}")
            return cls(vertices, list(edges), faces_list, resolved_face_colors, default_color_hex)
        except FileNotFoundError:
            print(f"Error: OBJ File not found: {filepath}")
        except Exception as e:
            print(f"Error loading OBJ {filepath}: {e}")
        return cls([], default_color_hex=default_color_hex)

