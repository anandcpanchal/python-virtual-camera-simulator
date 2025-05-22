import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw
from helper import *
from object_3d import Object3D

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import datetime

# --- Main Simulator Class ---
class VirtualCameraSimulator:
    # __init__ and other methods will go here
    # (The __init__ from previous response, slightly adapted for clarity)
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Camera Simulator")

        self.objects_3d = []
        self._create_default_object()

        self.canvas_width, self.canvas_height = 640, 320
        fx_init, fy_init = self.canvas_width * 1.2, self.canvas_height * 1.2  # Slightly narrower FoV
        cx_init, cy_init = self.canvas_width / 2.0, self.canvas_height / 2.0
        self.K_intrinsic = create_intrinsic_matrix(fx=fx_init, fy=fy_init, cx=cx_init, cy=cy_init)
        self.aperture = tk.DoubleVar(value=5.6)

        self.camera_pos_vars = {'x': tk.DoubleVar(value=0.0), 'y': tk.DoubleVar(value=0.0),
                                'z': tk.DoubleVar(value=100.0)}
        self.camera_rot_vars = {'rx': tk.DoubleVar(value=180.0), 'ry': tk.DoubleVar(value=0.0),
                                'rz': tk.DoubleVar(value=0.0)}
        self.camera_transform_configs = {'x': (-200, 200, 1), 'y': (-200, 200, 1), 'z': (1, 1000, 1),
                                         'rx': (-180, 180, 1), 'ry': (-360, 360, 1), 'rz': (-180, 180, 1)}
        self.object_position_offset = {'z': tk.DoubleVar(value=0.0)}

        self.last_mouse_x, self.last_mouse_y, self.dragging_mode, self.active_object_for_drag = 0, 0, None, None
        self.debug_mode_var = tk.BooleanVar(value=False)

        self.obj_transform_vars = {'tx': tk.DoubleVar(value=0.0), 'ty': tk.DoubleVar(value=0.0),
                                   'tz': tk.DoubleVar(value=0.0),
                                   'rx': tk.DoubleVar(value=0.0), 'ry': tk.DoubleVar(value=0.0),
                                   'rz': tk.DoubleVar(value=0.0),
                                   'sx': tk.DoubleVar(value=1.0), 'sy': tk.DoubleVar(value=1.0),
                                   'sz': tk.DoubleVar(value=1.0)}

        self.transform_configs = {'tx': (-100, 100, 1), 'ty': (-100, 100, 1), 'tz': (-100, 100, 1),
                                  'rx': (-180, 180, 5), 'ry': (-360, 360, 5), 'rz': (-180, 180, 5), 'sx': (0.1, 5, 0.1),
                                  'sy': (0.1, 5, 0.1), 'sz': (0.1, 5, 0.1)}

        self.controls_frame = ttk.LabelFrame(self.root, text="Controls (Units: mm, deg, px)")
        self.controls_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y, anchor='n')
        self.main_display_area = ttk.Frame(self.root)
        self.main_display_area.pack(side=tk.RIGHT, padx=10, pady=10, expand=True, fill="both")

        self.image_frame = ttk.LabelFrame(self.main_display_area, text="2D Projection (pixels)")
        self.image_canvas = None
        self.pil_image = Image.new("RGB", (self.canvas_width, self.canvas_height), "lightgrey")
        self.draw_context = ImageDraw.Draw(self.pil_image)

        self.view_3d_frame = ttk.LabelFrame(self.main_display_area, text="3D Scene View (mm)")
        self.fig_3d = Figure(figsize=(6, 5), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d_agg = None
        self.current_V_view_for_3d_plot = np.eye(4)

        self.obj_dims_var = tk.StringVar(value="Obj Dims (mm): N/A")
        self.pixel_coord_var = tk.StringVar(value="Cursor (px): (N/A)")
        self.measure_2d_status_var = tk.StringVar(value="2D Measure: OFF")
        self.measure_2d_x_measurement_var = tk.StringVar(value="X Measure: OFF")
        self.measure_2d_y_measurement_var = tk.StringVar(value="Y Measure: OFF")
        self.gsd_info_var = tk.StringVar(value="GSD (mm/px): N/A")
        self.measuring_2d_mode = False
        self.measurement_points_2d = []
        self.measurement_line_id_2d = None
        self.measurement_text_id_2d = None

        self.gsdx = None
        self.gsdy = None
        self._setup_gui()
        if self.objects_3d: self._display_object_dimensions(self.objects_3d[0])
        self.update_simulation()

    def _create_default_object(self):
        v = [[-10, -10, -10], [10, -10, -10], [10, 10, -10], [-10, 10, -10], [-10, -10, 10], [10, -10, 10],
             [10, 10, 10], [-10, 10, 10]]
        e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        # Define simple faces for the default cube for surface rendering
        f = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]  # CCW from outside
        # Default cube has one color, let Object3D handle default if no MTL
        self.objects_3d = [Object3D(v, e, f, default_color_hex="#4A90E2")]  # A nice blue

    def _display_object_dimensions(self, obj):
        if not obj or obj.vertices_local_orig.shape[0] == 0:
            self.obj_dims_var.set("Obj Dims (mm): N/A")
            return
        min_c, max_c = np.min(obj.vertices_local_orig, axis=0), np.max(obj.vertices_local_orig, axis=0)
        d = max_c - min_c
        self.obj_dims_var.set(f"Obj Dims (mm):\nX:{d[0]:.2f} Y:{d[1]:.2f} Z:{d[2]:.2f}")
        self.log_debug(f"Obj Dims (mm): X:{d[0]:.2f} Y:{d[1]:.2f} Z:{d[2]:.2f}")

    def _on_mouse_hover_2d_canvas(self, event):
        x, y = max(0, min(event.x, self.canvas_width - 1)), max(0, min(event.y, self.canvas_height - 1))
        self.pixel_coord_var.set(f"Cursor (px): ({x},{y})")

    def _on_mouse_leave_2d_canvas(self, event):
        self.pixel_coord_var.set("Cursor (px): (N/A)")

    def _toggle_measure_2d_mode(self):
        self.measuring_2d_mode = not self.measuring_2d_mode
        s = "2D Measure: Click 1st pt" if self.measuring_2d_mode else "2D Measure: OFF"
        self.measure_2d_status_var.set(s)
        self.measure_2d_x_measurement_var.set("")
        self.measure_2d_y_measurement_var.set("")
        self.measurement_points_2d = []
        self._clear_2d_measurement_drawing()
        if self.measuring_2d_mode and self.dragging_mode:
            self.dragging_mode = None
            self.active_object_for_drag = None

    def _clear_2d_measurement_drawing(self):
        if self.measurement_line_id_2d and self.image_canvas:
            self.image_canvas.delete(self.measurement_line_id_2d)
            self.measurement_line_id_2d = None
        if self.measurement_text_id_2d and self.image_canvas:
            self.image_canvas.delete(self.measurement_text_id_2d)
            self.measurement_text_id_2d = None

    def _handle_2d_measurement_click(self, event):
        if not self.measuring_2d_mode: return
        x, y = max(0, min(event.x, self.canvas_width - 1)), max(0, min(event.y, self.canvas_height - 1))
        self.measurement_points_2d.append((x, y))
        self.log_debug(f"2D Measure click@({x},{y}). Pts:{self.measurement_points_2d}")
        self._clear_2d_measurement_drawing()
        if len(self.measurement_points_2d) == 1:
            self.measure_2d_status_var.set("2D Measure: Click 2nd pt")
            p1 = self.measurement_points_2d[0]
            r = 3
            self.measurement_line_id_2d = self.image_canvas.create_oval(p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r,
                                                                        fill="cyan", outline="blue")
        elif len(self.measurement_points_2d) == 2:
            p1, p2 = self.measurement_points_2d[0], self.measurement_points_2d[1]
            self.measurement_line_id_2d = self.image_canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="red", width=2,
                                                                        dash=(4, 2))
            dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            txt = f"{dist:.2f} px"
            mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            self.measurement_text_id_2d = self.image_canvas.create_text(mid_x, mid_y - 10, text=txt, fill="blue",
                                                                        font=("Arial", 10, "bold"), anchor=tk.S)
            self.measure_2d_status_var.set(f"Measured: {txt}. Click 1st for new.")
            self.measure_2d_x_measurement_var.set(f"Measured: {dist * self.gsdx : .4f} mm if along x-axis")
            self.measure_2d_y_measurement_var.set(f"Measured: {dist * self.gsdy : .4f} mm if along y-axis")
            self.measurement_points_2d = []

    def _update_offset(self):
        self.update_simulation()

    # Inside VirtualCameraSimulator class:
    def _save_2d_projection_as_image(self):
        if self.pil_image is None:
            self.log_debug("No 2D image available to save.")
            # Optionally show a tk.messagebox.showinfo or .showerror
            tk.messagebox.showwarning("Save Error", "No 2D projection image is available to save.")
            return

        suggested_filename = self._generate_descriptive_filename()

        filepath = filedialog.asksaveasfilename(
            initialfile=suggested_filename,
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg;*.jpeg"),
                ("Bitmap files", "*.bmp"),
                ("GIF files", "*.gif"),
                ("All files", "*.*")
            ],
            title="Save 2D Projection As..."
        )

        if filepath:  # If the user didn't cancel
            try:
                self.pil_image.save(filepath)
                self.log_debug(f"2D projection saved to: {filepath}")
                tk.messagebox.showinfo("Save Successful", f"Image saved to:\n{filepath}")
            except Exception as e:
                self.log_debug(f"Error saving 2D projection: {e}")
                tk.messagebox.showerror("Save Error", f"Could not save image:\n{e}")
        else:
            self.log_debug("Save 2D projection cancelled by user.")

    def _generate_descriptive_filename(self):
        base = "projection"

        # Camera Intrinsic
        ci_fx = self.K_intrinsic[0][0]
        ci_fy = self.K_intrinsic[1][1]
        ci_s = self.K_intrinsic[0][1]
        ci_cx = self.K_intrinsic[0][2]
        ci_cy = self.K_intrinsic[1][2]
        cam_intrinsic_str = f"K_{ci_fx:.0f}-{ci_s:.0f}-{ci_cx:.0f}_{ci_fy:.0f}-{ci_cy:.0f}"

        # Camera Position
        cp_x = self.camera_pos_vars['x'].get()
        cp_y = self.camera_pos_vars['y'].get()
        cp_z = self.camera_pos_vars['z'].get()
        cam_pos_str = f"CPx{cp_x:.0f}y{cp_y:.0f}z{cp_z:.0f}"

        # Camera Rotation
        cr_p = self.camera_rot_vars['rx'].get()  # Pitch
        cr_y = self.camera_rot_vars['ry'].get()  # Yaw
        cr_r = self.camera_rot_vars['rz'].get()  # Roll
        cam_rot_str = f"CRp{cr_p:.0f}y{cr_y:.0f}r{cr_r:.0f}"

        obj_str = ""
        if self.objects_3d:
            obj = self.objects_3d[0]  # Using the first object's properties
            op_x, op_y, op_z = obj.translation[0], obj.translation[1], obj.translation[2]
            obj_pos_str = f"OPx{op_x:.0f}y{op_y:.0f}z{op_z:.0f}"

            or_p, or_y, or_r = obj.rotation_euler_deg[0], obj.rotation_euler_deg[1], obj.rotation_euler_deg[2]
            obj_rot_str = f"ORp{or_p:.0f}y{or_y:.0f}r{or_r:.0f}"
            obj_str = f"__{obj_pos_str}_{obj_rot_str}"  # Two underscores to separate clearly

        # Timestamp for uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Replace any characters that might be problematic in filenames (though less common with this format)
        # For simplicity, this example assumes the rounded floats don't produce issues.
        # Consider replacing '.' with 'p' if needed, e.g. "10p5" for 10.5

        filename = f"{base}_{cam_intrinsic_str}_{cam_pos_str}_{cam_rot_str}{obj_str}_{timestamp}.png"
        return filename

    def _setup_gui(self):
        # Camera Lens Parameters Frame
        cam_param_f = ttk.LabelFrame(self.controls_frame, text="Camera Lens (K in Pixels)")
        cam_param_f.pack(pady=5, fill=tk.X)
        intr_f = ttk.LabelFrame(cam_param_f, text="Intrinsic Matrix K")
        intr_f.pack(pady=5, padx=5, fill=tk.X)
        self.k_entries = {}
        k_labels_texts = [["fx", "s ", "cx"], ["0 ", "fy", "cy"], ["0 ", "0 ", "1 "]]  # Added spaces for alignment
        for r in range(3):
            for c in range(3):
                val, key_txt, key_e = self.K_intrinsic[r, c], k_labels_texts[r][c], f"k_{r}{c}"
                editable = not (
                        (r == 1 and c == 0) or (r == 2 and c == 0) or (r == 2 and c == 1) or (r == 2 and c == 2))
                ttk.Label(intr_f, text=key_txt + (": " if editable else "")).grid(row=r, column=2 * c, padx=(5, 0),
                                                                                  pady=2, sticky='w')
                if editable:
                    e = ttk.Entry(intr_f, width=7)
                    e.insert(0, str(val))
                    e.grid(row=r, column=2 * c + 1, padx=(0, 5), pady=2, sticky='ew')
                    e.bind("<FocusOut>", lambda ev, rr=r, cc=c: self._update_intrinsics_from_gui(rr, cc))
                    e.bind("<Return>", lambda ev, rr=r, cc=c: self._update_intrinsics_from_gui(rr, cc))
                    self.k_entries[key_e] = e
                else:
                    ttk.Label(intr_f, text=str(round(val, 2))).grid(row=r, column=2 * c + 1, padx=(0, 5), pady=2,
                                                                    sticky='w')

        ttk.Label(cam_param_f, text="Aperture (f-number):").pack(anchor='w', padx=5)
        ttk.Scale(cam_param_f, from_=1.0, to_=32.0, orient=tk.HORIZONTAL, variable=self.aperture,
                  command=lambda e: self.update_simulation()).pack(fill=tk.X, padx=5)
        self.ap_label = ttk.Label(cam_param_f)
        self.ap_label.pack(anchor='e', padx=5)
        self.aperture.trace_add("write", lambda *a: self.ap_label.config(text=f"{self.aperture.get():.1f}"))
        self.ap_label.config(text=f"{self.aperture.get():.1f}")  # Initial text

        # Camera Transform Frame
        cam_tf_f = ttk.LabelFrame(self.controls_frame, text="Camera Transform (Pos mm, Rot deg)")
        cam_tf_f.pack(pady=5, fill=tk.X)

        # Create a frame for Position controls
        pos_frame = ttk.Frame(cam_tf_f)
        pos_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nw')

        cam_pos_labs = {'x': "Pos X:", 'y': "Pos Y:", 'z': "Pos Z:"}
        ttk.Label(pos_frame, text="Position (mm):").grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 2))
        for i, (k, t) in enumerate(cam_pos_labs.items()):
            ttk.Label(pos_frame, text=t).grid(row=i + 1, column=0, sticky='w', pady=1)
            cfg = self.camera_transform_configs[k]
            ttk.Spinbox(pos_frame, from_=cfg[0], to=cfg[1], increment=cfg[2], textvariable=self.camera_pos_vars[k],
                        width=8, command=self.update_simulation).grid(row=i + 1, column=1, sticky='ew', pady=1)

        # Create a frame for Orientation controls
        rot_frame = ttk.Frame(cam_tf_f)
        rot_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nw')

        cam_rot_labs = {'rx': "PitchX°:", 'ry': "YawY°:", 'rz': "RollZ°:"}
        ttk.Label(rot_frame, text="Orientation (deg):").grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 2))
        for i, (k, t) in enumerate(cam_rot_labs.items()):
            ttk.Label(rot_frame, text=t).grid(row=i + 1, column=0, sticky='w', pady=1)
            cfg = self.camera_transform_configs[k]
            ttk.Spinbox(rot_frame, from_=cfg[0], to=cfg[1], increment=cfg[2], textvariable=self.camera_rot_vars[k],
                        width=8, command=self.update_simulation).grid(row=i + 1, column=1, sticky='ew', pady=1)

        # Configure column weights so frames expand nicely
        cam_tf_f.grid_columnconfigure(0, weight=1)
        cam_tf_f.grid_columnconfigure(1, weight=1)

        # Object Management Frame
        self.obj_mgmt_frame = ttk.LabelFrame(self.controls_frame, text="Object (Vertices in mm)")
        self.obj_mgmt_frame.pack(pady=5, fill=tk.X)
        ttk.Button(self.obj_mgmt_frame, text="Load (.obj)", command=self.load_object).pack(pady=(5, 0), padx=5,
                                                                                           fill=tk.X)
        dims_info_f = ttk.LabelFrame(self.obj_mgmt_frame, text="Info")
        dims_info_f.pack(pady=5, padx=5, fill=tk.X)
        ttk.Label(dims_info_f, textvariable=self.obj_dims_var, justify=tk.LEFT).pack(padx=5, pady=5, fill=tk.X)
        obj_tf_f = ttk.LabelFrame(self.obj_mgmt_frame, text="Transform (Translate mm)")
        obj_tf_f.pack(pady=5, padx=5, fill=tk.X)
        obj_tf_labs = {'tx': "TrX:", 'ty': "TrY:", 'tz': "TrZ:", 'rx': "RotX°:", 'ry': "RotY°:", 'rz': "RotZ°:",
                       'sx': "ScX:", 'sy': "ScY:", 'sz': "ScZ:"}
        for i, (k, t) in enumerate(obj_tf_labs.items()):
            r, c = divmod(i, 3)
            ttk.Label(obj_tf_f, text=t).grid(row=r, column=c * 2, sticky='w', padx=(5, 0), pady=1)
            cfg = self.transform_configs[k]
            ttk.Spinbox(obj_tf_f, from_=cfg[0], to=cfg[1], increment=cfg[2], textvariable=self.obj_transform_vars[k],
                        width=6, command=self._update_object_transform).grid(row=r, column=c * 2 + 1, sticky='ew',
                                                                             padx=(0, 5), pady=1)

        # Measurement Tools Frame
        measure_f = ttk.LabelFrame(self.controls_frame, text="Measurement Tools")
        measure_f.pack(pady=5, fill=tk.X)

        ttk.Button(measure_f, text="Measure 2D Dist (px)", command=self._toggle_measure_2d_mode).pack(pady=(5, 0),
                                                                                                      padx=5, fill=tk.X)
        ttk.Label(measure_f, textvariable=self.measure_2d_status_var).pack(padx=5)
        ttk.Label(measure_f, textvariable=self.measure_2d_x_measurement_var).pack(padx=5)
        ttk.Label(measure_f, textvariable=self.measure_2d_y_measurement_var).pack(padx=5)
        ttk.Label(measure_f, textvariable=self.gsd_info_var, justify=tk.LEFT).pack(pady=5, padx=5, fill=tk.X)

        # Create a new frame for the Offset-Z label and spinbox
        offset_z_frame = ttk.Frame(measure_f)
        offset_z_frame.pack(pady=(5, 0), padx=5, fill=tk.X)  # Pack this frame within measure_f

        # Now use grid within the new offset_z_frame
        ttk.Label(offset_z_frame, text="Offset-Z").grid(row=0, column=0, sticky='w',
                                                        padx=(0, 5))  # Removed padx on outer frame
        ttk.Spinbox(offset_z_frame, from_=-100, to=100, increment=1, textvariable=self.object_position_offset['z'],
                    width=6, command=self._update_offset).grid(row=0, column=1, sticky='ew')

        # Configure the column weights for the offset_z_frame to make spinbox expand
        offset_z_frame.grid_columnconfigure(0, weight=0)  # Label doesn't need to expand
        offset_z_frame.grid_columnconfigure(1, weight=1)  # Spinbox will take available space

        # Debug Frame
        debug_f = ttk.LabelFrame(self.controls_frame, text="Debugging")
        debug_f.pack(pady=5, fill=tk.X)
        ttk.Checkbutton(debug_f, text="Enable Debug Log", variable=self.debug_mode_var,
                        command=self._on_debug_toggle).pack(pady=5, padx=5, anchor='w')

        # Display Areas
        self.image_frame.pack(side=tk.TOP, padx=0, pady=(0, 5), expand=True, fill="both")  # No X padx for image_frame
        self.image_canvas = tk.Canvas(self.image_frame, width=self.canvas_width, height=self.canvas_height,
                                      bg="lightgrey")
        self.image_canvas.pack(expand=True, fill="both")

        save_button = ttk.Button(self.image_frame, text="Save 2D Projection Image", command=self._save_2d_projection_as_image)
        save_button.pack(pady=5, padx=5, fill=tk.X)
        ttk.Label(self.image_frame, textvariable=self.pixel_coord_var).pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=2)

        self.image_canvas.bind("<Motion>", self._on_mouse_hover_2d_canvas)
        self.image_canvas.bind("<Leave>", self._on_mouse_leave_2d_canvas)
        self.image_canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.image_canvas.bind("<ButtonPress-3>", self._on_mouse_press)
        self.image_canvas.bind("<B1-Motion>", self._on_mouse_motion)
        self.image_canvas.bind("<B3-Motion>", self._on_mouse_motion)
        self.image_canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.image_canvas.bind("<ButtonRelease-3>", self._on_mouse_release)

        self.view_3d_frame.pack(side=tk.BOTTOM, padx=0, pady=(5, 0), expand=True, fill="both")  # No X padx
        self.canvas_3d_agg = FigureCanvasTkAgg(self.fig_3d, master=self.view_3d_frame)
        self.canvas_3d_agg.draw()
        self.canvas_3d_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fig_3d.tight_layout()  # Prevent labels overlapping

    def _on_debug_toggle(self):
        self.log_debug(f"Debug mode: {self.debug_mode_var.get()}")

    def log_debug(self, msg):
        if self.debug_mode_var.get():
            if isinstance(msg, np.ndarray):
                print(f"[DBG]\n{np.array2string(msg, precision=3, suppress_small=True, max_line_width=120)}")
            else:
                print(f"[DBG] {msg}")

    def _on_mouse_press(self, event):
        if self.measuring_2d_mode and event.num == 1:
            self._handle_2d_measurement_click(event)
            return
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        if not self.objects_3d:
            self.active_object_for_drag = None
            return
        self.active_object_for_drag = self.objects_3d[0]
        if event.num == 1:
            self.dragging_mode = "translate"
            self.log_debug(f"Start obj translate drag")
        elif event.num == 3:
            self.dragging_mode = "rotate"
            self.log_debug(f"Start obj rotate drag")
        else:
            self.dragging_mode = None

    def _on_mouse_motion(self, event):
        if not self.active_object_for_drag or not self.dragging_mode: return
        dx, dy = event.x - self.last_mouse_x, event.y - self.last_mouse_y
        action = False
        obj = self.active_object_for_drag
        if self.dragging_mode == "translate":
            sens = (self.camera_pos_vars['z'].get() / 2.0) / (
                    self.canvas_width / 2.0)  # mm translation per pixel drag, adjust as needed
            R_wc = self.current_V_view_for_3d_plot[:3, :3]
            cam_x_w, cam_y_w = R_wc.T[:, 0], R_wc.T[:, 1]
            obj.translation += cam_x_w * dx * sens
            obj.translation -= cam_y_w * dy * sens
            for i, k in enumerate(['tx', 'ty', 'tz']): self.obj_transform_vars[k].set(round(obj.translation[i], 2))
            action = True
        elif self.dragging_mode == "rotate":
            sens_deg = 0.5
            obj.rotation_euler_deg[1] += dx * sens_deg
            obj.rotation_euler_deg[0] -= dy * sens_deg  # Yaw, Pitch
            for i, k in enumerate(['rx', 'ry']): self.obj_transform_vars[k].set(
                round(obj.rotation_euler_deg[i] % 360, 1))
            action = True
        if action: self.update_simulation()
        self.last_mouse_x, self.last_mouse_y = event.x, event.y

    def _on_mouse_release(self, event):
        self.log_debug(f"Drag release")
        self.dragging_mode, self.active_object_for_drag = None, None

    def _update_3d_view(self):
        self.ax_3d.clear()
        all_pts = []
        for obj in self.objects_3d:
            m = obj.get_model_matrix()
            world_v_h = (m @ obj.vertices_local.T).T
            world_v = world_v_h[:, :3] / np.maximum(world_v_h[:, 3, np.newaxis], 1e-9)
            all_pts.extend(world_v.tolist())
            if obj.faces:
                fvl, fcl = [], []
                for i, fi in enumerate(obj.faces):
                    if not all(0 <= idx < len(world_v) for idx in fi) or len(fi) < 3: continue
                    fv3d = [world_v[idx] for idx in fi]
                    fvl.append(fv3d)
                    rgb_int = obj.get_face_color_rgb_int(i)
                    fcl.append(tuple(c / 255. for c in rgb_int))
                if fvl: self.ax_3d.add_collection3d(Poly3DCollection(fvl, fc=fcl, lw=0.3, ec='dimgray', alpha=1.0))
            else:  # Wireframe fallback for 3D view
                for e in obj.edges:
                    p1, p2 = world_v[e[0]], world_v[e[1]]
                    self.ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                    color=rgb_tuple_to_hex(obj.default_rgb_int), lw=1)

        cam_p_w = np.array([self.camera_pos_vars[k].get() for k in ['x', 'y', 'z']])
        if not np.any(np.isnan(cam_p_w)): all_pts.append(cam_p_w.tolist())
        R_wc = self.current_V_view_for_3d_plot[:3, :3]
        cam_look_dir_w = R_wc.T[:, 2]

        scl_ref_pt = np.mean([p for p in all_pts if not np.any(np.isnan(p))], axis=0) if all_pts else cam_p_w
        dist_scl = np.linalg.norm(cam_p_w - scl_ref_pt)
        if dist_scl < 1e-1: dist_scl = np.linalg.norm(cam_p_w)  # If close to scene center, use dist from origin
        if dist_scl < 1e-1: dist_scl = 50.0  # Further fallback

        cone_h = max(1.0, dist_scl * 0.12)
        cone_r = cone_h * 0.4
        cV, cE = create_cone_wireframe(cam_p_w, cam_look_dir_w, cone_h, cone_r, 8)
        if cV.size > 0: all_pts.extend(cV.tolist())
        for e in cE:
            p1, p2 = cV[e[0]], cV[e[1]]
            self.ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='darkorchid', lw=1)

        valid_pts = [p for p in all_pts if not np.any(np.isnan(p))]
        if valid_pts:
            pts_arr = np.array(valid_pts)
            min_c, max_c = pts_arr.min(axis=0), pts_arr.max(axis=0)
            rng_d = np.maximum(max_c - min_c, np.array([1., 1., 1.]))
            ctr = (max_c + min_c) / 2.
            max_r = np.max(rng_d) * 0.65  # Ensure a bit more padding
            self.ax_3d.set_xlim(ctr[0] - max_r, ctr[0] + max_r)
            self.ax_3d.set_ylim(ctr[1] - max_r, ctr[1] + max_r)
            self.ax_3d.set_zlim(ctr[2] - max_r, ctr[2] + max_r)
        else:
            self.ax_3d.set_xlim([-5, 5])
            self.ax_3d.set_ylim([-5, 5])
            self.ax_3d.set_zlim([-5, 5])
        self.ax_3d.set_xlabel("X(mm)")
        self.ax_3d.set_ylabel("Y(mm)")
        self.ax_3d.set_zlabel("Z(mm)")
        self.ax_3d.set_title("3D Scene")
        self.canvas_3d_agg.draw()

    def _update_object_transform(self):
        if not self.objects_3d: return
        try:
            obj = self.objects_3d[0]
            obj.translation = np.array([self.obj_transform_vars[k].get() for k in ['tx', 'ty', 'tz']])
            obj.rotation_euler_deg = np.array([self.obj_transform_vars[k].get() for k in ['rx', 'ry', 'rz']])
            obj.scale = np.array([self.obj_transform_vars[k].get() for k in ['sx', 'sy', 'sz']])
            self.update_simulation()
        except ValueError:
            print("Err: Invalid obj transform val")

    def _update_intrinsics_from_gui(self, r, c):
        key = f"k_{r}{c}"
        if key not in self.k_entries: return  # Should not happen if UI setup correctly
        try:
            val = float(self.k_entries[key].get())
            self.K_intrinsic[r, c] = val
            self.log_debug(f"K[{r},{c}] updated to {val}")
            self.update_simulation()
        except ValueError:
            print(f"Err: Invalid K[{r},{c}]")
            e = self.k_entries[key]
            e.delete(0, tk.END)
            e.insert(0, str(
                self.K_intrinsic[r, c]))

    def load_object(self):
        fp = filedialog.askopenfilename(title="Load OBJ", filetypes=(("OBJ", "*.obj"), ("All", "*.*")))
        if fp:
            new_obj = Object3D.load_from_obj(fp)  # default_color_hex is used inside load_from_obj
            if new_obj and new_obj.vertices_local.shape[0] > 0:
                self.objects_3d = [new_obj]
                for k in ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']: self.obj_transform_vars[k].set(0.0)
                for k in ['sx', 'sy', 'sz']: self.obj_transform_vars[k].set(1.0)
                self._display_object_dimensions(new_obj)
                self._update_object_transform()
            else:
                print("Warn: Load OBJ failed or obj empty.")

    def project_points(self, verts_cam_h, K):  # verts_cam_h are (N,4) in mm, K is 3x3
        pts2d, z_depths_mm = [], []
        for v_cam_h in verts_cam_h:
            Xc, Yc, Zc, Wc = v_cam_h
            if abs(Wc) > 1e-9:
                Xc /= Wc
                Yc /= Wc
                Zc /= Wc
            z_depths_mm.append(Zc)
            if Zc <= 0.01:
                pts2d.append(None)
                continue
            uvw_prime = K @ np.array([Xc, Yc, Zc])  # u',v' in pixels*mm/mm=pixels w' in mm
            w_prime_mm = uvw_prime[2]  # This is Zc (mm)
            if abs(w_prime_mm) < 1e-6:
                pts2d.append(None)
                continue
            pts2d.append((uvw_prime[0] / w_prime_mm, uvw_prime[1] / w_prime_mm))
        return pts2d, z_depths_mm

    def _get_aperture_effect_color(self, base_rgb_tuple_int, z_cam_mm, focal_plane_z_cam_mm, f_stop):
        # This function is now less relevant if we are primarily drawing shaded polygons.
        # If used for wireframes, base_rgb_tuple_int is needed.
        abs_focal_mm, abs_z_mm = abs(focal_plane_z_cam_mm), abs(z_cam_mm)
        dof_factor = 1.0 / max(0.1, f_stop)
        sharp_mm = max(0.1 * abs_focal_mm, abs_focal_mm * dof_factor * 0.15)  # Heuristic range in mm
        diff_mm = abs(abs_z_mm - abs_focal_mm)
        if diff_mm < sharp_mm: return rgb_tuple_to_hex(base_rgb_tuple_int)  # Sharp

        blur_num = diff_mm - sharp_mm
        blur_den = abs_focal_mm * 0.75 + 1e-3  # Denominator for blur factor, in mm
        if blur_den < 1e-3: blur_den = max(1.0, abs_focal_mm)

        blur = min(1.0, blur_num / blur_den)
        dim = 1.0 - blur * 0.75  # Dim more
        rgb_dimmed = tuple(min(255, max(0, int(c * dim))) for c in base_rgb_tuple_int)
        # self.log_debug(f"AperCalc: z={z_cam_mm:.1f}, f_plane={focal_plane_z_cam_mm:.1f}, sharp_r={sharp_mm:.2f} -> blur={blur:.2f}")
        return rgb_tuple_to_hex(rgb_dimmed)

    def update_simulation(self, event=None):
        self.log_debug("--- SIMULATION UPDATE START ---")
        self.draw_context.rectangle([0, 0, self.canvas_width, self.canvas_height], fill="lightgrey")

        cam_p = np.array([self.camera_pos_vars[k].get() for k in ['x', 'y', 'z']])
        cam_r_deg = np.array([self.camera_rot_vars[k].get() for k in ['rx', 'ry', 'rz']])
        self.log_debug(f"Cam Pos(mm): {cam_p}, Rot(deg): {cam_r_deg}")

        Rrz = create_rotation_matrix_z(math.radians(cam_r_deg[2]))
        Rry = create_rotation_matrix_y(math.radians(cam_r_deg[1]))
        Rrx = create_rotation_matrix_x(math.radians(cam_r_deg[0]))
        R_cam_world = Rrz @ Rry @ Rrx  # ZYX order for Roll, Yaw, Pitch

        cam_fwd_loc_h = np.array([0, 0, -1, 0])
        cam_up_loc_h = np.array([0, 1, 0, 0])
        world_fwd = (R_cam_world @ cam_fwd_loc_h)[:3]
        world_up = (R_cam_world @ cam_up_loc_h)[:3]
        if np.linalg.norm(world_fwd) < 1e-6:
            world_fwd = np.array([0, 0, -1])
        else:
            world_fwd /= np.linalg.norm(world_fwd)
        if np.linalg.norm(world_up) < 1e-6:
            world_up = np.array([0, 1, 0])
        else:
            world_up /= np.linalg.norm(world_up)

        target_dist_mm = 50.0  # Could be a slider too
        cam_target_w = cam_p + world_fwd * target_dist_mm
        V_view = create_view_matrix(cam_p, cam_target_w, world_up)
        self.current_V_view_for_3d_plot = V_view
        self.log_debug(f"K (px):\n{self.K_intrinsic}")
        self.log_debug(f"V_view (mm):\n{V_view}")

        target_h_dof = np.append(cam_target_w, 1.0)
        focal_pt_cam_h = V_view @ target_h_dof
        focal_plane_Zc_mm = focal_pt_cam_h[2] / focal_pt_cam_h[3] if abs(focal_pt_cam_h[3]) > 1e-9 else focal_pt_cam_h[
            2]
        f_stop = self.aperture.get()
        self.log_debug(f"FocalPlane Zc: {focal_plane_Zc_mm:.2f}mm. Aperture: f/{f_stop:.1f}")

        # GSD
        if self.objects_3d:
            obj0 = self.objects_3d[0]
            M0 = obj0.get_model_matrix()
            MV0 = V_view @ M0
            obj0_orig_cam_h = MV0 @ np.array([0, 0, 0, 1])
            obj0_Zc_mm = obj0_orig_cam_h[2] / obj0_orig_cam_h[3] if abs(obj0_orig_cam_h[3]) > 1e-9 else obj0_orig_cam_h[2]

            #TODO: Z offset handling
            obj0_Zc_mm = obj0_Zc_mm - self.object_position_offset['z'].get()

            if obj0_Zc_mm > 0:
                fx, fy = self.K_intrinsic[0, 0], self.K_intrinsic[1, 1]
                self.gsdx = obj0_Zc_mm / fx if abs(fx) > 1e-6 else float('inf')
                self.gsdy = obj0_Zc_mm / fy if abs(fy) > 1e-6 else float('inf')
                self.gsd_info_var.set(f"GSD@Obj Zc={obj0_Zc_mm:.1f}mm \nX:{self.gsdx:.4f} mm/px \nY:{self.gsdy:.4f} mm/px")
            else:
                self.gsd_info_var.set(f"GSD: Obj Zc={obj0_Zc_mm:.1f}mm (Invalid Zc)")
        else:
            self.gsd_info_var.set("GSD (mm/px): No object")

        # 2D Surface Projection
        if not self.objects_3d:
            if self.image_canvas:
                self.tk_image = ImageTk.PhotoImage(self.pil_image)
                self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self._update_3d_view()
            return

        light_dir_w = np.array([0.6, 0.7, 1.0])
        light_dir_w /= np.linalg.norm(light_dir_w)
        amb = 0.35
        all_faces_2d = []

        for obj_i, obj in enumerate(self.objects_3d):
            M = obj.get_model_matrix()
            all_v_loc_h = obj.vertices_local
            all_v_world_h = (M @ all_v_loc_h.T).T
            all_v_cam_h = (V_view @ all_v_world_h.T).T
            all_v_world = all_v_world_h[:, :3] / np.maximum(all_v_world_h[:, 3, np.newaxis], 1e-9)

            if not obj.faces:
                self.log_debug(f"Obj {obj_i} no faces for 2D surf.")
                continue
            for face_j, face_indices in enumerate(obj.faces):
                if len(face_indices) < 3: continue
                face_v_w = [all_v_world[idx] for idx in face_indices]
                face_v_cam_h = [all_v_cam_h[idx] for idx in face_indices]

                v0w, v1w, v2w = face_v_w[0], face_v_w[1], face_v_w[2]
                norm_w = np.cross(v1w - v0w, v2w - v0w)
                if np.linalg.norm(norm_w) < 1e-6: continue
                norm_w /= np.linalg.norm(norm_w)

                center_w = np.mean(np.array(face_v_w), axis=0)
                view_to_face_w = center_w - cam_p  # Vector from camera eye to face center

                if np.dot(norm_w,
                          view_to_face_w) >= 0.0: continue  # Back-face culling (normal points away from camera or parallel)

                diff_int = max(0, np.dot(norm_w, light_dir_w))
                intensity = amb + (1 - amb) * diff_int
                base_rgb = obj.get_face_color_rgb_int(face_j)
                s_rgb = tuple(min(255, int(c * intensity)) for c in base_rgb)
                fill_hex = rgb_tuple_to_hex(s_rgb)

                scr_pts, face_Zc_mm_vals, valid_proj = [], [], True
                for vch in face_v_cam_h:
                    Xc, Yc, Zc, Wc = vch
                    if abs(Wc) > 1e-9:
                        Xc /= Wc
                        Yc /= Wc
                        Zc /= Wc
                    else:
                        valid_proj = False
                        break
                    face_Zc_mm_vals.append(Zc)
                    if Zc <= 0.01:
                        valid_proj = False
                        break  # Near clip
                    uvw_p = self.K_intrinsic @ np.array([Xc, Yc, Zc])
                    if abs(uvw_p[2]) < 1e-6:
                        valid_proj = False
                        break
                    scr_pts.append((int(round(uvw_p[0] / uvw_p[2])), int(round(uvw_p[1] / uvw_p[2]))))

                if not valid_proj or len(scr_pts) < 3: continue
                all_faces_2d.append((np.mean(face_Zc_mm_vals), scr_pts, fill_hex, "dimgray"))

        all_faces_2d.sort(key=lambda x: x[0], reverse=True)  # Painter's
        for _, pts, fill, outl in all_faces_2d:
            if len(pts) >= 3: self.draw_context.polygon(pts, fill=fill, outline=outl, width=1)

        if self.image_canvas:
            self.tk_image = ImageTk.PhotoImage(self.pil_image)
            self.image_canvas.create_image(0, 0,
                                           anchor=tk.NW,
                                           image=self.tk_image)
        self._update_3d_view()