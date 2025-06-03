import tkinter as tk
from tkinter import ttk, filedialog

import darkdetect
from PIL import Image, ImageTk, ImageDraw
from helper import *
from object_3d import Object3D

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # Add NavigationToolbar2Tk
import datetime

ENABLE_DEBUG = True

# --- Main Simulator Class ---
class VirtualCameraSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Camera Simulator")
        try:
            self.root.state('zoomed')
        except tk.TclError:
            # self.log_debug is not defined yet, so we can't call it here.
            # Consider defining a basic print logger early or skip logging for this specific error.
            print("INFO: Could not set root window to 'zoomed' state (platform dependent).")

        self.save_image_upscale_factor = 10

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        outer_scrollable_container = ttk.Frame(self.root)
        outer_scrollable_container.grid(row=0, column=0, sticky='nsew')
        outer_scrollable_container.grid_rowconfigure(0, weight=1)
        outer_scrollable_container.grid_columnconfigure(0, weight=1)

        self.root_canvas = tk.Canvas(outer_scrollable_container, borderwidth=0, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(outer_scrollable_container, orient="vertical", command=self.root_canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(outer_scrollable_container, orient="horizontal",
                                         command=self.root_canvas.xview)
        self.root_canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        self.root_canvas.grid(row=0, column=0, sticky='nsew')
        self.v_scrollbar.grid(row=0, column=1, sticky='ns')
        self.h_scrollbar.grid(row=1, column=0, sticky='ew')

        self.app_content_frame = ttk.Frame(self.root_canvas, padding=(5, 5))
        self.root_canvas.create_window((0, 0), window=self.app_content_frame, anchor="nw", tags="app_content_frame_tag")
        self.app_content_frame.bind("<Configure>", self._on_app_content_frame_configure)

        # Configure 3-Column Layout for app_content_frame
        self.app_content_frame.grid_columnconfigure(0, weight=1)  # Column 1 ratio
        self.app_content_frame.grid_columnconfigure(1, weight=2)  # Column 2 ratio
        self.app_content_frame.grid_columnconfigure(2, weight=1)  # Column 3 ratio
        self.app_content_frame.grid_rowconfigure(0, weight=1)  # Single row expands vertically

        # --- Initialize variables (based on your last provided __init__) ---
        self.objects_3d = []

        self.canvas_width, self.canvas_height = 1024, 768  # Your values
        fx_init = self.canvas_width * 1.0  # Adjusted for potentially less distortion, or your preference
        fy_init = self.canvas_width * 1.0  # Often fx=fy for square pixels effect, adjust height if aspect different
        cx_init, cy_init = self.canvas_width / 2.0, self.canvas_height / 2.0
        self.fx_direct_val = tk.DoubleVar(value=fx_init)
        self.fy_direct_val = tk.DoubleVar(value=fy_init)
        self.cx_val = tk.DoubleVar(value=cx_init)
        self.cy_val = tk.DoubleVar(value=cy_init)
        self.s_val = tk.DoubleVar(value=0.0)
        self.K_intrinsic = create_intrinsic_matrix(fx=self.fx_direct_val.get(), fy=self.fy_direct_val.get(),
                                                   cx=self.cx_val.get(), cy=self.cy_val.get(), s=self.s_val.get())
        self.aperture = tk.DoubleVar(value=5.6)
        self.focal_length_mm_var = tk.DoubleVar(value=8.0)  # Example
        self.pixel_width_micron_var = tk.DoubleVar(value=1.6)
        self.pixel_height_micron_var = tk.DoubleVar(value=1.6)
        self.intrinsic_input_mode_var = tk.StringVar(value="physical_params")
        self.camera_pos_vars = {'x': tk.DoubleVar(value=0.0), 'y': tk.DoubleVar(value=0.0),
                                'z': tk.DoubleVar(value=100.0)}
        self.camera_rot_vars = {'rx': tk.DoubleVar(value=180.0), 'ry': tk.DoubleVar(value=0.0),
                                'rz': tk.DoubleVar(value=0.0)}
        self.camera_transform_configs = {'x': (-2000, 2000, 5), 'y': (-2000, 2000, 5), 'z': (0, 5000, 5),
                                         'rx': (-360, 360, 5), 'ry': (-360, 360, 5), 'rz': (-360, 360, 5)}
        self.object_position_offset = {'z': tk.DoubleVar(value=0.0)}
        self.last_mouse_x, self.last_mouse_y, self.dragging_mode, self.active_object_for_drag = 0, 0, None, None
        self.debug_mode_var = tk.BooleanVar(value=False)
        self.obj_transform_vars = {'tx': tk.DoubleVar(value=0.0), 'ty': tk.DoubleVar(value=0.0), 'tz': tk.DoubleVar(value=0.0),
                                   'rx': tk.DoubleVar(value=0.0), 'ry': tk.DoubleVar(value=0.0), 'rz': tk.DoubleVar(value=0.0),
                                   'sx': tk.DoubleVar(value=1.0), 'sy': tk.DoubleVar(value=1.0), 'sz': tk.DoubleVar(value=1.0)}
        self.transform_configs = {'tx': (-2000, 2000, 5), 'ty': (-2000, 2000, 5), 'tz': (-2000, 2000, 5),
                                  'rx': (-360, 360, 5), 'ry': (-360, 360, 5), 'rz': (-360, 360, 5),
                                  'sx': (0.1, 10, 0.1), 'sy': (0.1, 10, 0.1), 'sz': (0.1, 10, 0.1)}

        # --- Column Frames (Children of self.app_content_frame) ---
        # Column 1: Will contain Camera Lens, Camera Transform, Object Management
        self.column1_frame = ttk.Frame(self.app_content_frame, padding=(0, 0, 5, 0))  # Original controls will go here
        self.column1_frame.grid(row=0, column=0, sticky='nsew')

        # Column 2: Displays (2D and 3D)
        self.main_display_area = ttk.Frame(self.app_content_frame)  # This effectively IS column 2
        self.main_display_area.grid(row=0, column=1, sticky='nsew', padx=2)
        self.main_display_area.grid_rowconfigure(0, weight=1)
        self.main_display_area.grid_rowconfigure(1, weight=1)
        self.main_display_area.grid_columnconfigure(0, weight=1)

        # Column 3: Measurement, Debug, View Options
        self.column3_frame = ttk.Frame(self.app_content_frame)
        self.column3_frame.grid(row=0, column=2, sticky='nsew', padx=(2, 0))

        # --- Initialize children of self.main_display_area (Column 2) ---
        self.image_frame = ttk.LabelFrame(self.main_display_area, text="2D Projection (pixels)")
        self.view_3d_frame = ttk.LabelFrame(self.main_display_area, text="3D Scene View (mm)")  # For embedded view

        self.image_canvas = None
        self.pil_image = Image.new("RGB", (self.canvas_width, self.canvas_height), "lightgrey")
        self.draw_context = ImageDraw.Draw(self.pil_image)

        # Matplotlib Figure and Axes for the SINGLE EMBEDDED 3D view
        self.fig_3d = Figure(figsize=(5, 4), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d_agg = None  # For the embedded view
        self.current_V_view_for_3d_plot = np.eye(4)

        # UI Variables for Measurement Info
        self.obj_dims_var = tk.StringVar(value="Obj Dims (mm): N/A")
        self.pixel_coord_var = tk.StringVar(value="Cursor (px): (N/A)")
        self.measure_2d_status_var = tk.StringVar(value="2D Measure: OFF")
        self.measure_2d_x_measurement_var = tk.StringVar(value="X Est. (mm):")
        self.measure_2d_y_measurement_var = tk.StringVar(value="Y Est. (mm):")
        self.gsd_info_var = tk.StringVar(value="GSD (mm/px): N/A")
        self.measuring_2d_mode, self.measurement_points_2d = False, []
        self.measurement_line_id_2d, self.measurement_text_id_2d = None, None
        self.obj0_Zc_mm = self.camera_pos_vars['z'].get()
        self.gsdx, self.gsdy = None, None
        self.show_2d_grid_var = tk.BooleanVar(value=False)
        self.grid_spacing_px_var = tk.IntVar(value=5)
        self.current_2d_zoom_scale = 1.0
        self.min_2d_zoom = 0.1
        self.max_2d_zoom = 10.0
        self.image_on_canvas_id = None

        self._create_default_object()

        self.root_canvas.bind_all("<MouseWheel>", self._on_root_mousewheel)
        self.root_canvas.bind_all("<Button-4>", self._on_root_mousewheel)
        self.root_canvas.bind_all("<Button-5>", self._on_root_mousewheel)
        if hasattr(self, '_bind_mousewheel_recursively'):
            self._bind_mousewheel_recursively(self.app_content_frame, self.root_canvas)

        self._setup_gui()
        if self.objects_3d: self._display_object_dimensions(self.objects_3d[0])
        self._on_intrinsic_mode_change()
        self.update_simulation()

    # --- Helper methods for root scrolling (ensure these are defined) ---
    def _on_app_content_frame_configure(self, event=None):
        self.root_canvas.configure(scrollregion=self.root_canvas.bbox("all"))
        self.root_canvas.itemconfig("app_content_frame_tag", width=self.root_canvas.winfo_width())

    def _on_root_mousewheel(self, event):
        widget_under_mouse = self.root.winfo_containing(event.x_root, event.y_root)
        is_over_3d_canvas = False
        if self.canvas_3d_agg:  # Check if embedded canvas exists
            current = widget_under_mouse
            while current is not None:
                if current == self.canvas_3d_agg.get_tk_widget():
                    is_over_3d_canvas = True
                    break
                current = current.master

        if is_over_3d_canvas: return  # Let Matplotlib handle scroll on its canvas

        if event.num == 4:
            self.root_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.root_canvas.yview_scroll(1, "units")
        elif hasattr(event, 'delta') and event.delta != 0:
            self.root_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _scroll_target_canvas(self, event, target_canvas):
        # Helper to prevent trying to scroll if target_canvas is None (e.g. during setup)
        if not target_canvas: return

        # Check if the event originated directly from the target_canvas or its scrollbars
        # to prevent recursive loops or double scrolling if target_canvas also has direct binds.
        # This check can be complex. A simple guard:
        if event.widget == target_canvas:  # If event already on the target, its direct bind should handle it
            return

        if event.num == 4:
            target_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            target_canvas.yview_scroll(1, "units")
        elif hasattr(event, 'delta') and event.delta != 0:
            target_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"  # Important to prevent event propagation if we handled it

    def _physical_params_changed(self, event=None):  # Called by physical param spinboxes
        if self.intrinsic_input_mode_var.get() == "physical_params":
            self._calculate_fx_fy_from_physical()
            self.update_simulation()  # Update K_intrinsic and then the full simulation

    def _calculate_fx_fy_from_physical(self):
        """Calculates fx, fy from physical parameters and updates K_intrinsic and UI."""
        try:
            f_mm = self.focal_length_mm_var.get()
            pw_um = self.pixel_width_micron_var.get()
            ph_um = self.pixel_height_micron_var.get()

            if pw_um <= 0 or ph_um <= 0:
                self.log_debug("Pixel width/height must be positive.")
                # Optionally show error to user or revert fx, fy display
                return False

            pw_mm = pw_um / 1000.0
            ph_mm = ph_um / 1000.0

            fx_calc = f_mm / pw_mm
            fy_calc = f_mm / ph_mm

            self.K_intrinsic[0, 0] = fx_calc
            self.K_intrinsic[1, 1] = fy_calc

            # Update the StringVars tied to the K matrix Entry fields for fx and fy
            if "k_00" in self.k_entry_vars: self.k_entry_vars["k_00"].set(f"{fx_calc:.3f}")
            if "k_11" in self.k_entry_vars: self.k_entry_vars["k_11"].set(f"{fy_calc:.3f}")

            self.log_debug(f"Calculated K from physical: fx={fx_calc:.3f}, fy={fy_calc:.3f}")
            return True
        except tk.TclError:  # If spinbox values are invalid during get()
            self.log_debug("Invalid physical parameter input.")
            return False
        except ZeroDivisionError:
            self.log_debug("Error: Pixel width or height is zero during K calculation.")
            return False

    def _update_K_from_direct_entry(self, k_key):
        """Updates K_intrinsic from direct entry if in 'K_direct' mode."""
        if self.intrinsic_input_mode_var.get() != "K_direct":
            # If in physical mode, re-assert calculated values if user tries to edit fx/fy
            if k_key in ["k_00", "k_11"]:
                self._calculate_fx_fy_from_physical()  # This will rewrite the entry
            return

        if k_key not in self.k_entry_vars or k_key not in self.k_entries:
            return

        try:
            val_str = self.k_entry_vars[k_key].get()
            val_float = float(val_str)

            # Determine r, c from k_key (e.g., "k_00" -> r=0, c=0)
            r, c = int(k_key[2]), int(k_key[3])

            self.K_intrinsic[r, c] = val_float
            self.log_debug(f"K_intrinsic directly updated: K[{r},{c}] = {val_float}")

            # If fx or fy were changed directly, physical params become 'N/A' or outdated
            if k_key == "k_00" or k_key == "k_11":
                self.focal_length_mm_var.set(8)  # Or some indicator of N/A
                self.pixel_width_micron_var.set(1.6)
                self.pixel_height_micron_var.set(1.6)
                # Or disable physical params entries via _on_intrinsic_mode_change
                self.log_debug("Direct fx/fy edit; physical params are now out of sync/N/A.")

            self.update_simulation()
        except ValueError:
            self.log_debug(f"Invalid direct input for K entry {k_key}: '{val_str}'")
            # Revert entry to current K_intrinsic value if parsing fails
            r, c = int(k_key[2]), int(k_key[3])
            self.k_entry_vars[k_key].set(f"{self.K_intrinsic[r, c]:.3f}")

    def _on_intrinsic_mode_change(self, event=None):
        """Handles UI changes when intrinsic input mode is switched."""
        mode = self.intrinsic_input_mode_var.get()
        is_physical_mode = (mode == "physical_params")

        # Toggle state of physical parameter spinboxes
        for child in self.physical_params_frame.winfo_children():  # Iterate through row_frames
            for spinbox_or_label in child.winfo_children():  # Iterate through label and spinbox
                if isinstance(spinbox_or_label, ttk.Spinbox):
                    spinbox_or_label.config(state=tk.NORMAL if is_physical_mode else tk.DISABLED)

        # Toggle state of fx, fy entries in K matrix
        if "k_00" in self.k_entries: self.k_entries["k_00"].config(state=tk.DISABLED if is_physical_mode else tk.NORMAL)
        if "k_11" in self.k_entries: self.k_entries["k_11"].config(state=tk.DISABLED if is_physical_mode else tk.NORMAL)

        if is_physical_mode:
            self.log_debug("Switched to Physical Parameters mode for K.")
            self._calculate_fx_fy_from_physical()  # Calculate and display fx, fy
        else:  # K_direct mode
            self.log_debug("Switched to Direct K Matrix Input mode.")
            # User can now edit fx, fy. Update K_intrinsic from current entry values.
            # This ensures K_intrinsic reflects what's shown in the (now editable) fx, fy entries.
            try:
                self.K_intrinsic[0, 0] = float(self.k_entry_vars["k_00"].get())
                self.K_intrinsic[1, 1] = float(self.k_entry_vars["k_11"].get())
            except:  # Handle case where entries might not be valid floats yet
                self.log_debug("Could not parse direct fx/fy on mode switch, K might be stale.")

        self.update_simulation()  # Refresh based on current K

    # And _bind_mousewheel_recursively if you choose to use it more widely for the root canvas
    # (The version binding to a *specific target_canvas* is needed)
    def _bind_mousewheel_recursively(self, widget, target_canvas):
        def _scroll_target(event):
            # Similar logic to _on_root_mousewheel to decide IF to scroll target_canvas
            # This needs to be context-aware. For now, just scroll the target.
            if event.num == 4:
                target_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                target_canvas.yview_scroll(1, "units")
            elif hasattr(event, 'delta') and event.delta != 0:
                target_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        widget.bind('<MouseWheel>', lambda e, tc=target_canvas: _scroll_target(e), add='+')
        widget.bind('<Button-4>', lambda e, tc=target_canvas: _scroll_target(e), add='+')
        widget.bind('<Button-5>', lambda e, tc=target_canvas: _scroll_target(e), add='+')

    def _create_default_object(self):
        v = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1],
             [1, 1, 1], [0, 1, 1]]
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

            actual_obj0_Zc_for_gsd = self.obj0_Zc_mm - self.object_position_offset['z'].get()
            fx, fy = self.K_intrinsic[0, 0], self.K_intrinsic[1, 1]
            gsdx = actual_obj0_Zc_for_gsd / fx if abs(fx) > 1e-6 else float('inf')
            gsdy = actual_obj0_Zc_for_gsd / fy if abs(fy) > 1e-6 else float('inf')

            self.measure_2d_status_var.set(f"Measured: {txt}. Click 1st for new.")
            self.measure_2d_x_measurement_var.set(f"Measured: {dist * gsdx : .4f} mm if along x-axis")
            self.measure_2d_y_measurement_var.set(f"Measured: {dist * gsdy : .4f} mm if along y-axis")
            self.measurement_points_2d = []

    def _update_offset(self):
        self.update_simulation()

    def _save_2d_projection_as_image(self):
        self.log_debug("Save 2D Projection with anti-aliasing requested...")

        suggested_filename = self._generate_descriptive_filename()
        filepath = filedialog.asksaveasfilename(
            initialfile=suggested_filename, defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp"), ("GIF", "*.gif"), ("All", "*.*")],
            title="Save Anti-aliased 2D Projection As..."
        )

        if not filepath:
            self.log_debug("Save 2D projection cancelled by user.")
            return

        self.log_debug(f"Preparing to save anti-aliased image to: {filepath}")

        # --- 1. Define Upscale Parameters ---
        upscale_factor = self.save_image_upscale_factor
        target_canvas_width = self.canvas_width  # Original target width
        target_canvas_height = self.canvas_height  # Original target height

        hires_width = target_canvas_width * upscale_factor
        hires_height = target_canvas_height * upscale_factor

        # --- 2. Create High-Resolution Temporary Image and Draw Context ---
        hires_pil_image = Image.new("RGB", (hires_width, hires_height), "lightgrey")  # Match your background
        hires_draw_context = ImageDraw.Draw(hires_pil_image)
        self.log_debug(f"Created temporary hires image: {hires_width}x{hires_height}")

        # --- 3. Get All Current Simulation Parameters Needed for Rendering ---
        # (These are the same parameters fetched at the start of your update_simulation)
        cam_p = np.array([self.camera_pos_vars[k].get() for k in ['x', 'y', 'z']])
        cam_r_deg = np.array([self.camera_rot_vars[k].get() for k in ['rx', 'ry', 'rz']])

        Rrz = create_rotation_matrix_z(math.radians(cam_r_deg[2]))
        Rry = create_rotation_matrix_y(math.radians(cam_r_deg[1]))
        Rrx = create_rotation_matrix_x(math.radians(cam_r_deg[0]))
        R_cam_world = Rrz @ Rry @ Rrx

        cam_fwd_loc_h = np.array([0, 0, -1, 0]);
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

        target_dist_mm_save = self.obj0_Zc_mm
        # Use a consistent way to get target distance for V_view, similar to update_simulation
        # This might involve using self.obj0_Zc_mm if it's reliably updated.
        # For simplicity in this standalone function, let's assume a fixed look_at_distance or use a robust way to get current object depth.
        if self.objects_3d and hasattr(self, 'obj0_Zc_mm') and self.obj0_Zc_mm is not None and self.obj0_Zc_mm > 0:
            current_obj0_Zc_mm_for_target = self.obj0_Zc_mm  # Use the last calculated if available
            # Re-calculate obj0_Zc_mm based on current cam_p and object for highest accuracy for focal plane
            temp_V_view_for_Zc = create_view_matrix(cam_p, cam_p + world_fwd * 50.0, world_up)  # Temp V_view
            obj0 = self.objects_3d[0];
            M0 = obj0.get_model_matrix();
            MV0 = temp_V_view_for_Zc @ M0
            obj0_orig_cam_h = MV0 @ np.array([0, 0, 0, 1])
            calculated_obj0_Zc_mm = obj0_orig_cam_h[2] / obj0_orig_cam_h[3] if abs(obj0_orig_cam_h[3]) > 1e-9 else \
            obj0_orig_cam_h[2]
            if calculated_obj0_Zc_mm > 0: target_dist_mm_save = calculated_obj0_Zc_mm

        cam_target_w_save = cam_p + world_fwd * target_dist_mm_save
        V_view_save = create_view_matrix(cam_p, cam_target_w_save, world_up)

        K_intrinsic_save = self.K_intrinsic  # Current K

        target_h_dof_save = np.append(cam_target_w_save, 1.0);
        focal_pt_cam_h_save = V_view_save @ target_h_dof_save
        focal_plane_Zc_mm_save = focal_pt_cam_h_save[2] / focal_pt_cam_h_save[3] if abs(
            focal_pt_cam_h_save[3]) > 1e-9 else focal_pt_cam_h_save[2]
        current_f_stop_save = self.aperture.get()

        # --- 5. Re-draw Objects onto hires_draw_context (Scaled) ---
        if self.objects_3d:
            light_dir_w = np.array([0.6, 0.7, 1.0]);
            light_dir_w /= np.linalg.norm(light_dir_w);
            amb = 0.9
            all_faces_to_draw_hires = []

            for obj_i, obj in enumerate(self.objects_3d):
                M = obj.get_model_matrix()
                all_v_loc_h = obj.vertices_local
                all_v_world_h = (M @ all_v_loc_h.T).T
                all_v_cam_h = (V_view_save @ all_v_world_h.T).T  # Use V_view_save
                all_v_world = all_v_world_h[:, :3] / np.maximum(all_v_world_h[:, 3, np.newaxis], 1e-9)

                if not obj.faces: continue
                for face_j, face_indices in enumerate(obj.faces):
                    if len(face_indices) < 3: continue
                    face_v_w = [all_v_world[idx] for idx in face_indices]
                    face_v_cam_h_cf = [all_v_cam_h[idx] for idx in face_indices]

                    v0w, v1w, v2w = face_v_w[0], face_v_w[1], face_v_w[2]
                    norm_w = np.cross(v1w - v0w, v2w - v0w)
                    if np.linalg.norm(norm_w) < 1e-6: continue
                    norm_w /= np.linalg.norm(norm_w)
                    center_w = np.mean(np.array(face_v_w), axis=0)
                    view_to_face_w = center_w - cam_p  # Use cam_p (current camera position for save)
                    if np.dot(norm_w, view_to_face_w) >= -0.01: continue

                    diff_int = max(0, np.dot(norm_w, light_dir_w));
                    intensity = amb + (1 - amb) * diff_int
                    base_rgb = obj.get_face_color_rgb_int(face_j);
                    s_rgb_lit = tuple(min(255, int(c * intensity)) for c in base_rgb)

                    orig_scr_pts, face_Zc_mm_vals, valid_proj = [], [], True
                    for vch in face_v_cam_h_cf:
                        Xc, Yc, Zc, Wc = vch
                        if abs(Wc) > 1e-9:
                            Xc /= Wc
                            Yc /= Wc
                            Zc /= Wc
                        else:
                            valid_proj = False;break
                        face_Zc_mm_vals.append(Zc)
                        if Zc <= 0.01: valid_proj = False;break
                        uvw_p = K_intrinsic_save @ np.array([Xc, Yc, Zc])  # Use K_intrinsic_save
                        if abs(uvw_p[2]) < 1e-6: valid_proj = False;break
                        orig_scr_pts.append((uvw_p[0] / uvw_p[2], uvw_p[1] / uvw_p[2]))  # These are 1x scale points

                    if not valid_proj or len(orig_scr_pts) < 3: continue
                    avg_Zc_mm_face = np.mean(face_Zc_mm_vals)
                    final_face_rgb = list(s_rgb_lit)

                    # Apply DoF effect
                    if current_f_stop_save < 22.0:
                        abs_focal_plane_dist_mm = abs(focal_plane_Zc_mm_save);
                        abs_face_z_dist_mm = abs(avg_Zc_mm_face)
                        dof_sharp_factor = max(0.05, min(2.5, current_f_stop_save / 16.0))
                        sharp_mm = abs_focal_plane_dist_mm * 0.10 * dof_sharp_factor
                        sharp_mm = max(1.0, min(sharp_mm, abs_focal_plane_dist_mm * 0.75))
                        diff_mm = abs(abs_face_z_dist_mm - abs_focal_plane_dist_mm)
                        if diff_mm > sharp_mm:
                            oof_mm = diff_mm - sharp_mm
                            trans_dist_mm = (abs_focal_plane_dist_mm * 0.5 + sharp_mm) / max(0.1, dof_sharp_factor)
                            trans_dist_mm = max(2.0, trans_dist_mm)
                            eff_str = min(1.0, oof_mm / trans_dist_mm)
                            dim_val = 1.0 - eff_str * 0.70
                            final_face_rgb = [min(255, max(0, int(c * dim_val))) for c in s_rgb_lit]

                    fill_hex_final = rgb_tuple_to_hex(tuple(final_face_rgb))

                    # Scale the original screen points for drawing on the hires_pil_image
                    scaled_scr_pts_for_draw = [
                        (int(round(u * upscale_factor)), int(round(v * upscale_factor)))
                        for u, v in orig_scr_pts
                    ]
                    all_faces_to_draw_hires.append((avg_Zc_mm_face, scaled_scr_pts_for_draw, fill_hex_final, None))

            all_faces_to_draw_hires.sort(key=lambda x: x[0], reverse=True)
            for _, scaled_pts, fill, outl in all_faces_to_draw_hires:
                if len(scaled_pts) >= 3:
                    hires_draw_context.polygon(scaled_pts, fill=fill, outline=outl, width=1)  # Draw on hires

        # --- 6. Downscale for Anti-aliasing ---
        if upscale_factor > 1:
            self.log_debug(
                f"Downscaling image from {hires_width}x{hires_height} to {target_canvas_width}x{target_canvas_height} for AA.")
            try:
                final_image_to_save = hires_pil_image.resize(
                    (target_canvas_width, target_canvas_height), Image.Resampling.LANCZOS
                )
            except AttributeError:  # Fallback for older Pillow versions
                final_image_to_save = hires_pil_image.resize(
                    (target_canvas_width, target_canvas_height), Image.ANTIALIAS
                )
        else:
            final_image_to_save = hires_pil_image  # No upscaling was done, save as is

        # --- 7. Save the final image ---
        try:
            final_image_to_save.save(filepath)
            self.log_debug(f"Anti-aliased 2D projection saved to: {filepath}")
            # tk.messagebox.showinfo("Save Successful", f"Anti-aliased image saved to:\n{filepath}")
        except Exception as e:
            self.log_debug(f"Error saving anti-aliased 2D projection: {e}")
            tk.messagebox.showerror("Save Error", f"Could not save image:\n{e}")

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
        # --- COLUMN 1: Primary Controls ---
        # Sections previously parented to self.controls_frame now go into self.column1_frame

        # == Camera Lens Parameters Frame ==
        cam_param_f = ttk.LabelFrame(self.column1_frame, text="Camera Lens Parameters")  # Parent is column1_frame
        cam_param_f.pack(pady=5, padx=5, fill=tk.X, anchor='n')
        # ... (All content of cam_param_f: mode_frame, physical_params_frame, intr_f, aperture, as per your _setup_gui)

        mode_frame = ttk.LabelFrame(cam_param_f, text="Intrinsic Definition Mode")
        mode_frame.pack(pady=5, fill=tk.X)
        ttk.Radiobutton(mode_frame, text="Physical Params (f-mm, pixel-µm)", variable=self.intrinsic_input_mode_var,
                        value="physical_params", command=self._on_intrinsic_mode_change).pack(anchor='w', padx=5)
        ttk.Radiobutton(mode_frame, text="Direct K Matrix (fx, fy in pixels)", variable=self.intrinsic_input_mode_var,
                        value="K_direct", command=self._on_intrinsic_mode_change).pack(anchor='w', padx=5)
        self.physical_params_frame = ttk.LabelFrame(cam_param_f, text="Physical Parameters")
        self.physical_params_frame.pack(pady=5, fill=tk.X)
        phys_labels = ["Focal Length (mm):", "Pixel Width (µm):", "Pixel Height (µm):"]
        phys_vars = [self.focal_length_mm_var, self.pixel_width_micron_var, self.pixel_height_micron_var]
        phys_defaults = [8.0, 1.6, 1.6]
        phys_configs = [(1.0, 500.0, 1.0), (0.1, 50.0, 0.1), (0.1, 50.0, 0.1)]

        # Custom styling to indicate primitive variables
        imp_spinbox_style = ttk.Style()
        imp_spinbox_style.configure("Important.TSpinbox",
                        foreground= "orange" if darkdetect.isDark() else 'red',  # Text color
                        padding=5)

        for i, label_text in enumerate(phys_labels):
            rf = ttk.Frame(self.physical_params_frame)
            rf.pack(fill=tk.X, padx=5, pady=1)
            ttk.Label(rf, text=label_text, width=18, anchor='w').pack(side=tk.LEFT)
            if phys_vars[i].get() == 0.0 and phys_defaults[i] != 0.0: phys_vars[i].set(phys_defaults[i])
            ttk.Spinbox(rf, from_=phys_configs[i][0], to=phys_configs[i][1], increment=phys_configs[i][2],
                        textvariable=phys_vars[i], width=10, command=self._physical_params_changed, style='Important.TSpinbox').pack(side=tk.LEFT,
                                                                                                         expand=True,
                                                                                                         fill=tk.X)
        self.intr_f = ttk.LabelFrame(cam_param_f, text="Intrinsic Matrix K (Calculated or Direct)")
        self.intr_f.pack(pady=5, fill=tk.X)
        self.k_entries = {}
        self.k_entry_vars = {}
        k_map = {"k_00": (0, 0, "fx", self.fx_direct_val, True), "k_01": (0, 1, "s", self.s_val, True),
                 "k_02": (0, 2, "cx", self.cx_val, True),
                 "k_10": (1, 0, "0", None, False), "k_11": (1, 1, "fy", self.fy_direct_val, True),
                 "k_12": (1, 2, "cy", self.cy_val, True),
                 "k_20": (2, 0, "0", None, False), "k_21": (2, 1, "0", None, False), "k_22": (2, 2, "1", None, False)}
        for key, (r, c, lbl, var, always_edit) in k_map.items():
            is_maj = key in ["k_00", "k_01", "k_02", "k_11", "k_12"]
            suffix = ": " if (is_maj and always_edit) or key in ["k_00", "k_11"] else ("  " if not is_maj else ": ")
            if lbl == "0" or lbl == "1":
                suffix = "  "
            ttk.Label(self.intr_f, text=lbl + suffix).grid(row=r, column=2 * c, padx=(5, 0), pady=2, sticky='w')
            if var:
                ev = tk.StringVar(value=f"{var.get():.2f}")
                self.k_entry_vars[key] = ev
                e = ttk.Entry(self.intr_f, width=8, textvariable=ev, style='Important.TSpinbox' if c==2 else '')
                e.grid(row=r, column=2 * c + 1, padx=(0, 5), pady=2, sticky='ew')
                if always_edit or key in ["k_00", "k_11"]:
                    e.bind("<FocusOut>", lambda ev, kb=key: self._update_K_from_direct_entry(kb))
                    e.bind("<Return>", lambda ev, kb=key: self._update_K_from_direct_entry(kb))
                self.k_entries[key] = e
            else:
                ttk.Label(self.intr_f, text=("0.0" if lbl == "0" else "1.0")).grid(row=r, column=2 * c + 1, padx=(0, 5),
                                                                                   pady=2, sticky='w')
        ttk.Label(cam_param_f, text="Aperture (f-number):").pack(anchor='w', padx=5, pady=(5, 0))
        ttk.Scale(cam_param_f, from_=1.0, to_=32.0, orient=tk.HORIZONTAL, variable=self.aperture,
                  command=lambda e: self.update_simulation()).pack(fill=tk.X, padx=5)
        self.ap_label = ttk.Label(cam_param_f, text=f"{self.aperture.get():.1f}")
        self.ap_label.pack(anchor='e', padx=5)
        self.aperture.trace_add("write", lambda *a: self.ap_label.config(text=f"{self.aperture.get():.1f}"))

        # == Camera Transform Frame == (Parent: column1_frame)
        cam_tf_f = ttk.LabelFrame(self.column1_frame, text="Camera Transform (Pos mm, Rot deg)")
        cam_tf_f.pack(pady=5, padx=5, fill=tk.X, anchor='n')
        # (Populate with pos_frame and rot_frame as in your provided _setup_gui)
        pos_frame = ttk.Frame(cam_tf_f)
        pos_frame.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        rot_frame = ttk.Frame(cam_tf_f)
        rot_frame.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        cam_tf_f.grid_columnconfigure(0, weight=1)
        cam_tf_f.grid_columnconfigure(1, weight=1)
        cam_pos_labs = {'x': "Pos X:", 'y': "Pos Y:", 'z': "Pos Z:"}
        ttk.Label(pos_frame, text="Position (mm):").grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 2))
        for i, (k, t) in enumerate(cam_pos_labs.items()):
            ttk.Label(pos_frame, text=t).grid(row=i + 1, column=0, sticky='w', pady=1, padx=(0, 2))
            cfg = self.camera_transform_configs[k]
            ttk.Spinbox(pos_frame, from_=cfg[0], to=cfg[1], increment=cfg[2], textvariable=self.camera_pos_vars[k],
                        width=7, command=self.update_simulation, style='Important.TSpinbox').grid(row=i + 1, column=1, sticky='ew', pady=1)
        cam_rot_labs = {'rx': "PitchX°:", 'ry': "YawY°:", 'rz': "RollZ°:"}
        ttk.Label(rot_frame, text="Orientation (deg):").grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 2))
        for i, (k, t) in enumerate(cam_rot_labs.items()):
            ttk.Label(rot_frame, text=t).grid(row=i + 1, column=0, sticky='w', pady=1, padx=(0, 2))
            cfg = self.camera_transform_configs[k]
            ttk.Spinbox(rot_frame, from_=cfg[0], to=cfg[1], increment=cfg[2], textvariable=self.camera_rot_vars[k],
                        width=7, command=self.update_simulation,  style='Important.TSpinbox').grid(row=i + 1, column=1, sticky='ew', pady=1)

        # == Object Management Frame == (Parent: column1_frame)
        obj_mgmt_f = ttk.LabelFrame(self.column1_frame, text="Object (Vertices in mm)")  # Using obj_mgmt_f as var
        obj_mgmt_f.pack(pady=5, padx=5, fill=tk.X, anchor='n')
        # (Populate with Load button, dims_info_f, obj_tf_f as in your provided _setup_gui)
        ttk.Button(obj_mgmt_f, text="Load (.obj)", command=self.load_object).pack(pady=(5, 0), padx=5, fill=tk.X)
        dims_info_f = ttk.LabelFrame(obj_mgmt_f, text="Object Info")
        dims_info_f.pack(pady=5, padx=5, fill=tk.X)
        ttk.Label(dims_info_f, textvariable=self.obj_dims_var, justify=tk.LEFT).pack(padx=5, pady=5, fill=tk.X)
        obj_tf_f = ttk.LabelFrame(obj_mgmt_f, text="Object Transform (Translate mm)")
        obj_tf_f.pack(pady=5, padx=5, fill=tk.X)
        obj_tf_labs = {'tx': "TrX:", 'ty': "TrY:", 'tz': "TrZ:", 'rx': "RotX°:", 'ry': "RotY°:", 'rz': "RotZ°:",
                       'sx': "ScX:", 'sy': "ScY:", 'sz': "ScZ:"}
        for i, (k, t) in enumerate(obj_tf_labs.items()):
            r_obj, c_obj = divmod(i, 3)
            ttk.Label(obj_tf_f, text=t).grid(row=r_obj, column=c_obj * 2, sticky='w', padx=(5, 0), pady=1)
            cfg = self.transform_configs[k]
            ttk.Spinbox(obj_tf_f, from_=cfg[0], to=cfg[1], increment=cfg[2], textvariable=self.obj_transform_vars[k],
                        width=6, command=self._update_object_transform, style='Important.TSpinbox').grid(row=r_obj, column=c_obj * 2 + 1,
                                                                             sticky='ew', padx=(0, 5), pady=1)
            obj_tf_f.grid_columnconfigure(c_obj * 2 + 1, weight=1)

        # --- COLUMN 2: Display Areas ---
        # self.image_frame and self.view_3d_frame are children of self.column2_displays_frame (which IS self.main_display_area)
        # Their .grid() calls place them within self.column2_displays_frame
        self.image_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 5))
        self.image_canvas = tk.Canvas(self.image_frame, width=self.canvas_width, height=self.canvas_height,
                                      bg="darkgrey", highlightthickness=0)  # Changed from lightgrey to darkgrey
        self.image_canvas.grid(row=0, column=0, sticky='nsew')
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_canvas_v_scroll = ttk.Scrollbar(self.image_frame, orient="vertical", command=self.image_canvas.yview)
        self.image_canvas_v_scroll.grid(row=0, column=1, sticky='ns')
        self.image_canvas_h_scroll = ttk.Scrollbar(self.image_frame, orient="horizontal",
                                                   command=self.image_canvas.xview)
        self.image_canvas_h_scroll.grid(row=1, column=0, sticky='ew')
        self.image_canvas.configure(yscrollcommand=self.image_canvas_v_scroll.set,
                                    xscrollcommand=self.image_canvas_h_scroll.set)

        bottom_bar_2d = ttk.Frame(self.image_frame)
        bottom_bar_2d.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(2, 0))

        # Grid controls are now in Column 3. Save button and pixel coord label remain here.
        ttk.Button(bottom_bar_2d, text="Save 2D Projection", command=self._save_2d_projection_as_image).pack(
            side=tk.LEFT, padx=5, pady=2)
        ttk.Label(bottom_bar_2d, textvariable=self.pixel_coord_var).pack(side=tk.RIGHT, padx=5, pady=2)

        # Bindings for image_canvas (as per your setup)
        self.image_canvas.bind("<Motion>", self._on_mouse_hover_2d_canvas)
        self.image_canvas.bind("<Leave>", self._on_mouse_leave_2d_canvas)
        self.image_canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.image_canvas.bind("<ButtonPress-3>", self._on_mouse_press)
        self.image_canvas.bind("<B1-Motion>", self._on_mouse_motion)
        self.image_canvas.bind("<B3-Motion>", self._on_mouse_motion)
        self.image_canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.image_canvas.bind("<ButtonRelease-3>", self._on_mouse_release)

        # Embedded 3D Scene View (uses self.fig_3d for the single embedded view)
        self.view_3d_frame.grid(row=1, column=0, sticky='nsew', pady=(5, 0))
        self.canvas_3d_agg = FigureCanvasTkAgg(self.fig_3d, master=self.view_3d_frame)  # Use self.fig_3d
        canvas_widget_3d_emb = self.canvas_3d_agg.get_tk_widget()
        canvas_widget_3d_emb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar_f_3d_emb = ttk.Frame(self.view_3d_frame)
        toolbar_f_3d_emb.pack(side=tk.BOTTOM, fill=tk.X, expand=False)
        toolbar_emb = NavigationToolbar2Tk(self.canvas_3d_agg, toolbar_f_3d_emb)
        toolbar_emb.update()
        try:
            self.fig_3d.tight_layout()  # Use self.fig_3d
        except Exception as e:
            self.log_debug(f"Note: fig_3d.tight_layout() failed: {e}")

        # --- COLUMN 3: Measurement, Debug, View Options ---
        col3_parent = self.column3_frame

        # == Measurement Tools Frame == (Parent: col3_parent)
        measure_f = ttk.LabelFrame(col3_parent, text="Measurement Tools")
        measure_f.pack(pady=5, padx=5, fill=tk.X, anchor='n')
        # (Populate with 2D measure button, status labels, GSD, Offset-Z as in your provided _setup_gui)
        ttk.Button(measure_f, text="Measure 2D Dist (px)", command=self._toggle_measure_2d_mode).pack(pady=(5, 0),
                                                                                                      padx=5, fill=tk.X)
        ttk.Label(measure_f, textvariable=self.measure_2d_status_var).pack(padx=5)
        ttk.Label(measure_f, textvariable=self.measure_2d_x_measurement_var).pack(padx=5)
        ttk.Label(measure_f, textvariable=self.measure_2d_y_measurement_var).pack(padx=5)
        ttk.Label(measure_f, textvariable=self.gsd_info_var, justify=tk.LEFT).pack(pady=5, padx=5, fill=tk.X)
        offset_z_frame = ttk.Frame(measure_f)
        offset_z_frame.pack(pady=(5, 0), padx=5, fill=tk.X)
        ttk.Label(offset_z_frame, text="GSD Obj Z-Offset(mm):").grid(row=0, column=0, sticky='w', padx=(0, 5))
        # Assuming self._update_offset exists or use self.update_simulation
        ttk.Spinbox(offset_z_frame, from_=-1000, to=1000, increment=1, textvariable=self.object_position_offset['z'],
                    width=6, command=getattr(self, '_update_offset', self.update_simulation)).grid(row=0, column=1,
                                                                                                   sticky='ew')
        offset_z_frame.grid_columnconfigure(1, weight=1)

        # == Debug Frame == (Parent: col3_parent)
        debug_f = ttk.LabelFrame(col3_parent, text="Debugging")
        debug_f.pack(pady=5, padx=5, fill=tk.X, anchor='n')
        ttk.Checkbutton(debug_f, text="Enable Debug Log", variable=self.debug_mode_var,
                        command=self._on_debug_toggle).pack(pady=5, padx=5, anchor='w')

        # == View Options Frame == (Parent: col3_parent)
        view_options_f = ttk.LabelFrame(col3_parent, text="2D View Options")  # Renamed for clarity
        view_options_f.pack(pady=5, padx=5, fill=tk.X, anchor='n')

        grid_controls_subframe = ttk.Frame(view_options_f)
        grid_controls_subframe.pack(fill=tk.X, pady=(5, 0), padx=0)
        grid_checkbutton = ttk.Checkbutton(grid_controls_subframe, text="Show Pixel Grid",
                                           variable=self.show_2d_grid_var, command=self.update_simulation)
        grid_checkbutton.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(grid_controls_subframe, text="Spacing(px):").pack(side=tk.LEFT, padx=(0, 2))
        grid_spacing_spinbox = ttk.Spinbox(grid_controls_subframe, from_=2, to=max(2, self.canvas_width // 4),
                                           increment=1, textvariable=self.grid_spacing_px_var,
                                           width=5, command=self.update_simulation)
        grid_spacing_spinbox.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 0))

        # Remove "Toggle Separate 3D View" button as per your request to discard that idea
        # if hasattr(self, '_toggle_3d_window'):
        #     open_sep_3d_button = ttk.Button(view_options_f, text="Toggle Separate 3D View", command=self._toggle_3d_window)
        #     open_sep_3d_button.pack(pady=5, fill=tk.X)

        # Initial update of aperture label
        if hasattr(self, 'ap_label'): self.ap_label.config(text=f"{self.aperture.get():.1f}")

    def _on_debug_toggle(self):
        self.log_debug(f"Debug mode: {self.debug_mode_var.get()}")

    def log_debug(self, msg):
        if ENABLE_DEBUG:
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
        all_plot_points = []

        # --- 1. Draw 3D Objects (as before) ---
        for obj_idx, obj in enumerate(self.objects_3d):
            model_matrix = obj.get_model_matrix()
            world_vertices_h = (model_matrix @ obj.vertices_local.T).T
            world_vertices = world_vertices_h[:, :3] / np.maximum(world_vertices_h[:, 3, np.newaxis], 1e-9)
            if world_vertices.size > 0: all_plot_points.extend(world_vertices.tolist())

            if obj.faces:
                # ... (Poly3DCollection logic for object faces as in your last working version) ...
                face_vertex_list_for_collection = []
                face_colors_for_collection = []
                for i, face_indices in enumerate(obj.faces):
                    if not all(0 <= idx < len(world_vertices) for idx in face_indices) or len(face_indices) < 3:
                        continue
                    face_verts_3d = [world_vertices[idx] for idx in face_indices]
                    face_vertex_list_for_collection.append(face_verts_3d)
                    base_rgb_int = obj.get_face_color_rgb_int(i)
                    matplotlib_color = tuple(c / 255.0 for c in base_rgb_int)
                    face_colors_for_collection.append(matplotlib_color)
                if face_vertex_list_for_collection:
                    poly_collection = Poly3DCollection(face_vertex_list_for_collection,
                                                       facecolors=face_colors_for_collection,
                                                       linewidths=0.3, edgecolors='dimgray', alpha=1.0)
                    self.ax_3d.add_collection3d(poly_collection)

            elif obj.edges:  # Fallback wireframe for 3D view if no faces
                for edge in obj.edges:
                    p1_idx, p2_idx = edge
                    if 0 <= p1_idx < len(world_vertices) and 0 <= p2_idx < len(world_vertices):
                        pt1, pt2 = world_vertices[p1_idx], world_vertices[p2_idx]
                        self.ax_3d.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]],
                                        color=rgb_tuple_to_hex(obj.default_rgb_int), lw=1.0)

        # --- 2. Camera Representation & Debug Visuals ---
        cam_p_w = np.array([self.camera_pos_vars[k].get() for k in ['x', 'y', 'z']])
        if not np.any(np.isnan(cam_p_w)): all_plot_points.append(cam_p_w.tolist())

        R_wc = self.current_V_view_for_3d_plot[:3, :3]
        R_cw = R_wc.T

        cam_x_axis_world = R_cw[:, 0]
        cam_y_axis_world = R_cw[:, 1]
        cam_look_dir_world = R_cw[:, 2]

        scl_ref_pt = np.mean([p for p in all_plot_points if p is not None and not np.any(np.isnan(p)) and len(p) == 3],
                             axis=0) if len(all_plot_points) > 1 else cam_p_w
        dist_scl = np.linalg.norm(cam_p_w - scl_ref_pt)
        if dist_scl < 1e-1: dist_scl = np.linalg.norm(cam_p_w)
        if dist_scl < 1e-1: dist_scl = 50.0

        cone_h = max(10.0, dist_scl * 0.10)
        cone_r = cone_h * 0.4
        cone_v, cone_e = create_cone_wireframe(cam_p_w, cam_look_dir_world, cone_h, cone_r, 8)
        if cone_v.size > 0: all_plot_points.extend(cone_v.tolist())
        for edge in cone_e:
            p1, p2 = cone_v[edge[0]], cone_v[edge[1]]
            self.ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='darkorchid', lw=1)

        # --- 3. Debug Mode Visuals ---
        if self.debug_mode_var.get():
            self.log_debug("Updating 3D debug visuals (axes, frustum)...")

            # == 3a. World Coordinate Axes == (as before)
            world_axis_length = max(5.0, dist_scl * 0.25)
            self.ax_3d.quiver(0, 0, 0, world_axis_length, 0, 0, color='#FF6666', arrow_length_ratio=0.1,
                              label='World X')
            self.ax_3d.text(world_axis_length * 1.1, 0, 0, "Xw", color='#FF6666')
            self.ax_3d.quiver(0, 0, 0, 0, world_axis_length, 0, color='#66FF66', arrow_length_ratio=0.1,
                              label='World Y')
            self.ax_3d.text(0, world_axis_length * 1.1, 0, "Yw", color='#66FF66')
            self.ax_3d.quiver(0, 0, 0, 0, 0, world_axis_length, color='#6666FF', arrow_length_ratio=0.1,
                              label='World Z')
            self.ax_3d.text(0, 0, world_axis_length * 1.1, "Zw", color='#6666FF')
            if not self.ax_3d.get_legend():
                self.ax_3d.legend(fontsize='x-small', loc='upper left', bbox_to_anchor=(0.0, 0.90))

            # == 3b. Camera Axes Lines and Labels ==
            cam_axis_vis_len = cone_h * 3  # Make them slightly longer than cone height for visibility
            # Camera Xc (red)
            self.ax_3d.quiver(cam_p_w[0], cam_p_w[1], cam_p_w[2], cam_x_axis_world[0], cam_x_axis_world[1],
                              cam_x_axis_world[2], length=cam_axis_vis_len, color='red', arrow_length_ratio=0.15,
                              linewidth=1.5)
            self.ax_3d.text(cam_p_w[0] + cam_x_axis_world[0] * cam_axis_vis_len * 1.1,
                            cam_p_w[1] + cam_x_axis_world[1] * cam_axis_vis_len * 1.1,
                            cam_p_w[2] + cam_x_axis_world[2] * cam_axis_vis_len * 1.1, "Xc", color='red',
                            fontsize='small')
            # Camera Yc (green)
            self.ax_3d.quiver(cam_p_w[0], cam_p_w[1], cam_p_w[2], cam_y_axis_world[0], cam_y_axis_world[1],
                              cam_y_axis_world[2], length=cam_axis_vis_len, color='green', arrow_length_ratio=0.15,
                              linewidth=1.5)
            self.ax_3d.text(cam_p_w[0] + cam_y_axis_world[0] * cam_axis_vis_len * 1.1,
                            cam_p_w[1] + cam_y_axis_world[1] * cam_axis_vis_len * 1.1,
                            cam_p_w[2] + cam_y_axis_world[2] * cam_axis_vis_len * 1.1, "Yc", color='green',
                            fontsize='small')
            # Camera Zc - look direction (blue)
            self.ax_3d.quiver(cam_p_w[0], cam_p_w[1], cam_p_w[2], cam_look_dir_world[0], cam_look_dir_world[1],
                              cam_look_dir_world[2], length=cam_axis_vis_len, color='blue', arrow_length_ratio=0.15,
                              linewidth=1.5)
            self.ax_3d.text(cam_p_w[0] + cam_look_dir_world[0] * cam_axis_vis_len * 1.1,
                            cam_p_w[1] + cam_look_dir_world[1] * cam_axis_vis_len * 1.1,
                            cam_p_w[2] + cam_look_dir_world[2] * cam_axis_vis_len * 1.1, "Zc", color='blue',
                            fontsize='small')

            # == 3c. Camera FOV (Frustum) Visualization ==
            fx, fy = self.K_intrinsic[0, 0], self.K_intrinsic[1, 1]
            cx, cy = self.K_intrinsic[0, 2], self.K_intrinsic[1, 2]
            img_W, img_H = self.canvas_width, self.canvas_height

            near_vis_dist_abs = max(0.1, cone_h * 0.25)  # Near plane for frustum visualization

            pts_cam_at_z1 = [  # Defines ray directions from camera origin in camera space
                np.array([(0 - cx) / fx, (0 - cy) / fy, 1.0]),
                np.array([(img_W - cx) / fx, (0 - cy) / fy, 1.0]),
                np.array([(img_W - cx) / fx, (img_H - cy) / fy, 1.0]),
                np.array([(0 - cx) / fx, (img_H - cy) / fy, 1.0])
            ]
            near_corners_cam = [p_at_z1 * near_vis_dist_abs for p_at_z1 in pts_cam_at_z1]
            near_corners_world = [cam_p_w + R_cw @ p_cam for p_cam in near_corners_cam]
            if near_corners_world:  # Add points if they are valid
                all_plot_points.extend(
                    c.tolist() for c in near_corners_world if c is not None and not np.any(np.isnan(c)))

            frustum_color = '#888888'
            frustum_style = '--'
            frustum_lw = 0.8
            for i in range(4):  # Draw Near Plane
                p1, p2 = near_corners_world[i], near_corners_world[(i + 1) % 4]
                self.ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=frustum_color,
                                linestyle=frustum_style, lw=frustum_lw)

            # --- Far Plane Calculation: Project to Z_world=0 plane ---
            far_corners_world_on_Z0 = []  # Renamed for clarity
            # Threshold to consider camera "on the plane" Z=0
            camera_on_plane_threshold = 0.1  # e.g., 0.1 mm

            # Check if camera's Z position is very close to the target plane (Z=0)
            if abs(cam_p_w[2]) < camera_on_plane_threshold:
                self.log_debug(
                    f"Camera is near/on Z=0 plane (Z_cam={cam_p_w[2]:.2f}mm). Visualizing frustum far plane at fixed distance.")
                # Fallback: draw far plane at a fixed distance along rays, not on Z=0
                far_vis_dist_abs = max(near_vis_dist_abs * 3.0, dist_scl * 0.7)
                far_corners_cam = [p_at_z1 * far_vis_dist_abs for p_at_z1 in pts_cam_at_z1]
                far_corners_world_on_Z0 = [cam_p_w + R_cw @ p_cam for p_cam in
                                           far_corners_cam]  # Use this list name for consistency
                # Draw this fixed far plane rectangle
                for i in range(4):
                    p1, p2 = far_corners_world_on_Z0[i], far_corners_world_on_Z0[(i + 1) % 4]
                    self.ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=frustum_color,
                                    linestyle=frustum_style, lw=frustum_lw)
            else:
                self.log_debug(f"Camera Z={cam_p_w[2]:.2f}mm. Projecting frustum far plane to Z_world=0.")
                plane_N_Z0 = np.array([0, 0, 1])  # Normal of Z=0 plane
                plane_P0_z_val = 0.0  # Z-value of the Z=0 plane

                # Max distance to draw ray if intersection is very far or t is huge
                # (t * norm(ray_dir_world) = world distance)
                # Let's cap the world distance for projection lines
                max_projection_line_length = dist_scl * 10

                for p_at_z1 in pts_cam_at_z1:
                    ray_dir_world = R_cw @ p_at_z1  # World direction vector of the ray from camera eye

                    # Denominator for t: dot(ray_dir_world, plane_N_Z0) which is ray_dir_world[2]
                    denom = ray_dir_world[2]

                    if abs(denom) > 1e-6:  # Ray is not parallel to the Z=0 plane
                        # t = (plane_P0_z_val - cam_p_w[2]) / denom
                        t = -cam_p_w[2] / denom  # Simpler for Z=0 plane

                        if t > 1e-3:  # Intersection must be in front of the camera (positive t)
                            intersect_pt = cam_p_w + t * ray_dir_world

                            # Cap distance for visualization if intersection is extremely far
                            if np.linalg.norm(intersect_pt - cam_p_w) > max_projection_line_length:
                                unit_ray_dir = ray_dir_world / np.linalg.norm(ray_dir_world)
                                intersect_pt = cam_p_w + max_projection_line_length * unit_ray_dir
                                self.log_debug(
                                    f"Capping frustum ray to Z=0 plane at {max_projection_line_length:.1f}mm")

                            far_corners_world_on_Z0.append(intersect_pt)
                        else:
                            far_corners_world_on_Z0.append(None)  # Intersection behind or at camera
                            self.log_debug(
                                f"Frustum ray (world_dir_z={ray_dir_world[2]:.2f}) for Z=0: t={t:.2f} (behind/at cam)")
                    else:  # Ray parallel to Z=0 plane
                        far_corners_world_on_Z0.append(None)
                        self.log_debug(f"Frustum ray (world_dir_z={ray_dir_world[2]:.2f}) parallel to Z=0 plane.")

                # Draw polygon on Z=0 if we have 4 valid corners
                valid_far_Z0_corners = [p for p in far_corners_world_on_Z0 if p is not None]
                if len(valid_far_Z0_corners) == 4:
                    poly_on_ground_verts = [[valid_far_Z0_corners[0], valid_far_Z0_corners[1],
                                             valid_far_Z0_corners[2], valid_far_Z0_corners[3]]]
                    poly_on_ground = Poly3DCollection(poly_on_ground_verts,
                                                      facecolors=frustum_color, alpha=0.10,  # More transparent
                                                      edgecolors=frustum_color, linestyle='-',
                                                      linewidth=frustum_lw * 1.2)  # Solid line for base
                    self.ax_3d.add_collection3d(poly_on_ground)
                    if valid_far_Z0_corners:  # Add points if they are valid
                        all_plot_points.extend(
                            c.tolist() for c in valid_far_Z0_corners if c is not None and not np.any(np.isnan(c)))

            # Draw connecting lines from near to far frustum corners
            # (far_corners_world_on_Z0 replaces the generic far_corners_world from previous logic)
            for i in range(4):
                if far_corners_world_on_Z0 and i < len(far_corners_world_on_Z0) and far_corners_world_on_Z0[
                    i] is not None:
                    p_near, p_far = near_corners_world[i], far_corners_world_on_Z0[i]
                    self.ax_3d.plot([p_near[0], p_far[0]], [p_near[1], p_far[1]], [p_near[2], p_far[2]],
                                    color=frustum_color, linestyle=frustum_style, lw=frustum_lw)

            # == 3d. Camera Target Point (visualization - as before) ==
            # ... (camera target scatter plot logic) ...
            target_dist_mm_sim = self.obj0_Zc_mm
            cam_target_sim_w = cam_p_w + cam_look_dir_world * target_dist_mm_sim
            self.ax_3d.scatter(cam_target_sim_w[0], cam_target_sim_w[1], cam_target_sim_w[2],
                               c='cyan', marker='X', s=50, label='Sim Target Pt', depthshade=False, edgecolors='blue')
            if not np.any(np.isnan(cam_target_sim_w)): all_plot_points.append(cam_target_sim_w.tolist())

            # --- 4. Set Plot Limits and Labels (as before) ---
            # ... (your existing plot limits logic using valid_pts_for_lims, max_r_plot, etc.) ...
            # (Ensure all_plot_points has been populated correctly before this)
        valid_pts_for_lims = [p for p in all_plot_points if p is not None and not np.any(np.isnan(p)) and len(p) == 3]
        if valid_pts_for_lims:
            pts_arr = np.array(valid_pts_for_lims)
            min_c, max_c = pts_arr.min(axis=0), pts_arr.max(axis=0)
            rng_d = np.maximum(max_c - min_c, np.array([dist_scl * 0.1, dist_scl * 0.1, dist_scl * 0.1]))
            if np.any(rng_d < 1.0): rng_d = np.maximum(rng_d, np.array([1., 1., 1.]))
            ctr = (max_c + min_c) / 2.
            max_r_plot = np.max(rng_d) * 0.75 + max(2.0, dist_scl * 0.1)
            self.ax_3d.set_xlim(ctr[0] - max_r_plot, ctr[0] + max_r_plot)
            self.ax_3d.set_ylim(ctr[1] - max_r_plot, ctr[1] + max_r_plot)
            self.ax_3d.set_zlim(ctr[2] - max_r_plot, ctr[2] + max_r_plot)
        else:
            self.ax_3d.set_xlim([-10, 10])
            self.ax_3d.set_ylim([-10, 10])
            self.ax_3d.set_zlim([-10, 10])

        self.ax_3d.set_xlabel("World X (mm)")
        self.ax_3d.set_ylabel("World Y (mm)")
        self.ax_3d.set_zlabel("World Z (mm)")
        self.ax_3d.set_title("3D Scene View")
        try:
            self.fig_3d.tight_layout()
        except:
            pass
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

        # --- Draw Pixel Grid if Enabled ---
        if self.show_2d_grid_var.get():
            grid_color = "#A0A0A0"
            try:
                current_grid_spacing = self.grid_spacing_px_var.get()  # Corrected variable name
                if current_grid_spacing < 2:  # Ensure a minimum practical spacing
                    current_grid_spacing = 2
                    self.grid_spacing_px_var.set(current_grid_spacing)
            except tk.TclError:
                current_grid_spacing = 20  # Fallback default
                self.grid_spacing_px_var.set(current_grid_spacing)

            self.log_debug(f"Drawing 2D grid with spacing: {current_grid_spacing} px")
            if current_grid_spacing > 0:
                for x_coord in range(current_grid_spacing, self.canvas_width, current_grid_spacing):
                    self.draw_context.line([(x_coord, 0), (x_coord, self.canvas_height)], fill=grid_color, width=1)
                for y_coord in range(current_grid_spacing, self.canvas_height, current_grid_spacing):
                    self.draw_context.line([(0, y_coord), (self.canvas_width, y_coord)], fill=grid_color, width=1)

        # --- Camera Setup ---
        cam_p = np.array([self.camera_pos_vars[k].get() for k in ['x', 'y', 'z']])
        cam_r_deg = np.array([self.camera_rot_vars[k].get() for k in ['rx', 'ry', 'rz']])
        self.log_debug(f"Cam Pos(mm): {cam_p}, Rot(deg): P={cam_r_deg[0]:.1f},Y={cam_r_deg[1]:.1f},R={cam_r_deg[2]:.1f}")

        Rrz = create_rotation_matrix_z(math.radians(cam_r_deg[2]))
        Rry = create_rotation_matrix_y(math.radians(cam_r_deg[1]))
        Rrx = create_rotation_matrix_x(math.radians(cam_r_deg[0]))
        R_cam_world = Rrz @ Rry @ Rrx

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

        target_dist_mm = self.obj0_Zc_mm
        cam_target_w = cam_p + world_fwd * target_dist_mm
        V_view = create_view_matrix(cam_p, cam_target_w, world_up)
        self.current_V_view_for_3d_plot = V_view
        self.log_debug(f"SIMULATOR K_INTRINSIC being used for projection:\n{self.K_intrinsic}")  # CRITICAL DEBUG
        self.log_debug(f"V_view (world_mm to cam_mm):\n{V_view}")

        target_h_dof = np.append(cam_target_w, 1.0)
        focal_pt_cam_h = V_view @ target_h_dof
        focal_plane_Zc_mm = focal_pt_cam_h[2] / focal_pt_cam_h[3] if abs(focal_pt_cam_h[3]) > 1e-9 else focal_pt_cam_h[2]
        current_f_stop = self.aperture.get()  # Renamed from f_stop for clarity
        self.log_debug(f"FocalPlane Zc: {focal_plane_Zc_mm:.2f}mm. Aperture: f/{current_f_stop:.1f}")

        # --- GSD Calculation ---
        if self.objects_3d:
            obj0 = self.objects_3d[0]
            M0 = obj0.get_model_matrix()
            MV0 = V_view @ M0
            obj0_orig_cam_h = MV0 @ np.array([0, 0, 0, 1])
            # Storing obj0_Zc_mm as an instance variable if your other methods use it (like the screenshot GSD display)
            self.obj0_Zc_mm = obj0_orig_cam_h[2] / obj0_orig_cam_h[3] if abs(obj0_orig_cam_h[3]) > 1e-9 else \
                obj0_orig_cam_h[2]

            # Apply Z offset for GSD calculation
            actual_obj0_Zc_for_gsd = self.obj0_Zc_mm - self.object_position_offset['z'].get()

            if actual_obj0_Zc_for_gsd > 0:
                fx, fy = self.K_intrinsic[0, 0], self.K_intrinsic[1, 1]
                gsdx = actual_obj0_Zc_for_gsd / fx if abs(fx) > 1e-6 else float('inf')
                gsdy = actual_obj0_Zc_for_gsd / fy if abs(fy) > 1e-6 else float('inf')
                self.gsd_info_var.set(
                    f"GSD@Obj Zc={actual_obj0_Zc_for_gsd:.1f}mm \nX:{gsdx:.4f} mm/px \nY:{gsdy:.4f} mm/px")
            else:
                self.gsd_info_var.set(f"GSD: Obj Zc={actual_obj0_Zc_for_gsd:.1f}mm (Invalid Zc)")
        else:
            self.gsd_info_var.set("GSD (mm/px): No object")

        # --- 2D Surface Projection ---
        if not self.objects_3d:
            if self.image_canvas:
                self.tk_image = ImageTk.PhotoImage(self.pil_image)
                self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self._update_3d_view()
            return

        light_dir_w = np.array([0.6, 0.7, 1.0])
        light_dir_w /= np.linalg.norm(light_dir_w)
        amb = 0.9
        all_faces_2d = []

        for obj_i, obj in enumerate(self.objects_3d):
            self.log_debug(f"--- Processing Object {obj_i} for 2D Projection ---")
            M = obj.get_model_matrix()
            self.log_debug(f"  Obj {obj_i} Model Matrix M:\n{M}")
            all_v_loc_h = obj.vertices_local
            self.log_debug(f"  Obj {obj_i} First 3 Local Vertices (Homogeneous, mm):\n{all_v_loc_h[:3]}")

            all_v_world_h = (M @ all_v_loc_h.T).T
            self.log_debug(f"  Obj {obj_i} First 3 World Vertices (Homogeneous, mm):\n{all_v_world_h[:3]}")

            all_v_cam_h = (V_view @ all_v_world_h.T).T
            self.log_debug(f"  Obj {obj_i} First 3 Camera Vertices (Homogeneous, mm):\n{all_v_cam_h[:3]}")

            all_v_world = all_v_world_h[:, :3] / np.maximum(all_v_world_h[:, 3, np.newaxis], 1e-9)

            if not obj.faces:
                self.log_debug(f"  Obj {obj_i} no faces for 2D surf.")
                continue

            for face_j, face_indices in enumerate(obj.faces):
                if len(face_indices) < 3:
                    continue
                self.log_debug(f"  Processing Obj {obj_i}, Face {face_j}, Indices: {face_indices}")

                face_v_w = [all_v_world[idx] for idx in face_indices]
                face_v_cam_h_current_face = [all_v_cam_h[idx] for idx in face_indices]

                v0w, v1w, v2w = face_v_w[0], face_v_w[1], face_v_w[2]
                norm_w = np.cross(v1w - v0w, v2w - v0w)
                if np.linalg.norm(norm_w) < 1e-6:
                    self.log_debug("Degenerate face normal.")
                    continue
                norm_w /= np.linalg.norm(norm_w)
                center_w = np.mean(np.array(face_v_w), axis=0)
                view_to_face_w = center_w - cam_p
                dot_prod_cull = np.dot(norm_w, view_to_face_w)
                if dot_prod_cull >= -0.01:  # Back-face culling
                    self.log_debug(
                        f"    Face {face_j} culled. Normal_w: {norm_w}, ViewToFace_w: {view_to_face_w}, Dot: {dot_prod_cull:.3f}")
                    continue

                diff_int = max(0, np.dot(norm_w, light_dir_w));
                intensity = amb + (1 - amb) * diff_int
                base_rgb = obj.get_face_color_rgb_int(face_j)
                s_rgb_lit = tuple(min(255, int(c * intensity)) for c in base_rgb)

                scr_pts, face_Zc_mm_vals_for_face, valid_proj = [], [], True
                self.log_debug(f"    Projecting Face {face_j} vertices:")
                for vert_idx_in_face, vch in enumerate(face_v_cam_h_current_face):
                    Xc, Yc, Zc, Wc = vch
                    self.log_debug(
                        f"      Vert {vert_idx_in_face} CamCoords_H (mm): Xc={Xc:.2f} Yc={Yc:.2f} Zc={Zc:.2f} Wc={Wc:.2f}")
                    if abs(Wc) > 1e-9:
                        Xc /= Wc;Yc /= Wc;Zc /= Wc
                    else:
                        valid_proj = False; self.log_debug("        Wc too small, invalid projection."); break
                    face_Zc_mm_vals_for_face.append(Zc)
                    self.log_debug(
                        f"      Vert {vert_idx_in_face} CamCoords_NonH (mm): Xc={Xc:.2f} Yc={Yc:.2f} Zc={Zc:.2f}")

                    if Zc <= 0.01: valid_proj = False; self.log_debug("        Zc <= 0.01 (near clip)."); break

                    uvw_p = self.K_intrinsic @ np.array([Xc, Yc, Zc])
                    w_prime_mm = uvw_p[2]  # This is Zc_mm if K[2,2]=1
                    self.log_debug(
                        f"      Vert {vert_idx_in_face} K @ [Xc,Yc,Zc] = uvw_prime: {uvw_p}, w_prime_mm (Zc): {w_prime_mm:.2f}")

                    if abs(w_prime_mm) < 1e-6: valid_proj = False; self.log_debug(
                        "        w_prime_mm (Zc from K) too small."); break

                    u_px, v_px = uvw_p[0] / w_prime_mm, uvw_p[1] / w_prime_mm
                    self.log_debug(
                        f"      Vert {vert_idx_in_face} Projected Screen Coords (px): u={u_px:.2f}, v={v_px:.2f}")
                    scr_pts.append((int(round(u_px)), int(round(v_px))))

                if not valid_proj or len(scr_pts) < 3: self.log_debug(
                    f"    Face {face_j} projection failed or not enough points."); continue
                avg_Zc_mm_face = np.mean(face_Zc_mm_vals_for_face)
                self.log_debug(f"    Face {face_j} avg_Zc_mm: {avg_Zc_mm_face:.2f}")

                final_face_rgb = list(s_rgb_lit)
                if current_f_stop < 22.0:
                    abs_focal_plane_dist_mm = abs(focal_plane_Zc_mm)
                    abs_face_z_dist_mm = abs(avg_Zc_mm_face)
                    dof_sharpness_factor = max(0.05, min(2.5, current_f_stop / 16.0))
                    sharp_depth_range_mm = abs_focal_plane_dist_mm * 0.10 * dof_sharpness_factor
                    sharp_depth_range_mm = max(1.0, min(sharp_depth_range_mm, abs_focal_plane_dist_mm * 0.75))
                    depth_diff_mm = abs(abs_face_z_dist_mm - abs_focal_plane_dist_mm)
                    if depth_diff_mm > sharp_depth_range_mm:
                        oof_mm = depth_diff_mm - sharp_depth_range_mm
                        trans_dist_mm = (abs_focal_plane_dist_mm * 0.5 + sharp_depth_range_mm) / max(0.1,
                                                                                                     dof_sharpness_factor)
                        trans_dist_mm = max(2.0, trans_dist_mm)
                        eff_str = min(1.0, oof_mm / trans_dist_mm)
                        dim_val = 1.0 - eff_str * 0.70
                        final_face_rgb = [min(255, max(0, int(c * dim_val))) for c in s_rgb_lit]
                        self.log_debug(
                            f"    DoF Applied to Face {face_j}: EffectStr={eff_str:.2f}, DimVal={dim_val:.2f}")

                fill_hex_final = rgb_tuple_to_hex(tuple(final_face_rgb))
                all_faces_2d.append((avg_Zc_mm_face, scr_pts, fill_hex_final, None))  # Outline is None

        all_faces_2d.sort(key=lambda x: x[0], reverse=True)
        for avg_Zc, pts, fill, outl in all_faces_2d:
            if len(pts) >= 3:
                self.log_debug(f"  Drawing 2D polygon: AvgZc={avg_Zc:.2f}, Pts={pts}, Fill={fill}")
                self.draw_context.polygon(pts, fill=fill, outline=outl, width=1)

        if self.image_canvas:
            self.tk_image = ImageTk.PhotoImage(self.pil_image)
            self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self._update_3d_view()
        self.log_debug("--- SIMULATION UPDATE END ---")
