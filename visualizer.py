import streamlit as st
import xml.etree.ElementTree as ET
import yaml
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R_scipy
import plotly.graph_objects as go  # For interactive 3D plotting


# --- YAML Custom Constructor for OpenCV Matrices ---
def opencv_matrix_constructor(loader, node):
    """
    Custom YAML constructor to parse OpenCV matrix structures.
    It handles various data types and shapes, including the '2f' type for image points.
    """
    mapping = loader.construct_mapping(node, deep=True)
    mat_data = mapping.get('data', [])  # Use .get for safety
    rows = int(mapping['rows'])
    cols = int(mapping['cols'])  # 'cols' from YAML, may be interpreted differently for dt='2f'
    dt = mapping.get('dt', 'f')  # Default to float if not specified

    # Ensure mat_data is a list (handles single-element data like 'data: 0.038')
    if not isinstance(mat_data, list):
        mat_data = [mat_data]

    # Determine numpy data type based on dt
    if 'd' in dt:
        np_dtype = np.float64
    elif 'f' in dt:
        np_dtype = np.float32  # Covers 'f' and '2f' for base type
    elif 'i' in dt or 's' in dt:
        np_dtype = np.int32
    elif 'u' in dt:
        np_dtype = np.uint32
    else:
        np_dtype = np.float32  # Default

    # Handle cases where mat_data might be empty
    if not mat_data:
        if dt == '2f':
            # For empty '2f' data, shape should be (rows, 0, 2)
            return np.zeros((rows, 0, 2), dtype=np_dtype)
        else:
            # For other empty data, shape is (rows, cols)
            return np.zeros((rows, cols), dtype=np_dtype)

    try:
        # Convert data to numpy array
        numpy_array = np.array(mat_data, dtype=np_dtype)
    except ValueError as e:
        st.error(
            f"Error converting data to numpy array for matrix with dt='{dt}', rows={rows}, cols={cols}. Data: {mat_data}. Error: {e}")
        # Fallback based on dt
        if dt == '2f':
            return np.zeros((rows, 0, 2), dtype=np_dtype)
        else:
            return np.zeros((rows, cols), dtype=np_dtype)

    # If numpy_array was successfully created, then get its size.
    num_elements_in_data = numpy_array.size

    # Reshape the array
    try:
        if dt == '2f':
            if rows > 0 and num_elements_in_data > 0 and num_elements_in_data % (rows * 2) == 0:
                num_points_per_row = num_elements_in_data // (rows * 2)
                return numpy_array.reshape(rows, num_points_per_row, 2)
            elif num_elements_in_data == 0 and rows >= 0:
                return numpy_array.reshape(rows, 0, 2)
            else:
                st.error(
                    f"Error reshaping '2f' matrix. Data size {num_elements_in_data} "
                    f"is incompatible with rows={rows} for paired floats (data size must be a multiple of rows*2). "
                    f"YAML cols was {cols}."
                )
                return np.zeros((rows, 0, 2), dtype=np_dtype)
        else:
            if num_elements_in_data == rows * cols:
                return numpy_array.reshape(rows, cols)
            elif num_elements_in_data == 0 and rows >= 0:
                return np.zeros((rows, cols), dtype=np_dtype)
            else:
                st.error(
                    f"Error reshaping matrix (dt='{dt}'). Data size {num_elements_in_data} "
                    f"does not match expected rows*cols ({rows}*{cols}={rows * cols})."
                )
                return np.zeros((rows, cols), dtype=np_dtype)

    except ValueError as e:
        st.error(
            f"Generic error reshaping numpy array. dt='{dt}', rows={rows}, cols={cols}. Array size: {numpy_array.size}. Error: {e}")
        if dt == '2f':
            return np.zeros((rows, 0, 2), dtype=np_dtype)
        else:
            return np.zeros((rows, cols), dtype=np_dtype)


# Add the constructor to PyYAML. SafeLoader is generally recommended.
yaml.add_constructor('!opencv-matrix', opencv_matrix_constructor, Loader=yaml.SafeLoader)


# --- File Parsing Functions ---
def parse_input_xml(xml_file_path):
    """Parses the input.xml file to extract image paths."""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        image_paths_text = root.find('images').text
        if image_paths_text:
            image_paths = image_paths_text.strip().split('\n')
            # Filter out any empty strings that might result from split
            image_paths = [path for path in image_paths if path.strip()]
            return image_paths
        else:
            st.warning(f"No image paths found within the <images> tag in {xml_file_path}.")
            return []
    except ET.ParseError as e:
        st.error(f"Error parsing XML file {xml_file_path}: {e}")
        return []
    except FileNotFoundError:
        st.error(f"XML file not found: {xml_file_path}")
        return []
    except AttributeError:  # Handles case where 'images' tag or its text is None
        st.error(f"Error parsing XML: 'images' tag not found or is empty in {xml_file_path}.")
        return []


def load_and_parse_calib_yml(yml_file_path):
    """
    Loads and parses the calib_camera.yml file.
    Includes text preprocessing to replace '!!opencv-matrix' with '!opencv-matrix'
    to ensure compatibility with the registered custom YAML constructor.
    """
    try:
        with open(yml_file_path, 'r') as f:
            raw_content = f.read()

        processed_content = raw_content.replace('!!opencv-matrix', '!opencv-matrix')

        calib_data = yaml.load(processed_content, Loader=yaml.SafeLoader)

        if 'grid_points' in calib_data and isinstance(calib_data['grid_points'], list):
            try:
                calib_data['grid_points'] = np.array(calib_data['grid_points'], dtype=np.float32).reshape(-1, 3)
            except ValueError as e:
                st.warning(f"Could not reshape grid_points. It might have an unexpected number of elements. Error: {e}")
        return calib_data
    except FileNotFoundError:
        st.error(f"YAML file not found: {yml_file_path}")
        return None
    except yaml.YAMLError as e:
        st.error(
            f"Error parsing YAML file {yml_file_path} (after text processing): {e}\nContent snippet: {processed_content[:500]}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading/parsing YAML {yml_file_path}: {e}")
        return None


# --- Image Processing Function ---
def draw_points_on_image(image_cv, points_array, color=(0, 255, 0), radius=5, thickness=-1):
    """
    Draws points on an OpenCV image.
    image_cv: OpenCV image (numpy array BGR).
    points_array: numpy array of shape (N, 2) with (x, y) coordinates.
    Returns a new image with points drawn.
    """
    drawn_image = image_cv.copy()
    if points_array is not None and points_array.ndim == 2 and points_array.shape[1] == 2:
        for point in points_array:
            center = (int(round(point[0])), int(round(point[1])))
            cv2.circle(drawn_image, center, radius, color, thickness)
    else:
        if points_array is not None:
            st.warning(
                f"Invalid format for image points (shape: {points_array.shape if hasattr(points_array, 'shape') else 'N/A'}). Cannot draw them.")
    return drawn_image


def get_euler_angles(rvec):
    """Converts a rotation vector to Euler angles (yaw, pitch, roll) in degrees."""
    try:
        # Convert rotation vector to rotation matrix
        R_mat, _ = cv2.Rodrigues(rvec)
        # Create a Rotation object from the rotation matrix
        r = R_scipy.from_matrix(R_mat)
        euler_angles = r.as_euler('zyx', degrees=True)
        return euler_angles  # [yaw, pitch, roll]
    except Exception as e:
        st.warning(f"Could not convert rotation vector to Euler angles: {e}")
        return [None, None, None]


def plot_camera_pose_3d_plotly(rvec, tvec, K_matrix=None, target_size=0.2):
    """
    Plots the camera pose and a representation of the calibration target in an interactive 3D Plotly plot.
    rvec: Rotation vector (camera orientation relative to world)
    tvec: Translation vector (camera position relative to world)
    K_matrix: Optional camera intrinsic matrix (for frustum scaling)
    target_size: Size of the calibration target cube at the world origin.
    Returns the Plotly figure object.
    """
    if rvec is None or tvec is None:
        st.warning("Rotation or translation vector is None, cannot plot camera pose.")
        fig = go.Figure()
        fig.update_layout(title="Pose data not available",
                          scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        return fig

    try:
        R_world_to_cam, _ = cv2.Rodrigues(rvec)  # Rotation from World to Camera
        t_world_to_cam = tvec.reshape(3, 1)  # Translation from World to Camera
    except Exception as e:
        st.warning(f"Error converting rvec to R_mat for plotting: {e}")
        fig = go.Figure()
        fig.update_layout(title="Error in pose data", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        return fig

    # To get camera pose in World coordinates (position and orientation of camera frame in world frame):
    # R_cam_in_world = R_world_to_cam.T
    # t_cam_in_world = -R_world_to_cam.T @ t_world_to_cam
    R_cam_in_world = R_world_to_cam.T
    cam_center_world = (-R_world_to_cam.T @ t_world_to_cam).reshape(3)

    # Camera axes in world coordinates (columns of R_cam_in_world)
    cam_x_axis_world = R_cam_in_world[:, 0]
    cam_y_axis_world = R_cam_in_world[:, 1]
    cam_z_axis_world = R_cam_in_world[:, 2]  # Viewing direction

    # Frustum scaling factor
    f_scale = 0.5  # Default frustum scale
    if K_matrix is not None and K_matrix.shape == (3, 3):
        fx = K_matrix[0, 0]
        fy = K_matrix[1, 1]
        # Heuristic scaling based on average focal length, normalized
        # The division factor (e.g., 2000.0) is arbitrary and might need tuning
        # depending on the typical scale of your tvec components.
        # We want f_scale to be a reasonable visual size relative to the scene.
        avg_focal_length = (fx + fy) / 2.0
        # If tvec components are large (e.g., camera is far), frustum should be larger.
        # If tvec components are small (camera is close), frustum should be smaller.
        # Let's try to scale f_scale based on the magnitude of the camera center distance.
        dist_to_origin = np.linalg.norm(cam_center_world)
        if dist_to_origin > 1e-3:  # Avoid division by zero or very small number
            f_scale = dist_to_origin * 0.1  # Frustum depth is 10% of distance to origin
        else:
            f_scale = 0.1  # Default small frustum if camera is at origin
        f_scale = max(0.05, min(f_scale, 1.0))  # Clamp to reasonable visual size

    frustum_scale_x = 0.5 * f_scale
    frustum_scale_y = 0.4 * f_scale
    frustum_depth = 1.0 * f_scale

    frustum_pts_cam = np.array([
        [0, 0, 0],
        [-frustum_scale_x, -frustum_scale_y, frustum_depth],
        [frustum_scale_x, -frustum_scale_y, frustum_depth],
        [frustum_scale_x, frustum_scale_y, frustum_depth],
        [-frustum_scale_x, frustum_scale_y, frustum_depth]
    ])

    # Transform frustum points from camera frame to world frame
    frustum_pts_world = (R_cam_in_world @ frustum_pts_cam.T).T + cam_center_world

    fig_data = []
    axis_len_world = max(0.5, target_size * 20)  # Make world axes visible relative to target

    # 1. World Axes (Xw, Yw, Zw at origin)
    fig_data.append(
        go.Scatter3d(x=[0, axis_len_world], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='magenta', width=4),
                     name='World X (Xw)'))
    fig_data.append(
        go.Scatter3d(x=[0, 0], y=[0, axis_len_world], z=[0, 0], mode='lines', line=dict(color='cyan', width=4),
                     name='World Y (Yw)'))
    fig_data.append(
        go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_len_world], mode='lines', line=dict(color='yellow', width=4),
                     name='World Z (Zw)'))

    # 2. Calibration Target Representation (Cube at World Origin)
    s = target_size / 5.0
    cube_corners = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # bottom face
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]  # top face
    ])
    fig_data.append(go.Mesh3d(
        x=cube_corners[:, 0], y=cube_corners[:, 1], z=cube_corners[:, 2],
        i=[0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7],  # Indices for faces
        j=[1, 3, 4, 2, 2, 5, 6, 3, 6, 7, 4, 5, 7, 6, 7, 0],
        k=[2, 7, 5, 6, 5, 2, 4, 7, 3, 5, 0, 6, 0, 1, 2, 3],
        opacity=0.6, color='dodgerblue', name='Target', showlegend=True
    ))

    # 3. Camera Center
    fig_data.append(
        go.Scatter3d(x=[cam_center_world[0]], y=[cam_center_world[1]], z=[cam_center_world[2]], mode='markers',
                     marker=dict(color='black', size=1, symbol='diamond'), name='Cam Center (Xc,Yc,Zc Origin)'))

    # 4. Camera Axes (Xc, Yc, Zc)
    cam_axis_plot_len = f_scale * 100  # Length of plotted camera axes
    fig_data.append(go.Scatter3d(x=[cam_center_world[0], cam_center_world[0] + cam_axis_plot_len * cam_x_axis_world[0]],
                                 y=[cam_center_world[1], cam_center_world[1] + cam_axis_plot_len * cam_x_axis_world[1]],
                                 z=[cam_center_world[2], cam_center_world[2] + cam_axis_plot_len * cam_x_axis_world[2]],
                                 mode='lines', line=dict(color='red', width=5), name='Cam X (Xc)'))
    fig_data.append(go.Scatter3d(x=[cam_center_world[0], cam_center_world[0] + cam_axis_plot_len * cam_y_axis_world[0]],
                                 y=[cam_center_world[1], cam_center_world[1] + cam_axis_plot_len * cam_y_axis_world[1]],
                                 z=[cam_center_world[2], cam_center_world[2] + cam_axis_plot_len * cam_y_axis_world[2]],
                                 mode='lines', line=dict(color='green', width=5), name='Cam Y (Yc)'))
    fig_data.append(go.Scatter3d(x=[cam_center_world[0], cam_center_world[0] + cam_axis_plot_len * cam_z_axis_world[0]],
                                 y=[cam_center_world[1], cam_center_world[1] + cam_axis_plot_len * cam_z_axis_world[1]],
                                 z=[cam_center_world[2], cam_center_world[2] + cam_axis_plot_len * cam_z_axis_world[2]],
                                 mode='lines', line=dict(color='blue', width=5), name='Cam Z (Zc - View)'))

    # 5. Camera Frustum (lines from camera center to target corners for simplicity, or actual frustum)
    # For simplicity, let's draw lines from camera center to target cube corners
    # This is more like "lines of sight" to the target corners than a true FOV frustum.
    for corner_idx in range(cube_corners.shape[0]):
        fig_data.append(go.Scatter3d(
            x=[cam_center_world[0], cube_corners[corner_idx, 0]],
            y=[cam_center_world[1], cube_corners[corner_idx, 1]],
            z=[cam_center_world[2], cube_corners[corner_idx, 2]],
            mode='lines', line=dict(color='rgba(128,0,128,0.3)', width=1, dash='dot'), showlegend=False
        ))

    fig = go.Figure(data=fig_data)

    # Set plot layout - make sure all elements are visible
    all_plot_pts = np.vstack([
        np.array([[0, 0, 0], [axis_len_world, 0, 0], [0, axis_len_world, 0], [0, 0, axis_len_world]]),
        cam_center_world.reshape(1, 3),
        frustum_pts_world,  # Using the actual frustum points for bounds
        cube_corners
    ])

    min_vals = np.min(all_plot_pts, axis=0)
    max_vals = np.max(all_plot_pts, axis=0)
    scene_center = (min_vals + max_vals) / 2.0
    scene_range = np.max(max_vals - min_vals) * 0.6  # Add some padding

    fig.update_layout(
        title='Interactive Camera and Target Pose',
        scene=dict(
            xaxis=dict(title='World X (mm)', range=[scene_center[0] - scene_range, scene_center[0] + scene_range]),
            yaxis=dict(title='World Y (mm)', range=[scene_center[1] - scene_range, scene_center[1] + scene_range]),
            zaxis=dict(title='World Z (mm)', range=[scene_center[2] - scene_range, scene_center[2] + scene_range]),
            aspectmode='cube',  # Enforce cubic aspect ratio
            camera_eye=dict(x=1.25, y=1.25, z=1.25)
        ),
        margin=dict(l=10, r=10, b=10, t=50),
        legend=dict(orientation="v", yanchor="top", y=0.95, xanchor="left", x=0.01)
    )
    return fig


# --- Main Streamlit Application ---
def main():
    st.set_page_config(layout="wide")
    st.title("📷 Camera Calibration Visualizer")

    # --- 1. Input Section ---
    st.header("1. Input Settings")
    default_path = ""
    folder_path = st.text_input("1.1 Path to calibration folder:", default_path)

    selected_image_index = None  # Initialize
    image_paths = []
    calib_data = None

    if folder_path and os.path.isdir(folder_path):
        input_xml_path = os.path.join(folder_path, "input.xml")
        calib_yml_path = os.path.join(folder_path, "calib_camera.yml")

        if not os.path.exists(input_xml_path):
            st.error(f"`input.xml` not found in the specified folder: {folder_path}")
            return
        if not os.path.exists(calib_yml_path):
            st.error(f"`calib_camera.yml` not found in the specified folder: {folder_path}")
            return

        image_paths = parse_input_xml(input_xml_path)
        calib_data = load_and_parse_calib_yml(calib_yml_path)

        if not calib_data:
            st.error(
                "Failed to load or parse `calib_camera.yml`. Cannot proceed. Please check the file and console for specific errors from the YAML parser.")
            return

        if not image_paths:
            st.info(
                "No images listed in input.xml to select for per-view data. Displaying general calibration parameters if available.")
        else:
            image_basenames = [os.path.basename(p) for p in image_paths]
            selected_image_index = st.selectbox(
                "1.2 Choose an image to view:",
                range(len(image_basenames)),
                format_func=lambda x: f"{x}: {image_basenames[x]}"
            )
    elif folder_path:  # Input is given but not a valid directory
        st.error("The entered path is not a valid directory. Please check the path and try again.")
        return  # Stop further execution if path is invalid
    else:  # No folder path entered yet
        st.info("👋 Welcome! Please enter the path to your camera calibration folder above to begin.")
        return  # Stop further execution until path is provided

    st.markdown("---")

    # --- 2. Image with Markings (Full Width) ---
    # Make this section collapsible
    image_expander_label = "2. Image View and Statistics"
    if selected_image_index is not None and image_paths:
        current_image_basename = os.path.basename(image_paths[selected_image_index])
        image_expander_label = f"2. Image: {current_image_basename} & Statistics"

        with st.expander(image_expander_label, expanded=True):
            current_image_path = image_paths[selected_image_index]
            if os.path.exists(current_image_path):
                try:
                    image_cv = cv2.imread(current_image_path)
                    if image_cv is None:
                        st.error(f"Could not load image: {current_image_path}. Check path and file integrity.")
                    else:
                        image_points_for_view = None
                        if "image_points" in calib_data and \
                                isinstance(calib_data.get("image_points"), np.ndarray) and \
                                calib_data["image_points"].ndim == 3 and \
                                selected_image_index < calib_data["image_points"].shape[0]:
                            image_points_for_view = calib_data["image_points"][selected_image_index]

                        image_cv_with_points = draw_points_on_image(image_cv, image_points_for_view, color=(0, 255, 0))
                        image_display = cv2.cvtColor(image_cv_with_points, cv2.COLOR_BGR2RGB)
                        caption = f"{current_image_basename} (detected points in green)"
                        if image_points_for_view is None and "image_points" in calib_data:
                            caption = current_image_basename + " (no valid points to draw for this view)"
                        elif "image_points" not in calib_data:
                            caption = current_image_basename + " (image_points data not found in YML)"
                        st.image(image_display, caption=caption, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading or processing image {current_image_path}: {e}")
            else:
                st.error(f"Image file not found: {current_image_path}")
                xml_dir = os.path.dirname(input_xml_path)
                relative_image_path = os.path.join(xml_dir, os.path.basename(current_image_path))
                if os.path.exists(relative_image_path):
                    st.info(
                        f"Attempting to load image from relative path: {relative_image_path} as absolute path failed. Consider using relative paths in input.xml or ensure absolute paths are correct.")
                else:
                    st.error(f"Also tried relative path {relative_image_path}, but image not found.")

        # --- 2.1 Calibration Statistics Grid ---
        st.subheader("2.1 Calibration Statistics")
        stats_cols = st.columns(3)
        with stats_cols[0]:
            st.metric("Image Width", f"{calib_data.get('image_width', 'N/A')}")
            st.metric("Image Height", f"{calib_data.get('image_height', 'N/A')}")
        with stats_cols[1]:
            st.metric("Frames (XML)", f"{len(image_paths) if image_paths else 'N/A'}")
            st.metric("Frames (YML)", f"{calib_data.get('nframes', 'N/A')}")

        avg_err_val = None
        if "avg_reprojection_error" in calib_data:
            avg_err = calib_data['avg_reprojection_error']
            avg_err_val = avg_err.item() if isinstance(avg_err, np.ndarray) and avg_err.size == 1 else (
                avg_err if isinstance(avg_err, (float, int)) else None)
        with stats_cols[2]:
            st.metric("Avg. Reproj. Error", f"{avg_err_val:.4f} px" if avg_err_val is not None else "N/A")

            err_view_val = None
            if "per_view_reprojection_errors" in calib_data and \
                    isinstance(calib_data.get("per_view_reprojection_errors"), np.ndarray) and \
                    selected_image_index is not None and \
                    selected_image_index < calib_data["per_view_reprojection_errors"].shape[0]:
                err_view_arr = calib_data["per_view_reprojection_errors"][selected_image_index]
                err_view_val = err_view_arr.item() if isinstance(err_view_arr,
                                                                 np.ndarray) and err_view_arr.size == 1 else (
                    err_view_arr if isinstance(err_view_arr, (float, int)) else None)

            if err_view_val is not None:
                st.metric(f"Reproj. Error (View {selected_image_index + 1})", f"{err_view_val:.4f} px")
            elif selected_image_index is not None:
                st.text(f"Reproj. Error (View {selected_image_index + 1}): N/A")


    elif not image_paths and calib_data:
        st.header("General Calibration Data (No Image Selected)")
        st.subheader("2.1 Calibration Statistics")
        stats_cols = st.columns(3)
        with stats_cols[0]:
            st.metric("Image Width", f"{calib_data.get('image_width', 'N/A')}")
        with stats_cols[1]:
            st.metric("Image Height", f"{calib_data.get('image_height', 'N/A')}")
        with stats_cols[2]:
            st.metric("Frames (YML)", f"{calib_data.get('nframes', 'N/A')}")
        avg_err_val = None
        if "avg_reprojection_error" in calib_data:
            avg_err = calib_data['avg_reprojection_error']
            avg_err_val = avg_err.item() if isinstance(avg_err, np.ndarray) and avg_err.size == 1 else (
                avg_err if isinstance(avg_err, (float, int)) else None)
        st.metric("Avg. Reproj. Error", f"{avg_err_val:.4f} px" if avg_err_val is not None else "N/A")
    elif not calib_data and folder_path:
        pass

    st.markdown("---")

    # --- 3. Camera Intrinsic Parameters ---
    if calib_data:
        st.header("3. Camera Intrinsic Parameters")
        col3_1, col3_2 = st.columns(2)
        with col3_1:
            st.markdown("##### Camera Matrix (K)")
            if "camera_matrix" in calib_data and isinstance(calib_data.get("camera_matrix"), np.ndarray):
                st.dataframe(pd.DataFrame(calib_data["camera_matrix"]))
            else:
                st.text("Not available or invalid format.")
        with col3_2:
            st.markdown("##### Distortion Coefficients (D)")
            if "distortion_coefficients" in calib_data and isinstance(calib_data.get("distortion_coefficients"),
                                                                      np.ndarray):
                st.dataframe(pd.DataFrame(calib_data["distortion_coefficients"]))
            else:
                st.text("Not available or invalid format.")

        # --- 4. Camera Extrinsic Parameters ---
        if selected_image_index is not None and image_paths:
            st.header(f"4. Camera Extrinsic Parameters (View {selected_image_index + 1})")

            rvec, tvec, R_mat = None, None, None  # Initialize
            K_intrinsic_matrix = calib_data.get("camera_matrix") if isinstance(calib_data.get("camera_matrix"),
                                                                               np.ndarray) else None

            if "extrinsic_parameters" in calib_data and \
                    isinstance(calib_data.get("extrinsic_parameters"), np.ndarray) and \
                    selected_image_index < calib_data["extrinsic_parameters"].shape[0]:
                extrinsics_for_view = calib_data["extrinsic_parameters"][selected_image_index]
                if extrinsics_for_view.ndim == 1 and extrinsics_for_view.shape[0] == 6:
                    rvec, tvec = extrinsics_for_view[:3], extrinsics_for_view[3:]
                    try:
                        R_mat, _ = cv2.Rodrigues(rvec)
                    except Exception as e:
                        st.warning(f"Could not convert rvec to Rotation Matrix: {e}")
                else:
                    st.warning(
                        f"Extrinsics for view {selected_image_index + 1} have unexpected shape: {extrinsics_for_view.shape}. Expected 1D array of 6 elements.")
            else:
                st.warning(f"Extrinsics not available/valid for view {selected_image_index + 1}.")

            # Layout for text data and plot
            col4_text, col4_plot = st.columns([1, 1.2])  # Give a bit more space to plot

            with col4_text:
                st.markdown("##### Translation Vector (tvec)")
                if tvec is not None:
                    st.code(f"{tvec}")
                else:
                    st.text("Not available.")

                st.markdown("##### Rotation")
                if R_mat is not None and rvec is not None:
                    st.markdown("###### Rotation Vector (rvec - Rodrigues form)")
                    st.code(f"{rvec}")
                    st.markdown("###### Rotation Matrix (R_mat - Matrix form)")
                    st.dataframe(pd.DataFrame(R_mat))

                    yaw, pitch, roll = get_euler_angles(rvec)
                    st.markdown("###### Euler Angles (degrees)")

                    euler_col1, euler_col2, euler_col3 = st.columns(3)
                    if yaw is not None and pitch is not None and roll is not None:
                        euler_col1.text(f"Roll (X): {roll + 180:.2f}°")
                        euler_col2.text(f"Pitch (Y): {pitch:.2f}°")
                        euler_col3.text(f"Yaw (Z): {yaw:.2f}°")
                    else:
                        st.text("Euler angles could not be computed.")
                else:
                    st.text("Rotation data not available.")

            with col4_plot:
                st.markdown("##### Camera Pose Visualization")
                if rvec is not None and tvec is not None:
                    try:
                        # Use a default target size, can be made configurable later
                        target_cube_size = 0.2
                        if 'grid_points' in calib_data and isinstance(calib_data.get('grid_points'), np.ndarray) and \
                                calib_data['grid_points'].size > 0:
                            # Estimate target size from grid points if available
                            # This is a rough estimation, assuming grid points are somewhat centered around origin
                            max_coords = np.max(np.abs(calib_data['grid_points']), axis=0)
                            target_cube_size = np.max(max_coords) * 0.1  # Scale down for visualization
                            target_cube_size = max(0.05, target_cube_size)  # Ensure a minimum size

                        pose_fig = plot_camera_pose_3d_plotly(rvec, tvec, K_intrinsic_matrix,
                                                              target_size=target_cube_size)
                        st.plotly_chart(pose_fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting camera pose with Plotly: {e}")
                else:
                    st.text("Not enough data to plot camera pose.")

        st.markdown("---")

        # --- 5. Points Data ---
        st.header("5. Points Data")
        col5_1, col5_2 = st.columns(2)
        with col5_1:
            st.markdown("##### 5.1 3D Object Points (Grid Points from YML)")
            if "grid_points" in calib_data and isinstance(calib_data.get("grid_points"), np.ndarray):
                st.dataframe(pd.DataFrame(calib_data["grid_points"], columns=['X', 'Y', 'Z']))
            else:
                st.text("Not available or invalid format.")

        with col5_2:
            st.markdown(
                f"##### 5.2 Detected Image Points (View {selected_image_index + 1 if selected_image_index is not None else 'N/A'})")
            if selected_image_index is not None and image_paths and \
                    "image_points" in calib_data and \
                    isinstance(calib_data.get("image_points"), np.ndarray) and \
                    calib_data["image_points"].ndim == 3 and \
                    selected_image_index < calib_data["image_points"].shape[0]:
                image_points_for_view = calib_data["image_points"][selected_image_index]
                if image_points_for_view.ndim == 2 and image_points_for_view.shape[1] == 2:
                    st.dataframe(pd.DataFrame(image_points_for_view, columns=['x', 'y']))
                else:
                    st.warning(
                        f"Image points for view {selected_image_index + 1} have unexpected shape: {image_points_for_view.shape}")
            elif selected_image_index is None and calib_data:
                st.text("No image selected to display detected points.")
            elif calib_data:
                st.text("Not available or invalid format for this view.")


if __name__ == '__main__':
    main()