import streamlit as st
import xml.etree.ElementTree as ET
import yaml
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd


# --- YAML Custom Constructor for OpenCV Matrices ---
def opencv_matrix_constructor(loader, node):
    """
    Custom YAML constructor to parse OpenCV matrix structures.
    It handles various data types and shapes, including the '2f' type for image points.
    """
    mapping = loader.construct_mapping(node, deep=True)
    mat_data = mapping.get('data', [])  # Use .get for safety
    rows = int(mapping['rows'])
    cols = int(mapping['cols'])
    dt = mapping.get('dt', 'f')  # Default to float if not specified

    # Ensure mat_data is a list (handles single-element data like 'data: 0.038')
    if not isinstance(mat_data, list):
        mat_data = [mat_data]

    # If data is empty for some reason (e.g. optional matrix not present)
    if not mat_data and rows > 0 and cols > 0:
        # Return an empty array of the expected type and rough shape
        if 'd' in dt:
            np_dtype = np.float64
        elif 'f' in dt:
            np_dtype = np.float32
        else:
            np_dtype = np.int32  # Default for other simple types

        if dt == '2f':
            return np.zeros((rows, cols // 2, 2), dtype=np_dtype)
        else:
            return np.zeros((rows, cols), dtype=np_dtype)

    # Determine numpy data type
    if 'd' in dt:  # double
        np_dtype = np.float64
    elif 'f' in dt:  # float (covers 'f' and '2f' for base type)
        np_dtype = np.float32
    elif 'i' in dt or 's' in dt:  # signed integer, short
        np_dtype = np.int32
    elif 'u' in dt:  # unsigned integer
        np_dtype = np.uint32
    else:  # Default if dt is not recognized or complex
        np_dtype = np.float32
        # st.warning(f"Unrecognized OpenCV matrix data type '{dt}'. Defaulting to float32.")

    try:
        numpy_array = np.array(mat_data, dtype=np_dtype)
    except ValueError as e:
        st.error(
            f"Error converting data to numpy array for matrix with dt='{dt}', rows={rows}, cols={cols}. Data: {mat_data}. Error: {e}")
        if dt == '2f':
            return np.zeros((rows, cols // 2, 2), dtype=np_dtype)
        else:
            return np.zeros((rows, cols), dtype=np_dtype)

    # Reshape the array
    try:
        if dt == '2f':
            # '2f' means pairs of floats (e.g., image points x,y)
            return numpy_array.reshape(rows, cols // 2, 2)
        else:
            return numpy_array.reshape(rows, cols)
    except ValueError as e:
        st.error(
            f"Error reshaping numpy array for matrix. Expected shape based on rows={rows}, cols={cols}, dt='{dt}'. Array shape: {numpy_array.shape}. Error: {e}")
        if dt == '2f':
            return np.zeros((rows, cols // 2, 2), dtype=np_dtype)
        else:
            return np.zeros((rows, cols), dtype=np_dtype)

        # Add the constructor to PyYAML. SafeLoader is generally recommended.


# This constructor is registered for the '!opencv-matrix' tag.
yaml.add_constructor('!opencv-matrix', opencv_matrix_constructor, Loader=yaml.SafeLoader)


# --- File Parsing Functions ---
def parse_input_xml(xml_file_path):
    """Parses the input.xml file to extract image paths."""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        image_paths =  root.find('images').text.strip().split('\n')
        return image_paths
    except ET.ParseError as e:
        st.error(f"Error parsing XML file {xml_file_path}: {e}")
        return []
    except FileNotFoundError:
        st.error(f"XML file not found: {xml_file_path}")
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

        # Pre-process the YAML content as text, as requested by the user.
        # This replaces the shorthand '!!opencv-matrix' with '!opencv-matrix',
        # which matches the tag our custom constructor is registered for.
        processed_content = raw_content.replace('!!opencv-matrix', '!opencv-matrix')

        # Add more replacements here if other tags cause issues:
        # processed_content = processed_content.replace('some_other_tag', '!my_tag_handler')

        calib_data = yaml.load(processed_content, Loader=yaml.SafeLoader)

        # Post-process grid_points if it exists (it's not an opencv-matrix)
        if 'grid_points' in calib_data and isinstance(calib_data['grid_points'], list):
            try:
                # Assuming grid_points are 3D coordinates (X, Y, Z)
                calib_data['grid_points'] = np.array(calib_data['grid_points'], dtype=np.float32).reshape(-1, 3)
            except ValueError as e:
                st.warning(f"Could not reshape grid_points. It might have an unexpected number of elements. Error: {e}")
        return calib_data
    except FileNotFoundError:
        st.error(f"YAML file not found: {yml_file_path}")
        return None
    except yaml.YAMLError as e:
        st.error(
            f"Error parsing YAML file {yml_file_path} (after text processing): {e}\nContent snippet: {processed_content[:500]}")  # Show snippet for debug
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
        # Avoid showing a warning if points_array is None intentionally (e.g. not found)
        if points_array is not None:
            st.warning("Invalid format for image points. Cannot draw them.")
    return drawn_image


# --- Main Streamlit Application ---
def main():
    st.set_page_config(layout="wide")
    st.title("📷 Camera Calibration Visualizer")

    st.sidebar.header("Input Folder")
    default_path = ""
    folder_path = st.sidebar.text_input("Path to calibration folder:", default_path)

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

        if not image_paths:
            st.warning("No image paths found in `input.xml` or error during parsing.")
            # Allow proceeding if calib_data is fine, maybe user wants to see general params
        if not calib_data:
            st.warning("Could not load or parse `calib_camera.yml`.")
            return  # Critical if calib_data is None

        st.sidebar.header("Image Selection")
        if not image_paths:  # If image_paths is empty, selectbox will error
            st.sidebar.info("No images listed in input.xml to select for per-view data.")
            # Display general calibration data if available
            selected_image_index = None
        else:
            image_basenames = [os.path.basename(p) for p in image_paths]
            selected_image_index = st.sidebar.selectbox(
                "Choose an image to view:",
                range(len(image_basenames)),
                format_func=lambda x: f"{x}: {image_basenames[x]}"
            )

        # Main display area
        if selected_image_index is not None and image_paths:  # Ensure image_paths is not empty
            current_image_path = image_paths[selected_image_index]
            current_image_basename = os.path.basename(current_image_path)
            st.header(f"Displaying: {current_image_basename}")
            col1, col2 = st.columns(2)
        else:  # No image selected or no images available, show only general data if possible
            st.header("General Calibration Data")
            # Use a single column layout if no image is being shown
            col1, col2 = st, st  # Make col2 effectively the same as col1 for general data

        with col1:
            if selected_image_index is not None and image_paths:  # Only if an image is selected
                st.subheader("Calibrated Image")
                if os.path.exists(current_image_path):
                    try:
                        image_cv = cv2.imread(current_image_path)
                        if image_cv is None:
                            st.error(f"Could not load image: {current_image_path}. Check path and file integrity.")
                        else:
                            image_points_for_view = None
                            if "image_points" in calib_data and \
                                    isinstance(calib_data.get("image_points"), np.ndarray) and \
                                    selected_image_index < calib_data["image_points"].shape[0]:
                                image_points_for_view = calib_data["image_points"][selected_image_index]

                            image_cv_with_points = draw_points_on_image(image_cv, image_points_for_view,
                                                                        color=(0, 255, 0))
                            image_display = cv2.cvtColor(image_cv_with_points, cv2.COLOR_BGR2RGB)
                            caption = f"{current_image_basename} (detected points in green)"
                            if image_points_for_view is None:
                                caption = current_image_basename + " (no points to draw for this view)"
                            st.image(image_display, caption=caption, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error loading or processing image {current_image_path}: {e}")
                else:
                    st.error(f"Image file not found: {current_image_path}")
                    xml_dir = os.path.dirname(input_xml_path)
                    relative_image_path = os.path.join(xml_dir, os.path.basename(current_image_path))
                    if os.path.exists(relative_image_path):
                        st.info(
                            f"Found image at relative path: {relative_image_path}. Consider using relative paths in input.xml or ensure absolute paths are correct.")
            elif not image_paths and calib_data:  # No images in XML, but calib_data exists
                st.info("No images listed in input.xml. Displaying general calibration parameters only.")

        with col2:
            st.subheader("Calibration Data Details")

            st.markdown("##### Intrinsic Parameters (K, D)")
            if "camera_matrix" in calib_data and isinstance(calib_data.get("camera_matrix"), np.ndarray):
                st.text("Camera Matrix (K):")
                st.dataframe(pd.DataFrame(calib_data["camera_matrix"]))
            else:
                st.text("Camera Matrix (K): Not available or invalid format.")

            if "distortion_coefficients" in calib_data and isinstance(calib_data.get("distortion_coefficients"),
                                                                      np.ndarray):
                st.text("Distortion Coefficients (D):")
                st.dataframe(pd.DataFrame(calib_data["distortion_coefficients"]))
            else:
                st.text("Distortion Coefficients (D): Not available or invalid format.")

            if selected_image_index is not None and image_paths:  # Per-view data
                st.markdown(f"##### Extrinsic Parameters (View {selected_image_index + 1})")
                if "extrinsic_parameters" in calib_data and \
                        isinstance(calib_data.get("extrinsic_parameters"), np.ndarray) and \
                        selected_image_index < calib_data["extrinsic_parameters"].shape[0]:
                    extrinsics_for_view = calib_data["extrinsic_parameters"][selected_image_index]
                    if extrinsics_for_view.shape == (6,):
                        rvec, tvec = extrinsics_for_view[:3], extrinsics_for_view[3:]
                        st.text("Rotation Vector (rvec):");
                        st.code(f"{rvec}")
                        st.text("Translation Vector (tvec):");
                        st.code(f"{tvec}")
                        try:
                            R, _ = cv2.Rodrigues(rvec)
                            st.text("Rotation Matrix (R):");
                            st.dataframe(pd.DataFrame(R))
                        except Exception as e:
                            st.warning(f"Could not convert rvec to R: {e}")
                    else:
                        st.warning(
                            f"Extrinsics for view {selected_image_index + 1} shape: {extrinsics_for_view.shape}. Expected (6,).")
                else:
                    st.warning(f"Extrinsics not available/valid for view {selected_image_index + 1}.")

            st.markdown("##### Calibration Statistics")
            if "image_width" in calib_data and "image_height" in calib_data:
                st.text(f"Calibrated Image Size: {calib_data['image_width']}x{calib_data['image_height']}")
            if "nframes" in calib_data: st.text(f"Number of frames in YML: {calib_data['nframes']}")
            if image_paths: st.text(f"Number of images in XML: {len(image_paths)}")

            if "avg_reprojection_error" in calib_data:
                avg_err = calib_data['avg_reprojection_error']
                avg_err_val = avg_err.item() if isinstance(avg_err, np.ndarray) and avg_err.size == 1 else (
                    avg_err if isinstance(avg_err, (float, int)) else None)
                if avg_err_val is not None:
                    st.metric("Avg. Reprojection Error:", f"{avg_err_val:.4f} pixels")
                else:
                    st.text(f"Avg. Reprojection Error: {avg_err} (unparsed)")

            if selected_image_index is not None and image_paths:  # Per-view data
                if "per_view_reprojection_errors" in calib_data and \
                        isinstance(calib_data.get("per_view_reprojection_errors"), np.ndarray) and \
                        selected_image_index < calib_data["per_view_reprojection_errors"].shape[0]:
                    err_view_arr = calib_data["per_view_reprojection_errors"][selected_image_index]
                    err_view = err_view_arr.item() if err_view_arr.size == 1 else None
                    if err_view is not None:
                        st.metric(f"Reprojection Error (View {selected_image_index + 1}):", f"{err_view:.4f} pixels")
                    else:
                        st.text(f"Per-view Error (View {selected_image_index + 1}): {err_view_arr} (unparsed)")
                else:
                    st.warning(f"Per-view error not available/valid for view {selected_image_index + 1}.")

                if "image_points" in calib_data and \
                        isinstance(calib_data.get("image_points"), np.ndarray) and \
                        selected_image_index < calib_data["image_points"].shape[0]:
                    image_points_for_view = calib_data["image_points"][selected_image_index]
                    st.markdown(f"##### Detected Image Points (View {selected_image_index + 1})")
                    st.dataframe(pd.DataFrame(image_points_for_view, columns=['x', 'y']))

            if "grid_points" in calib_data and isinstance(calib_data.get("grid_points"), np.ndarray):
                st.markdown("##### 3D Object Points (Grid Points from YML)")
                st.text(f"Shape: {calib_data['grid_points'].shape}")
                st.dataframe(pd.DataFrame(calib_data["grid_points"], columns=['X', 'Y', 'Z']))

    elif folder_path:
        st.error("The entered path is not a valid directory. Please check and try again.")
    else:
        st.info("👋 Welcome! Please enter the path to your camera calibration folder in the sidebar to begin.")


if __name__ == '__main__':
    main()