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
    # This line is moved here to ensure num_elements_in_data is always defined before the next try block.
    num_elements_in_data = numpy_array.size

    # Reshape the array
    try:
        if dt == '2f':
            # num_elements_in_data is now defined
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
            # num_elements_in_data is now defined
            # For other data types (not '2f'), assume 'cols' is the direct number of columns.
            if num_elements_in_data == rows * cols:
                return numpy_array.reshape(rows, cols)
            elif num_elements_in_data == 0 and rows >= 0:
                return np.zeros((rows, cols), dtype=np_dtype)  # Return a correctly shaped zero array
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
        image_paths = root.find('images').text.strip().split('\n')
        # Filter out any empty strings that might result from split
        image_paths = [path for path in image_paths if path.strip()]
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
    except Exception as e:  # Catching the UnboundLocalError here if it originates from constructor
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

        if not calib_data:  # This check is crucial
            st.error(
                "Failed to load or parse `calib_camera.yml`. Cannot proceed. Please check the file and console for specific errors from the YAML parser.")
            return

        st.sidebar.header("Image Selection")
        selected_image_index = None
        if not image_paths:
            st.sidebar.info("No images listed in input.xml to select for per-view data.")
        else:
            image_basenames = [os.path.basename(p) for p in image_paths]
            selected_image_index = st.sidebar.selectbox(
                "Choose an image to view:",
                range(len(image_basenames)),
                format_func=lambda x: f"{x}: {image_basenames[x]}"
            )

        if selected_image_index is not None and image_paths:
            current_image_path = image_paths[selected_image_index]
            current_image_basename = os.path.basename(current_image_path)
            st.header(f"Displaying: {current_image_basename}")
            col1, col2 = st.columns(2)
        else:
            st.header("General Calibration Data")
            col1, col2 = st, st

        with col1:
            if selected_image_index is not None and image_paths:
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
                                    calib_data["image_points"].ndim == 3 and \
                                    selected_image_index < calib_data["image_points"].shape[0]:
                                image_points_for_view = calib_data["image_points"][selected_image_index]

                            image_cv_with_points = draw_points_on_image(image_cv, image_points_for_view,
                                                                        color=(0, 255, 0))
                            image_display = cv2.cvtColor(image_cv_with_points, cv2.COLOR_BGR2RGB)
                            caption = f"{current_image_basename} (detected points in green)"
                            if image_points_for_view is None and "image_points" in calib_data:
                                caption = current_image_basename + " (no valid points to draw for this view)"
                            elif "image_points" not in calib_data:
                                caption = current_image_basename + " (image_points data not found in YML)"

                            st.image(image_display, caption=caption, use_column_width=True)

                    except Exception as e:
                        st.error(f"Error loading or processing image {current_image_path}: {e}")
                else:
                    st.error(f"Image file not found: {current_image_path}")
                    xml_dir = os.path.dirname(input_xml_path)
                    relative_image_path = os.path.join(xml_dir, os.path.basename(current_image_path))
                    if os.path.exists(relative_image_path):
                        st.info(
                            f"Found image at relative path: {relative_image_path}. Consider using relative paths in input.xml or ensure absolute paths are correct.")
            elif not image_paths and calib_data:
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

            if selected_image_index is not None and image_paths:
                st.markdown(f"##### Extrinsic Parameters (View {selected_image_index + 1})")
                if "extrinsic_parameters" in calib_data and \
                        isinstance(calib_data.get("extrinsic_parameters"), np.ndarray) and \
                        selected_image_index < calib_data["extrinsic_parameters"].shape[0]:
                    extrinsics_for_view = calib_data["extrinsic_parameters"][selected_image_index]
                    if extrinsics_for_view.ndim == 1 and extrinsics_for_view.shape[0] == 6:
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
                            f"Extrinsics for view {selected_image_index + 1} have unexpected shape: {extrinsics_for_view.shape}. Expected 1D array of 6 elements.")
                else:
                    st.warning(f"Extrinsics not available/valid for view {selected_image_index + 1}.")

            st.markdown("##### Calibration Statistics")
            if "image_width" in calib_data and "image_height" in calib_data:
                st.text(
                    f"Calibrated Image Size: {calib_data.get('image_width', 'N/A')}x{calib_data.get('image_height', 'N/A')}")
            if "nframes" in calib_data: st.text(f"Number of frames in YML: {calib_data.get('nframes', 'N/A')}")
            if image_paths: st.text(f"Number of images in XML: {len(image_paths)}")

            if "avg_reprojection_error" in calib_data:
                avg_err = calib_data['avg_reprojection_error']
                avg_err_val = avg_err.item() if isinstance(avg_err, np.ndarray) and avg_err.size == 1 else (
                    avg_err if isinstance(avg_err, (float, int)) else None)
                if avg_err_val is not None:
                    st.metric("Avg. Reprojection Error:", f"{avg_err_val:.4f} pixels")
                else:
                    st.text(f"Avg. Reprojection Error: {avg_err} (could not parse as scalar)")

            if selected_image_index is not None and image_paths:
                if "per_view_reprojection_errors" in calib_data and \
                        isinstance(calib_data.get("per_view_reprojection_errors"), np.ndarray) and \
                        selected_image_index < calib_data["per_view_reprojection_errors"].shape[0]:
                    err_view_arr = calib_data["per_view_reprojection_errors"][selected_image_index]
                    err_view = err_view_arr.item() if isinstance(err_view_arr,
                                                                 np.ndarray) and err_view_arr.size == 1 else (
                        err_view_arr if isinstance(err_view_arr, (float, int)) else None)

                    if err_view is not None:
                        st.metric(f"Reprojection Error (View {selected_image_index + 1}):", f"{err_view:.4f} pixels")
                    else:
                        st.text(
                            f"Per-view Error (View {selected_image_index + 1}): {err_view_arr} (unexpected format or size)")
                else:
                    st.warning(f"Per-view error not available/valid for view {selected_image_index + 1}.")

                if "image_points" in calib_data and \
                        isinstance(calib_data.get("image_points"), np.ndarray) and \
                        calib_data["image_points"].ndim == 3 and \
                        selected_image_index < calib_data["image_points"].shape[0]:
                    image_points_for_view = calib_data["image_points"][selected_image_index]
                    st.markdown(f"##### Detected Image Points (View {selected_image_index + 1})")
                    if image_points_for_view.ndim == 2 and image_points_for_view.shape[1] == 2:
                        st.dataframe(pd.DataFrame(image_points_for_view, columns=['x', 'y']))
                    else:
                        st.warning(
                            f"Image points for view {selected_image_index + 1} have unexpected shape: {image_points_for_view.shape}")

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