# Folder data processing
# Point cloud segmentation (splite_bound --> pointcloudwith) --> Template library creation (creat_template --> trans_01_02 --> trans_02_03) --> Dataset creation (creat_data --> divide_train_test)
import time
import numpy as np
import vtk
import pyvista as pv
from scipy.spatial import cKDTree
import os
from sklearn.cluster import DBSCAN
from function_pool import model_judgment_corroded_ground_line, template_matching, insulator_correction
import chardet
import xml.etree.ElementTree as ET
from pyproj import Transformer


def get_template(file_path_template):
    # Read template
    txt_files = [f for f in os.listdir(file_path_template) if f.endswith('.txt')]
    # Read each txt file and add its content to all_data list
    # template_ground_two_points = []
    # template_width_points = []
    all_template_library, all_template_direction, all_template_insulator = [], [], []
    for txt_file in txt_files:
        a_template_library, a_template_direction, a_template_insulator = [], [], []
        # Read file
        with open(os.path.join(file_path_template, txt_file), 'r') as f:
            lines = f.readlines()
        # Read Tower sampling points
        start_idx = lines.index('Tower sampling points:\n') + 1
        end_idx = lines.index('insulator direct(0 is horizontal, 1 is inclined, and 2 is vertical):\n')
        a_template_library = [list(map(float, line.split())) for line in lines[start_idx:end_idx] if line.strip()]
        # Read insulator direct
        start_idx = end_idx + 1
        end_idx = lines.index('insulator three point:\n')
        a_template_direction = [float(line.strip()) for line in lines[start_idx:end_idx] if line.strip()]
        # Read insulator three point
        start_idx = end_idx + 1
        a_template_insulator = [list(map(float, line.split())) for line in lines[start_idx:] if line.strip()]
        # Convert to numpy arrays
        a_template_library = np.array(a_template_library)
        a_template_direction = np.array(a_template_direction)
        a_template_insulator = np.array(a_template_insulator)
        all_template_library.append(a_template_library)
        all_template_direction.append(a_template_direction)
        all_template_insulator.append(a_template_insulator)
    # template_ground_two_points = np.array(template_ground_two_points)
    # template_width_points = np.array(template_width_points)
    all_template_library = np.array(all_template_library)
    return all_template_library, all_template_direction, all_template_insulator

def pick_point(plotter, click_pos):
    picker = vtk.vtkPointPicker()
    picker.Pick(click_pos[0], click_pos[1], 0, plotter.renderer)
    picked_point = picker.GetPickPosition()
    if picked_point != (0.0, 0.0, 0.0):  # Check if point is picked
        return picked_point
    return None

is_processing = False

def right_click_callback(iren, event):
    global is_processing
    if is_processing:
        return  # If processing, ignore this click

    click_pos = iren.GetEventPosition()
    picked_point = pick_point(plotter, click_pos)
    if picked_point is not None:
        # Set flag variable to True, indicating processing
        is_processing = True
        on_right_click(picked_point)
        # After processing in on_right_click is complete, reset flag to False

def on_right_click(picked_point):
    global is_processing
    picked_coords = np.array(picked_point)
    print(f"Right-click:{picked_coords}, starting program execution\n")

    all_time = []

    all_start_time = time.time()
    # First use pointnet network plus corrosion algorithm for prior knowledge calculation
    # Return parameters: power line, top power line, tower center point, ground line distance, ground line direction
    start_time = time.time()
    print("Starting feature extraction phase")
    tower_central_point, hengxiang_vec, max_hengxiang_dist, top_line_set, top_line_direct = model_judgment_corroded_ground_line(points, tree, colors, plotter)
    # plotter.add_mesh(points[list(top_line_set)], color='green', point_size=8)
    plotter.add_mesh(tower_central_point, color='red', point_size=12)
    l_line = pv.Line(tower_central_point, tower_central_point + 20 * top_line_direct[0])
    plotter.add_mesh(l_line, color='red', line_width=5, label='Same Direction')
    l_line = pv.Line(tower_central_point, tower_central_point + 20 * top_line_direct[1])
    plotter.add_mesh(l_line, color='red', line_width=5, label='Same Direction')
    l_line = pv.Line(tower_central_point + hengxiang_vec / 2, tower_central_point - hengxiang_vec / 2)
    plotter.add_mesh(l_line, color='red', line_width=5, label='Same Direction')
    plotter.update()
    end_time = time.time()
    all_time.append(end_time - start_time)
    print(f"Time consumed for power line judgment using network: {end_time - start_time}s")

    # Then match with template library
    start_time = time.time()
    print("Starting template matching phase")
    gbest_solution, gbest_index, all_tower_set, match_sample, match_insulator, match_direction = template_matching(template_library, template_insulator, template_direction, points, tree, tower_central_point, hengxiang_vec, top_line_set, top_line_direct, max_hengxiang_dist, plotter)
    # plotter.add_mesh(match_sample, color='blue', point_size=8)
    # plotter.add_mesh(points[list(all_tower_set)], color='blue', point_size=4)
    # plotter.update()
    # plotter.add_mesh(match_insulator, color='red', point_size=4)
    # plotter.update()
    end_time = time.time()
    all_time.append(end_time - start_time)
    print(f"Template matching algorithm consumed: {end_time - start_time}s")

    start_time = time.time()
    print("Starting insulator correction phase")
    reshaped_insulator, insulator_set = insulator_correction(points, tree, top_line_direct, match_direction, match_insulator, plotter)
    # plotter.add_mesh(reshaped_insulator, color='red', point_size=12)
    # plotter.add_mesh(points[list(insulator_set)], color='red', point_size=12)
    # plotter.update()
    end_time = time.time()
    all_time.append(end_time - start_time)
    print(f"Insulator correction module consumed: {end_time - start_time}s")

    all_end_time = time.time()
    print(f"Total algorithm consumption time: {all_end_time - all_start_time}s")

    # # Perform correction results
    top_line_set -= insulator_set
    all_tower_set -= insulator_set
    all_tower_set -= top_line_set
    plotter.add_mesh(points[list(top_line_set)], color='green', point_size=4)    # Power line
    plotter.add_mesh(points[list(all_tower_set)], color='blue', point_size=4)    # Tower
    plotter.add_mesh(points[list(insulator_set)], color='red', point_size=4)     # Insulator
    plotter.add_mesh(reshaped_insulator, color='yellow', point_size=8)           # Three insulator points
    plotter.update()
    time.sleep(1)

    # Save three points of insulator string
    save_name = file_path.split('\\')[-1].split('.')[0]
    np.savetxt(f"insulator_points/{save_name}.txt", reshaped_insulator)

    # Save running time
    np.savetxt(f"110kv_template_matching_result/running_time/{save_name}.txt", all_time)

    # Save visualization results
    # Save visualization results
    plotter.view_xz()
    # Calculate angle (0-360 degrees)
    angle_rad = np.arctan2(hengxiang_vec[1], hengxiang_vec[0])  # Returns radian value
    angle_deg = np.degrees(angle_rad)  # Convert to degrees
    # Ensure angle is between 0 and 360 degrees
    if angle_deg < 0:
        angle_deg += 360
    plotter.camera.azimuth = angle_deg + 30
    plotter.camera.elevation = 30
    plotter.camera.focal_point = np.array(
        (tower_central_point[0], tower_central_point[1], np.mean(points[list(all_tower_set)][:, 2])))  # Directly set focal point
    plotter.camera.zoom(2)  # Zoom in view, adjust zoom level
    # Force update rendering and ensure scene is fully rendered
    plotter.update()
    # Wait for rendering to complete, can increase wait time appropriately
    time.sleep(2)
    # Ensure screenshot after scene rendering
    plotter.screenshot(f"matching_result_figures/{save_name}.png")

    print("Save completed")

    is_processing = False


if __name__ == "__main__":
    # Read tower point cloud data
    folder_path = 'G:/BaiduNetdiskDownload/DataSet'  # Replace with your folder path
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                  os.path.isfile(os.path.join(folder_path, f))]
    file_paths = np.array(file_paths)
    file_paths = [path for path in file_paths if 'new_' in os.path.basename(path)]

    # Read template
    # Get all txt files in the folder
    file_path_template = 'knowledge_based'
    template_library, template_direction, template_insulator = get_template(file_path_template)

    # np.random.shuffle(file_paths)
    # file_paths = file_paths[::-1]
    file_paths = file_paths[1:]
    # Loop to read files
    for file_path in file_paths:
        try:
            # file_path = '../输电线路切分的txt点云/800kv陕武线/陕武线#1810_49N.txt'
            print(file_path)
            # Use np.loadtxt to read file
            point_cloud_data = np.loadtxt(file_path, dtype=np.float64)  # Can set dtype as needed
            point_cloud_data = np.unique(point_cloud_data, axis=0)
            point_cloud_data = point_cloud_data[np.where(point_cloud_data[:, 2] != 0)]
            # Create copies of original_points and original_colors
            original_points = point_cloud_data[:, :3].copy()  # Point cloud
            original_colors = (point_cloud_data[:, 3:]).copy()  # Colors
            # points and colors point to new copies
            points = point_cloud_data[:, :3].copy()  # Point cloud
            # colors = (point_cloud_data[:, 3:]).copy()  # Colors
            if np.max(point_cloud_data[:, 3]) > 1:
                colors = 0.7 * (point_cloud_data[:, 3:]).copy() / 255  # Colors
            else:
                colors = 0.7 * (point_cloud_data[:, 3:]).copy()  # Colors

            # heights = points[:, 2]  # Take z-axis values as height

            # Create KDTree index
            tree = cKDTree(points)
            # Create PolyData object and add color information
            cloud = pv.PolyData(points)
            cloud['RGB'] = colors
            # cloud['Height'] = heights  # Add height data as scalars to point cloud
            # Create Plotter object and add point cloud
            plotter = pv.Plotter()
            # plotter.add_mesh(cloud, scalars='Height', cmap='cividis', point_size=1.5)
            plotter.add_mesh(cloud, scalars='RGB', rgb=True, point_size=1.5)

            # Add right-click event
            iren = plotter.iren
            iren.add_observer("RightButtonPressEvent", right_click_callback)            # Show point cloud
            plotter.show()
        except Exception as e:
            print(f"Error loading file '{file_path}': {e}")