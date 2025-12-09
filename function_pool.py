import math
import os
import random
import time
import xml
from itertools import chain
from xml.dom import minidom
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
import pyvista as pv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import argparse
import importlib
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from scipy.stats import mode
from sklearn.cluster import DBSCAN, KMeans
from AlgorithmParams import Parameters
import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")


def model_judgment_corroded_ground_line(points, tree, colors, plotter):
    import gc
    gc.collect()

    def parse_args_cls():
        '''PARAMETERS'''
        parser = argparse.ArgumentParser('Testing')
        parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
        parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
        parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
        parser.add_argument('--num_category', default=2, type=int, choices=[10, 40, 2], help='training on ModelNet10/40/3')
        parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
        parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')
        parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
        parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
        return parser.parse_args()

    def load_model(model_pth_file, model_py_file, args):
        # Check if GPU is available
        device = torch.device('cpu')

        # Dynamically load model
        model = importlib.import_module(model_py_file)

        # Load model based on whether to use normal information
        if hasattr(args, 'use_normals') and args.use_normals is not None:
            classifier = model.get_model(args.num_category, normal_channel=args.use_normals)
        else:
            classifier = model.get_model(args.num_category)

        # Move model to GPU or CPU
        classifier = classifier.to(device)

        # Load model weights, ensure weights are loaded on the same device
        # Load model weights
        checkpoint = torch.load(model_pth_file, map_location=device, weights_only=False)
        classifier.load_state_dict(checkpoint['model_state_dict'])

        # Set model to evaluation mode
        classifier.eval()

        return classifier

    def model_to_judgment_cls(temp_sample_data, classifier, args):
        dealwith = []
        # Data processing
        for one_sam in temp_sample_data:
            coord_min, coord_max = np.amin(one_sam, axis=0)[:3], np.amax(one_sam, axis=0)[:3]
            one_sam[:, 0] = (one_sam[:, 0] - coord_min[0]) / (coord_max[0] - coord_min[0])
            one_sam[:, 1] = (one_sam[:, 1] - coord_min[1]) / (coord_max[1] - coord_min[1])
            one_sam[:, 2] = (one_sam[:, 2] - coord_min[2]) / (coord_max[2] - coord_min[2])
            one_sam[:, 3:] = 0
            dealwith.append(one_sam)
        temp_sample_data = np.array(dealwith).astype(np.float32)
        # Data processing, ensure can be input into the model
        if args.use_normals:
            # Note: Here we don't need special processing for temp_sample_data, because use_normals may only be a parameter affecting model input
            tensor_data = torch.from_numpy(temp_sample_data)
            sample_dataloader = DataLoader(TensorDataset(tensor_data),
                                           batch_size=args.batch_size, shuffle=True)
        else:
            # If not using normals, ensure the processing logic here is also correct (assume already correct)
            tensor_data = torch.from_numpy(temp_sample_data)
            sample_dataloader = DataLoader(TensorDataset(tensor_data),
                                           batch_size=args.batch_size, shuffle=True)

            # Start prediction
        with torch.no_grad():
            classifier = classifier.eval()
            all_pred_choices = []
            for j, (sample,) in tqdm(enumerate(sample_dataloader), total=len(sample_dataloader)):
                sample = sample.transpose(2, 1)
                pred, _ = classifier(sample.float())
                pred_choice = pred.argmax(dim=1)  # Use .argmax() instead of .data.max(1)[1]
                all_pred_choices.append(pred_choice.numpy())  # No longer need .cpu().numpy(), already on CPU

            # Merge all pred_choice into one array
            all_pred_choices = np.concatenate(all_pred_choices)

        return all_pred_choices

    def central_corrosion_fun(start_points, corrosion_max_len=6):
        # Ensure start_points is a list or array
        start_points = np.array(start_points)

        # Assume tree is already built, and points is the corresponding point set
        _, index = tree.query(start_points, k=1)
        res_index = {index}  # Use set to store visited point indices
        r = params.corrosion_r
        next_corrosion_index = {index}

        while True:
            temp_id = set()
            for row_index in next_corrosion_index:
                indices = tree.query_ball_point(points[row_index], r)
                temp_id.update(indices)
            res_index.update(next_corrosion_index)
            cur_ind = next_corrosion_index.copy()
            next_corrosion_index = temp_id - res_index
            if len(next_corrosion_index) == 0:
                r += 0.1
                next_corrosion_index = cur_ind
            else:
                r = params.corrosion_r

            # Get points
            get_points = points[list(res_index)]
            x_max, y_max, z_max = np.max(get_points, axis=0)
            x_min, y_min, z_min = np.min(get_points, axis=0)
            if x_max - x_min > corrosion_max_len or y_max - y_min > corrosion_max_len or z_max - z_min > corrosion_max_len:
                break

        # Extract sample points and colors
        sample_points = points[list(res_index)]
        sample_colors = colors[list(res_index)]

        # Sampling processing
        one_data = np.hstack((sample_points, sample_colors))
        target_num = args_cls.num_point

        # Combine up/down sampling logic optimization
        if len(one_data) > target_num:
            one_data = farthest_point_sample(one_data, target_num)
        elif len(one_data) < target_num:
            repeat_indices = np.random.choice(len(one_data), size=target_num - len(one_data), replace=True)
            one_data = np.vstack((one_data, one_data[repeat_indices]))

        return one_data

    def farthest_point_sample(point, npoint):
        N, D = point.shape
        xyz = point[:, :3]
        centroids = np.zeros((npoint,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        point = point[centroids.astype(np.int32)]
        return point

    def corrosion_top_line(clu_points, points):
        temp_top_line_set = set()
        top_end_point = []
        end_p = []
        top_line_direct = []
        _, start_ind = tree.query(clu_points, k=1)
        # Record starting point
        start_p = np.mean(points[list(start_ind)], axis=0)
        next_ind = set(start_ind)
        temp_res_index = set(next_ind)
        cur_ind = set()
        corrosion_r = params.corrosion_r
        count = 0
        while next_ind:
            temp_id = set()
            # Update next corrosion indices
            for row_index in next_ind:
                indices = tree.query_ball_point(points[row_index], corrosion_r)
                temp_id.update(set(indices))

            temp_res_index.update(next_ind)
            las_ind = cur_ind.copy()
            cur_ind = next_ind.copy()
            next_ind = temp_id - temp_res_index

            # Determine if there is discontinuity
            if len(next_ind) < 3 and corrosion_r < 2 * params.corrosion_r and count > 10:
                next_ind = set(cur_ind)
                corrosion_r = corrosion_r + 0.1
                continue
            elif len(next_ind) < 3 and corrosion_r < 2 * params.corrosion_r and count < 10:
                break
            elif corrosion_r > 2 * params.corrosion_r:
                break
            else:
                corrosion_r = params.corrosion_r

            # Add branching logic: If corrosion splits into two categories, corrode separately
            if len(next_ind) != len(cur_ind):
                # Re-run DBSCAN clustering to determine if branching occurred
                cluster_labels_split = DBSCAN(eps=params.corrosion_r, min_samples=3).fit_predict(
                    points[list(next_ind)])
                unique_labels_split = np.unique(cluster_labels_split)
                unique_labels_split = unique_labels_split[unique_labels_split != -1]
                is_fencha = 0
                in_count = [0, 0]
                in_temp_line = set()
                if len(unique_labels_split) == 2:
                    # There is branching, perform independent corrosion
                    for sub_label in unique_labels_split:
                        sub_indices = np.where(cluster_labels_split == sub_label)[0]
                        sub_set = set(np.array(list(next_ind))[sub_indices])
                        in_las_sub_set = set()
                        while sub_set:
                            temp_id = set()
                            # Update next corrosion indices
                            for row_index in sub_set:
                                indices = tree.query_ball_point(points[row_index], corrosion_r)
                                temp_id.update(set(indices))
                            in_las_sub_set = sub_set.copy()
                            in_temp_line.update(sub_set)
                            sub_set = temp_id - in_temp_line - temp_res_index
                            if len(sub_set) < 2:
                                end_p.append(np.mean(points[list(sub_set)], axis=0))
                                break
                            # Perform DBSCAN clustering on current corrosion area to detect branching
                            split_labels = DBSCAN(eps=params.corrosion_r, min_samples=2).fit_predict(
                                points[list(sub_set)])
                            unique_split_labels = np.unique(split_labels)
                            unique_split_labels = unique_split_labels[unique_split_labels != -1]  # Remove noise points
                            if len(unique_split_labels) > 1:
                                end_p.append(np.mean(points[list(in_las_sub_set)], axis=0))
                                break

                            # Detect density
                            # Calculate number of points in range, calculate volume
                            x_max, y_max, z_max = np.max(points[list(sub_set)], axis=0)
                            x_min, y_min, z_min = np.min(points[list(sub_set)], axis=0)
                            v_xyz = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
                            if (v_xyz > 0.25 if math.pow(params.corrosion_r, 3) < 0.25 else math.pow(params.corrosion_r, 3)):
                                end_p.append(np.mean(points[list(in_las_sub_set)], axis=0))
                                break

                            in_count[sub_label] += 1
                    if in_count[0] > 10 and in_count[1] > 10:
                        is_fencha = 1
                        temp_top_line_set.update(temp_res_index)
                        temp_top_line_set.update(in_temp_line)
                        # Here calculate the direction of the power line
                        vec = start_p - np.mean(points[list(cur_ind)], axis=0)
                        vec = vec / np.linalg.norm(vec)
                        top_line_direct = vec
                if is_fencha == 1:
                    break
                else:
                    end_p = []

            # Calculate number of points in range, calculate volume
            x_max, y_max, z_max = np.max(points[list(next_ind)], axis=0)
            x_min, y_min, z_min = np.min(points[list(next_ind)], axis=0)

            # Check if there are points above
            up_index = np.where((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                                (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
                                (points[:, 2] >= z_max + 2) & (points[:, 2] <= z_max + 3))[0]
            if len(up_index) > 10 and count > 10:
                end_p = np.mean(points[list(las_ind)], axis=0)
                if np.array_equal(end_p, start_p) or np.all(np.isnan(end_p)):
                    end_p = []
                    break
                # temp_res_index.difference_update(cur_ind.union(las_ind))
                temp_top_line_set.update(temp_res_index.union(next_ind))
                # Here calculate the direction of the power line
                vec = start_p - end_p
                vec = vec / np.linalg.norm(vec)
                top_line_direct = vec
                break
            # Calculate number of points in range, calculate volume
            v_xyz = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
            if (v_xyz > 0.5 if math.pow(params.corrosion_r, 3) < 0.5 else math.pow(params.corrosion_r, 3)) and count > 10:
                end_p = np.mean(points[list(las_ind)], axis=0)
                if np.array_equal(end_p, start_p) or np.all(np.isnan(end_p)):
                    end_p = []
                    break
                # temp_res_index.difference_update(cur_ind.union(las_ind))
                temp_top_line_set.update(temp_res_index)
                # Here calculate the direction of the power line
                vec = start_p - end_p
                vec = vec / np.linalg.norm(vec)
                top_line_direct = vec
                break
            # Here interrupt the ground for judgment error
            elif (v_xyz > 0.5 if math.pow(params.corrosion_r, 3) < 0.5 else math.pow(params.corrosion_r, 3)) and count > 2 and count <= 10:
                end_p = []
                break
            count += 1

        # Stop point
        if len(end_p) > 0:
            top_end_point = end_p

        return temp_top_line_set, top_line_direct, top_end_point

    def angle_between(v1, v2):
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    # Get some built-in parameters of the model
    args_cls = parse_args_cls()
    # Get some predefined parameters
    params = Parameters()
    # Model path, load model
    model_pth_file_cls = "best_model_distance_6.pth"
    model_py_file_cls = "pointnet_cls"
    classifier_cls = load_model(model_pth_file_cls, model_py_file_cls, args_cls)
    x_max, y_max, z_max = np.max(points, axis=0)
    x_min, y_min, z_min = np.min(points, axis=0)
    bound_indexs = np.where(
        ((points[:, 0] >= x_min) & (points[:, 0] <= x_min + 2) | (points[:, 0] >= x_max - 2) & (points[:, 0] <= x_max)) |
        ((points[:, 1] >= y_min) & (points[:, 1] <= y_min + 2) | (points[:, 1] >= y_max - 2) & (points[:, 1] <= y_max))
    )[0]
    # plotter.add_mesh(points[list(bound_indexs)], color='blue', point_size=8, render_points_as_spheres=True)
    # plotter.update()
    # Randomly select 100,000 elements from bound_indexs (if possible)
    # Calculate the number to select
    # num_to_select = int(len(bound_indexs) * 0.7)
    # # If array length is greater than or equal to the number to select, randomly select 80% of elements
    # bound_indexs = np.random.choice(bound_indexs, size=num_to_select, replace=False)
    # Randomly select 100,000 elements from bound_indexs (if possible)
    num_to_select = 200000
    if len(bound_indexs) >= num_to_select:
        bound_indexs = np.random.choice(bound_indexs, size=num_to_select, replace=False)

    dbscan = DBSCAN(eps=2, min_samples=2)
    cluster_labels = dbscan.fit_predict(points[bound_indexs])
    # Get unique cluster labels
    unique_labels = np.unique(cluster_labels[cluster_labels != -1])

    # # Get number of clusters
    # num_clusters = len(unique_labels)
    # # Create color map (generate enough colors based on number of clusters)
    # cmap = plt.get_cmap('tab20')  # 'tab20' is a color map with 20 colors
    # colors = [cmap(i / num_clusters) for i in range(num_clusters)]  # Generate one color for each cluster
    # # Select a color for each cluster category and visualize
    # for i, label in enumerate(unique_labels):
    #     # Get point indices under this cluster label
    #     cluster_points = points[bound_indexs][cluster_labels == label]
    #     # Use generated color
    #     color = colors[i]
    #     # Add point cloud of this class on plotter, using different colors
    #     plotter.add_mesh(cluster_points, color=color, point_size=10, render_points_as_spheres=True)
    # # Update display
    # plotter.update()

    # Define a variable to save power line
    line_set = set()
    yes_line_class = []
    background = set()
    # Loop through each category to determine if it's a power line
    a_1, a_2 = set(), set()
    # Determine if it's power line or external environment
    for label in unique_labels:
        # Get current cluster indices
        ind = np.where(cluster_labels == label)[0]
        random_indices = np.random.choice(ind, 3, replace=True)
        outer_indices = bound_indexs[random_indices]
        # Get sample data for each category
        sample_data = np.array([central_corrosion_fun(np.array(points[idx])) for idx in outer_indices])
        preds = model_to_judgment_cls(sample_data.copy(), classifier_cls, args_cls)
        if mode(preds)[0] == 0:  # Power line
            a_1.update(set(bound_indexs[ind]))
            line_set.update(set(bound_indexs[ind]))
            yes_line_class.append(label)
        if mode(preds)[0] == 1:  # Tower
            background.update(set(bound_indexs[ind]))
            a_2.update(set(bound_indexs[ind]))
    # plotter.add_mesh(points[list(line_set)], color='green', point_size=8, render_points_as_spheres=True)
    # plotter.add_mesh(points[list(background)], color='blue', point_size=12, render_points_as_spheres=True)
    # plotter.update()
    # time.sleep(1)

    # Define function to process single cluster
    from concurrent.futures import ThreadPoolExecutor
    def process_cluster(clu_lab):
        this_ind = np.where(cluster_labels == clu_lab)[0]
        temp_top_line_set, temp_top_line_direct, temp_top_line_end = corrosion_top_line(points[bound_indexs[this_ind]], points)
        return list(temp_top_line_set), temp_top_line_direct, temp_top_line_end
    # Parallel processing
    top_line_set, top_line_direct, top_line_end = [], [], []
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_cluster, yes_line_class)
        # Collect results
        for temp_top_line_set, temp_top_line_direct, temp_top_line_end in results:
            top_line_set.append(temp_top_line_set)
            top_line_direct.append(temp_top_line_direct)
            top_line_end.append(temp_top_line_end)

    top_line_set = [item for sublist in top_line_set for item in sublist]
    top_line_end = np.vstack([item for item in top_line_end if len(item) > 0])
    top_line_direct = np.vstack([item for item in top_line_direct if len(item) > 0])
    top_line_end = top_line_end[~np.isnan(top_line_end).any(axis=1)]

    # Calculate maximum lateral distance
    max_hengxiang_dist = np.max(pdist(top_line_end[:, :2]))

    tower_central_point = np.mean(top_line_end, axis=0)
    tower_central_point[2] = np.max(top_line_end[:, 2])
    # for xx in top_line_direct:
    #     l_line = pv.Line(tower_central_point, tower_central_point + 20 * xx)
    #     plotter.add_mesh(l_line, color='pink', line_width=5, label='Same Direction')
    # plotter.update()

    # plotter.add_mesh(points[list(top_line_set)], color='green', point_size=8, render_points_as_spheres=True)
    # plotter.add_mesh(top_line_end, color='red', point_size=10, render_points_as_spheres=True)
    # plotter.update()

    # Calculate the two directions of the power line through the four vectors of top_line_direct
    # Calculate new directions
    temp_top_line_direct = list(top_line_direct.copy())
    new_top_line_direct = []
    count_smal = []
    while temp_top_line_direct:
        dir = temp_top_line_direct.pop(0)
        similar_dirs = [dir]
        # Use np.allclose() instead of np.array_equal for approximate judgment
        for other_dir in temp_top_line_direct[:]:
            if angle_between(dir, other_dir) < np.deg2rad(10):  # Angle less than 10 degrees
                similar_dirs.append(other_dir)
                # Use np.allclose() to avoid precision issues
                temp_top_line_direct = [arr for arr in temp_top_line_direct if not np.allclose(arr, other_dir)]
        # Calculate average of similar vectors and normalize
        count_smal.append(len(similar_dirs))
        avg_dir = np.mean(similar_dirs, axis=0)
        avg_dir = avg_dir / np.linalg.norm(avg_dir)
        new_top_line_direct.append(avg_dir)
    if len(count_smal) > 2:
        max_two_indices = np.argsort(count_smal)[::-1][:2]  # Sort from large to small
        new_top_line_direct = np.array(new_top_line_direct)[max_two_indices, :]

    # If only one direction is obtained, add another direction
    if len(new_top_line_direct) == 1:
        new_top_line_direct.append(-new_top_line_direct[0])
    # If the angle between the two directions is less than 30 degrees, correct one direction
    if angle_between(new_top_line_direct[0], new_top_line_direct[1]) < np.deg2rad(30):
        new_top_line_direct[1] = -new_top_line_direct[0]
    top_line_direct = new_top_line_direct.copy()

    # Filter the top power lines, method is to cluster into two categories, then filter the two highest categories in each category, if the height difference of power lines in the category is too large (3m), record only one
    # Get the highest 4 points
    # Remove rows containing NaN
    max_0 = top_line_end[top_line_end[:, 2].argsort()[-4:]][::-1]
    # Remove points with too large height difference
    max_z = np.max(max_0[:, 2])
    # Filter points where z-axis value differs from maximum by less than 2

    if len(max_0[np.abs(max_0[:, 2] - max_z) < 2]) < 2:
        print("Power lines not accurately identified")
    elif len(max_0[np.abs(max_0[:, 2] - max_z) < 2]) > 2:
        max_0 = max_0[np.abs(max_0[:, 2] - max_z) < 2]
    # Select the farthest two points as lateral vectors
    # Use K-means clustering, cluster into two categories, but only based on x0y clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(max_0[:, :2])
    # Get cluster centers
    centroids = kmeans.cluster_centers_
    hengxiang_vec = centroids[0] - centroids[1]
    hengxiang_vec = np.array((hengxiang_vec[0], hengxiang_vec[1], 0))
    # Record center point coordinates
    tower_central_point = np.mean(max_0, axis=0)
    tower_central_point[2] = np.max(max_0[:, 2])

    # Add a judgment here, purpose is for the topmost points
    x_min, y_min, z_min = np.min(max_0, axis=0)
    x_max, y_max, z_max = np.max(max_0, axis=0)
    is_have_points = np.where((points[:, 0] >= x_min - 1.5) & (points[:, 0] <= x_max + 1.5) &
                              (points[:, 1] >= y_min - 1.5) & (points[:, 1] <= y_max + 1.5) &
                              (points[:, 2] >= z_max + 2))[0]
    filtered_points = points[is_have_points]
    if len(is_have_points) > 10:
        labels = DBSCAN(eps=0.2, min_samples=15).fit_predict(filtered_points)
        majority_points = filtered_points[labels == np.argmax(np.bincount(labels[labels >= 0]))]
        highest_point = majority_points[np.argmax(majority_points[:, 2])] if len(majority_points) > 0 else None
        highest_point[2] = highest_point[2] - 0.2
        print("There are points above")
        new_point_left, new_point_right = highest_point - 0.25 * hengxiang_vec, highest_point + 0.25 * hengxiang_vec
        two_points = np.vstack((new_point_left, new_point_right))
        tower_central_point = np.mean(two_points, axis=0)

    # Correct lateral vector
    if angle_between(np.array((top_line_direct[0][0], top_line_direct[0][1], 0)), hengxiang_vec) < np.deg2rad(60)\
            or angle_between(-np.array((top_line_direct[0][0], top_line_direct[0][1], 0)), hengxiang_vec) < np.deg2rad(60)\
            or angle_between(np.array((top_line_direct[1][0], top_line_direct[1][1], 0)), hengxiang_vec) < np.deg2rad(60)\
            or angle_between(-np.array((top_line_direct[1][0], top_line_direct[1][1], 0)), hengxiang_vec) < np.deg2rad(60):
        hengxiang_dist = np.linalg.norm(hengxiang_vec)
        if top_line_direct[0][0] + top_line_direct[1][0] < 0.01:
            hengxiang_vec = np.array((-top_line_direct[0][1], top_line_direct[0][0], 0))
        else:
            hengxiang_vec = top_line_direct[0] + top_line_direct[1]
        hengxiang_vec[2] = 0
        hengxiang_vec = hengxiang_dist * hengxiang_vec / np.linalg.norm(hengxiang_vec)

    # Return are: power line set, top power line set, ground line distance
    return tower_central_point, hengxiang_vec, max_hengxiang_dist, set(top_line_set), top_line_direct


import concurrent.futures
def template_matching(template_library, template_insulator, template_direction, points, tree, tower_central_point, hengxiang_vec, top_line_set, top_line_direct, max_hengxiang_dist, plotter):
    def farthest_point_sample(point, npoint):

        N, D = point.shape
        xyz = point[:, :3]
        centroids = np.zeros((npoint,), dtype=int)
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)
        return point[centroids]

    def chamfer_distance(pc1, pc2):

        tree1 = cKDTree(pc1)
        tree2 = cKDTree(pc2)
        dist_pc1_to_pc2, _ = tree1.query(pc2)
        dist_pc2_to_pc1, _ = tree2.query(pc1)
        return dist_pc1_to_pc2.mean() + dist_pc2_to_pc1.mean()

    def fitness_function(solution, template_library, this_tower_points):
        all_loss = []
        all_scaled_rotated_template = []
        for i_t in range(len(template_library)):
            selected_template = template_library[i_t].copy()
            selected_template[:, :2] = selected_template[:, :2] * (1 - solution[4]) 
            selected_template[:, 2] = selected_template[:, 2] * (1 - solution[5]) 


            theta = np.radians(-solution[0])
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            scaled_rotated_template = np.dot(selected_template[:, :3], rotation_matrix.T) + solution[1:4]
            all_scaled_rotated_template.append(scaled_rotated_template)
            new_sample = this_tower_points[i_t]
            this_loss = chamfer_distance(scaled_rotated_template, new_sample)
            all_loss.append(this_loss)

        min_loss, min_index = min((val, idx) for idx, val in enumerate(all_loss))
        return min_loss, min_index, all_scaled_rotated_template[min_index]

    def parallel_fitness(pop, template_library, this_tower_points):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            temp_data = list(
                executor.map(lambda p: list(fitness_function(p, template_library, this_tower_points)), pop))
        return temp_data

    def pso(this_tower_points, lower_bounds, upper_bounds, step_limits, template_library, num_p=50, max_iter=50, w=0.3, c1=2,
            c2=2):
        dim = len(lower_bounds)
        pop = np.random.uniform(lower_bounds, upper_bounds, (num_p, dim))
        vel = np.random.uniform(-abs(upper_bounds - lower_bounds), abs(upper_bounds - lower_bounds), (num_p, dim))

        pbest_position = pop.copy()
        temp_data = parallel_fitness(pop, template_library, this_tower_points)


        pbest_fitness, res_index, res_sample = [], [], []
        for item in temp_data:
            pbest_fitness.append(item[0])
            res_index.append(item[1])
            res_sample.append(item[2])
        pbest_fitness, res_index, res_sample = np.array(pbest_fitness), np.array(res_index), np.array(res_sample)

        gbest_position = pop[np.argmin(pbest_fitness)]
        gbest_fitness = np.min(pbest_fitness)
        gbest_index = res_index[np.argmin(pbest_fitness)]
        gbest_sample = res_sample[np.argmin(pbest_fitness)]

        for iteration in range(max_iter):
            r1, r2 = np.random.rand(num_p, dim), np.random.rand(num_p, dim)
            vel = (w * vel + c1 * r1 * (pbest_position - pop) + c2 * r2 * (gbest_position - pop))
            pop = np.clip(pop + vel, lower_bounds, upper_bounds)
            if np.random.rand() > 0.5:
                pop += np.random.uniform(-np.array(step_limits) / 2, np.array(step_limits) / 2, (num_p, dim))  
                vel += np.random.uniform(-np.array(step_limits) / 10, np.array(step_limits) / 10, (num_p, dim)) 

            temp_data = parallel_fitness(pop, template_library, this_tower_points)
            fitness_vals, res_index, res_sample = [], [], []
            for item in temp_data:
                fitness_vals.append(item[0])
                res_index.append(item[1])
                res_sample.append(item[2])
            fitness_vals, res_index, res_sample = np.array(fitness_vals), np.array(res_index), np.array(res_sample)


            improved = fitness_vals < pbest_fitness
            pbest_position[improved] = pop[improved]
            pbest_fitness[improved] = fitness_vals[improved]

            min_fitness_idx = np.argmin(fitness_vals)
            if fitness_vals[min_fitness_idx] < gbest_fitness:
                gbest_fitness = fitness_vals[min_fitness_idx]
                gbest_index = res_index[min_fitness_idx]
                gbest_sample = res_sample[min_fitness_idx]
                gbest_position = pop[min_fitness_idx]
            print(f"{iteration + 1}/{max_iter}, bestfitness:{gbest_fitness}")

        return gbest_position, gbest_fitness, gbest_index, gbest_sample


    params = Parameters()
    half_len = max_hengxiang_dist / 2
    local_x_min, local_y_min = tower_central_point[0] - half_len, tower_central_point[1] - half_len
    local_x_max, local_y_max = tower_central_point[0] + half_len, tower_central_point[1] + half_len
    this_tower_points = []


    for ku in range(len(template_library)):
        ku_z_min, ku_z_max = min(template_library[ku][:, 2]), max(template_library[ku][:, 2])

        local_index = np.where(
            (points[:, 0] >= local_x_min - 5) & (points[:, 0] <= local_x_max + 5) &
            (points[:, 1] >= local_y_min - 5) & (points[:, 1] <= local_y_max + 5) &
            (points[:, 2] >= tower_central_point[2] - (ku_z_max - ku_z_min)) &
            (points[:, 2] <= tower_central_point[2] + 5)
        )[0]

        if len(local_index) > 20000:
            local_index = np.random.choice(local_index, 20000, replace=False)

        local_points = points[local_index]

        local_labels = DBSCAN(eps=1, min_samples=20).fit_predict(local_points)

        t_distances = np.linalg.norm(local_points - tower_central_point, axis=1)

        closest_point_index = np.argmin(t_distances)

        closest_label = local_labels[closest_point_index]

        indices_with_closest_label = np.where(local_labels == closest_label)[0]

        final_indices = local_index[indices_with_closest_label]
        final_indices = list(set(final_indices) - set(top_line_set))

        temp_this_p = farthest_point_sample(points[final_indices], template_library[0].shape[0])

        this_tower_points.append(temp_this_p)

    this_tower_points = np.array(this_tower_points)


    angle_to_rotate = np.degrees(np.arccos(np.dot([1, 0], hengxiang_vec[:2]) / np.linalg.norm(hengxiang_vec[:2])))
    if hengxiang_vec[1] > 0:
        angle_to_rotate = 360 - angle_to_rotate

    ref_solution = np.hstack([angle_to_rotate, tower_central_point, 0, 0])  # 角度，xyz，xoy缩放因子，高度缩放因子
    ref_solution_rev = ref_solution.copy()
    if ref_solution_rev[0] > 180:
        ref_solution_rev[0] -= 180
    else:
        ref_solution_rev[0] += 180


    step_limits = [30, 3, 3, 1, 0.05, 0.05]


    lower_bounds_1 = ref_solution - step_limits
    upper_bounds_1 = ref_solution + step_limits
    gbest_position_1, gbest_fitness_1, gbest_index_1, gbest_sample_1 = pso(this_tower_points, lower_bounds_1,
                                                                           upper_bounds_1, step_limits, template_library)


    lower_bounds_2 = ref_solution_rev - step_limits
    upper_bounds_2 = ref_solution_rev + step_limits
    gbest_position_2, gbest_fitness_2, gbest_index_2, gbest_sample_2 = pso(this_tower_points, lower_bounds_2,
                                                                               upper_bounds_2, step_limits, template_library)


    if gbest_fitness_1 <= gbest_fitness_2:
        gbest_solution, gbest_index = gbest_position_1, gbest_index_1
    else:
        gbest_solution, gbest_index = gbest_position_2, gbest_index_2


    theta = np.radians(-gbest_solution[0])
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    match_sample = np.dot(template_library[gbest_index][:, :3], rotation_matrix.T) + gbest_solution[1:4]
    match_insulator = np.dot(template_insulator[gbest_index][:, :3], rotation_matrix.T) + gbest_solution[1:4]
    match_direction = template_direction[gbest_index]


    all_tower_points_indices = tree.query_ball_point(match_sample, params.corrosion_r)

    all_tower_set = {idx for indices in all_tower_points_indices for idx in indices}

    return gbest_solution, gbest_index, all_tower_set, match_sample, match_insulator, match_direction

def insulator_correction(points, tree, top_line_direct, match_direction, match_insulator, plotter):
    def vector_closest_to_AB(AB, O1, O2):
        dot_AB_O1 = np.dot(AB[:2], O1[:2])
        mag_AB = np.linalg.norm(AB[:2])
        mag_O1 = np.linalg.norm(O1[:2])
        cos_theta_AB_O1 = dot_AB_O1 / (mag_AB * mag_O1)

        dot_AB_O2 = np.dot(AB[:2], O2[:2])
        mag_O2 = np.linalg.norm(O2[:2])
        cos_theta_AB_O2 = dot_AB_O2 / (mag_AB * mag_O2)

        if cos_theta_AB_O1 > cos_theta_AB_O2:
            return O1
        else:
            return O2

    def find_points_between_planes(new_A, direction_vector, new_C, points):
        new_A = np.array(new_A)
        direction_vector = np.array(direction_vector)
        new_C = np.array(new_C)
        points = np.array(points)

        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        distance_AC = np.dot(new_C - new_A, direction_vector)


        distances_to_plane1 = np.dot(points - new_A, direction_vector)

        distances_to_plane2 = np.dot(points - new_C, direction_vector)


        between_planes = (np.abs(distances_to_plane1) < distance_AC) & (np.abs(distances_to_plane2) < distance_AC)


        return np.where(between_planes)[0]


    reshaped_array = match_insulator.reshape((len(match_direction), 3, 3))
    params = Parameters()

    insulator_set = set()

    for i in range(len(match_direction)):
        if match_direction[i] == 0:  

            this_insulator = reshaped_array[i].copy()
            A, C = this_insulator[0, :], this_insulator[2, :] 
            AB = C - A
            AB_dist = np.linalg.norm(AB)
            rotating_reference_vector = vector_closest_to_AB(AB, top_line_direct[0], top_line_direct[1])

            B = A + rotating_reference_vector * AB_dist
            cen_AB = (A + B) / 2

            reshaped_array[i] = np.vstack((A, cen_AB, B))
            reshaped_array[i][0] = np.array(A)
            reshaped_array[i][1] = np.array(cen_AB)
            reshaped_array[i][2] = np.array(B)

            this_insulator = reshaped_array[i].copy() 
            A, C = this_insulator[0, :], this_insulator[2, :]
            this_insulator = reshaped_array[i].copy()
            vec = (C - A) / np.linalg.norm(C - A) 
            d = np.linalg.norm(C - A)  
            B = this_insulator[1]

            this_ind = find_points_between_planes(B, vec, B + params.corrosion_r * vec, points)
            temp_tree = cKDTree(points[this_ind])
            this_dist, b = temp_tree.query(B, k=3)
            if max(this_dist) > params.find_radius_template_point:
                continue  
            else:

                C_ind = find_points_between_planes(C, vec, C + params.corrosion_r * vec, points)
                temp_C_tree = cKDTree(points[C_ind])
                _, ind_ = temp_C_tree.query(C, k=3)
                a = tree.query_ball_point(points[C_ind[ind_]], params.corrosion_r)
                b = {item for sublist in a for item in sublist}


                cor_1_index = tree.query_ball_point(points[list(b)], params.corrosion_r)
                cor_1_index = {item for sublist in cor_1_index for item in sublist}
                new_C = np.mean(points[list(cor_1_index)], axis=0)


                new_A = new_C - d * vec
                new_A[2] = A[2]
                _, A_ind = tree.query(new_A, k=3)
                new_A = np.mean(points[A_ind], axis=0)


                top_direct = top_line_direct[0] - top_line_direct[1]
                if np.linalg.norm(top_direct) < 2:
                    top_direct = np.array([top_direct[1], -top_direct[0], top_direct[2]])
                top_dist = np.linalg.norm(top_direct) / 2
                num_points = int(np.ceil(top_dist / 0.1))
                top_direct /= top_dist


                while True:
                    count = 0
                    for n_i in range(num_points):
                        temp_points_1 = new_A + (n_i / (num_points - 1)) * top_direct
                        temp_points_2 = new_A - (n_i / (num_points - 1)) * top_direct
                        if len(tree.query_ball_point(temp_points_1, 0.1)) > 0 or len(
                                tree.query_ball_point(temp_points_2, 0.1)) > 0:
                            count += 1
                    if count >= 1.5 * num_points / 5:
                        new_A += 0.05 * vec
                        new_C += 0.05 * vec
                    else:
                        break


                direction_vector = new_C - new_A
                A_C_ind = find_points_between_planes(new_A, direction_vector, new_C, points)


                distances = np.linalg.norm(np.cross(points[A_C_ind] - new_A, direction_vector),
                                           axis=1) / np.linalg.norm(direction_vector)


                temp_insulator = A_C_ind[np.where(distances <= params.corrosion_r)[0]]


                insulator_set.update(set(temp_insulator))
                new_B = (new_A + new_C) / 2


                reshaped_array[i][0] = np.array(new_A)
                reshaped_array[i][1] = np.array(new_B)
                reshaped_array[i][2] = np.array(new_C)

        elif match_direction[i] == 1: 
            print("此功能未开放")
        elif match_direction[i] == 2:  
            this_insulator = reshaped_array[i].copy()  
            central_B = this_insulator[1, :]  

            this_ind = np.where(
                (points[:, 0] >= central_B[0] - params.find_radius_template_point) & (
                            points[:, 0] <= central_B[0] + params.find_radius_template_point) &  
                (points[:, 1] >= central_B[1] - params.find_radius_template_point) & (
                            points[:, 1] <= central_B[1] + params.find_radius_template_point) &  
                (points[:, 2] >= central_B[2] - params.corrosion_r) & (
                            points[:, 2] <= central_B[2] + params.corrosion_r) 
            )[0]

            if len(this_ind) == 0:
                print("附近没有点，直接采用模板中的点位")
            else:
                if np.linalg.norm(top_line_direct[0][:2] - top_line_direct[1][:2]) < 2:
                    new_B = (this_insulator[0, :] + this_insulator[2, :]) / 2
                    t_point_tree = cKDTree(points[this_ind])
                    _, t_ind = t_point_tree.query(new_B, k=1)
                    new_B = np.array(points[this_ind])[t_ind]
                    new_A = np.array([new_B[0], new_B[1], this_insulator[0][2]])
                    new_C = np.array([new_B[0], new_B[1], this_insulator[2][2]])
                else:
                    AB = top_line_direct[0][:2] - top_line_direct[1][:2]
                    AC = this_insulator[0][:2] - top_line_direct[1][:2]
                    proj_AB_AC = (np.dot(AC, AB) / np.dot(AB, AB)) * AB
                    temp_xy = top_line_direct[1][:2] + proj_AB_AC
                    temp_xy = (temp_xy + this_insulator[0][:2]) / 2

                    new_A = np.array([temp_xy[0], temp_xy[1], this_insulator[0][2]])
                    new_C = np.array([temp_xy[0], temp_xy[1], this_insulator[2][2]])

                    t_point_tree = cKDTree(points[this_ind])
                    _, t_ind = t_point_tree.query((new_A + new_C) / 2, k=1)
                    new_B = np.array(points[this_ind])[t_ind]
                    new_A = np.array([new_B[0], new_B[1], this_insulator[0][2]])
                    new_C = np.array([new_B[0], new_B[1], this_insulator[2][2]])

                count = 0
                while True:
                    w_ind = tree.query_ball_point(new_C, params.corrosion_r)
                    if len(w_ind) == 0:
                        new_A[2] += 0.1
                        new_C[2] += 0.1
                        count += 1
                    elif count > params.corrosion_r * 10 or len(w_ind) > 0:
                        if np.mean(points[w_ind], axis=0)[2] > new_C[2]:
                            new_A[2] += (np.mean(points[w_ind], axis=0)[2] - new_C[2])
                            new_C[2] += (np.mean(points[w_ind], axis=0)[2] - new_C[2])
                        break

                heng_central = np.mean(top_line_direct, axis=0)
                heng_vec = new_A - heng_central
                heng_vec[2] = 0
                heng_dist = np.linalg.norm(heng_vec)
                heng_vec /= heng_dist
                heng_range_index = find_points_between_planes(
                    heng_central + (heng_dist - params.corrosion_r) * heng_vec, heng_vec,
                    heng_central + (heng_dist + params.corrosion_r) * heng_vec, points)

                vector_to_A = points[heng_range_index] - new_A
                vector_to_C = points[heng_range_index] - new_C
                direction_vector = new_C - new_A
                direction_vector_norm = np.linalg.norm(direction_vector)
                if direction_vector_norm == 0:
                    raise ValueError("error")
                projection_ratio = np.clip(np.dot(vector_to_A, direction_vector) / direction_vector_norm ** 2, 0, 1)
                projected_points = new_A + projection_ratio[:, np.newaxis] * direction_vector

                distances_to_projected = np.linalg.norm(points[heng_range_index] - projected_points, axis=1)
                distances_to_endpoints = np.minimum(np.linalg.norm(points[heng_range_index] - new_A, axis=1),
                                                    np.linalg.norm(points[heng_range_index] - new_C, axis=1))
                distances = np.minimum(distances_to_projected, distances_to_endpoints)

                temp_insulator = heng_range_index[np.where(distances <= params.find_insulator_r)[0]]
                temp_insulator = temp_insulator[np.where(points[temp_insulator, 2] < new_A[2])]

                insulator_set.update(set(temp_insulator))

                new_C[2] -= params.corrosion_r / 2
                new_A[2] -= params.corrosion_r / 3
                new_B = (new_A + new_C) / 2

                new_xy = np.mean(np.array([new_A, new_B, new_C])[:, :2], axis=0)
                new_A[:2], new_B[:2], new_C[:2] = new_xy, new_xy, new_xy

                reshaped_array[i][0] = np.array(new_A)
                reshaped_array[i][1] = np.array(new_B)
                reshaped_array[i][2] = np.array(new_C)
    return np.vstack(reshaped_array), insulator_set


